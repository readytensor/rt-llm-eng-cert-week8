import time
import torch
import statistics
import threading
from fastapi import FastAPI
from peft import PeftModel
from pydantic import BaseModel
from fastapi.responses import StreamingResponse
from concurrent.futures import ThreadPoolExecutor
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer

from peft import PeftModel


BASE_MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
LORA_ADAPTER_ID = "moo3030/Llama-3.2-1B-QLoRA-Summarizer-adapters"


DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

# --------------------
# Load tokenizer
# --------------------
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# --------------------
# Load base model + LoRA
# --------------------
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    torch_dtype=DTYPE,
    device_map="auto" if DEVICE == "cuda" else None,
)

model = PeftModel.from_pretrained(base_model, LORA_ADAPTER_ID)
model.eval()

# --------------------
# FastAPI app
# --------------------
app = FastAPI(title="LLM FastAPI Server", version="1.0")


class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True


class GenerateResponse(BaseModel):
    response: str


@app.post("/generate", response_model=GenerateResponse)
@torch.no_grad()
def generate(req: GenerateRequest):
    inputs = tokenizer(
        req.prompt,
        return_tensors="pt",
        padding=True,
    )

    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    output_ids = model.generate(
        **inputs,
        max_new_tokens=req.max_new_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
        do_sample=req.do_sample,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    generated_text = tokenizer.decode(
        output_ids[0],
        skip_special_tokens=True,
    )

    return GenerateResponse(response=generated_text)


@app.post("/generate_stream")
def generate_stream(req: GenerateRequest):

    def token_stream():
        inputs = tokenizer(
            req.prompt,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        streamer = TextIteratorStreamer(
            tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        generation_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            do_sample=req.do_sample,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

        thread = threading.Thread(
            target=model.generate,
            kwargs=generation_kwargs,
        )
        thread.start()

        for token in streamer:
            yield token

    return StreamingResponse(
        token_stream(),
        media_type="text/plain",
    )


# --------------------
# Request / Response
# --------------------
class TTFTITLRequest(BaseModel):
    input_tokens: int
    generated_tokens: int
    num_prompts: int
    batch_size: int


class LatencyStats(BaseModel):
    mean_ms: float
    median_ms: float
    p99_ms: float


class TTFTITLReport(BaseModel):
    ttft: LatencyStats
    tpot: LatencyStats
    itl: LatencyStats
    num_prompts: int
    batch_size: int
    num_batches: int


# --------------------
# Single-run measurement
# --------------------
def _measure_once(input_tokens: int, generated_tokens: int):
    input_ids = torch.randint(
        low=0,
        high=tokenizer.vocab_size,
        size=(1, input_tokens),
        device=model.device,
    )

    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
    )

    generation_kwargs = dict(
        input_ids=input_ids,
        streamer=streamer,
        max_new_tokens=generated_tokens,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    start_time = time.perf_counter()
    first_token_time = None
    token_times = []

    def run_generation():
        model.generate(**generation_kwargs)

    thread = threading.Thread(target=run_generation)
    thread.start()

    for _ in streamer:
        now = time.perf_counter()
        if first_token_time is None:
            first_token_time = now
        token_times.append(now)

    end_time = token_times[-1]

    ttft_ms = (first_token_time - start_time) * 1000
    total_gen_ms = (end_time - start_time) * 1000

    # TPOT = (total generation time - TTFT) / (tokens - 1)
    if generated_tokens > 1:
        tpot_ms = (total_gen_ms - ttft_ms) / (generated_tokens - 1)
    else:
        tpot_ms = 0.0

    # ITL samples (true inter-token latency)
    itl_samples = [
        (token_times[i] - token_times[i - 1]) * 1000 for i in range(1, len(token_times))
    ]

    return ttft_ms, tpot_ms, itl_samples


# --------------------
# Helpers
# --------------------
def _stats(values: list[float]) -> LatencyStats:
    if not values:
        return LatencyStats(mean_ms=0.0, median_ms=0.0, p99_ms=0.0)

    values = sorted(values)
    return LatencyStats(
        mean_ms=round(statistics.mean(values), 2),
        median_ms=round(statistics.median(values), 2),
        p99_ms=round(values[int(0.99 * (len(values) - 1))], 2),
    )


# --------------------
# Batched endpoint
# --------------------
@app.post("/ttft_itl_batched", response_model=TTFTITLReport)
def measure_ttft_itl_batched(req: TTFTITLRequest):

    num_batches = (req.num_prompts + req.batch_size - 1) // req.batch_size

    ttft_samples = []
    tpot_samples = []
    itl_samples = []

    remaining = req.num_prompts

    for _ in range(num_batches):
        current_batch = min(req.batch_size, remaining)

        with ThreadPoolExecutor(max_workers=current_batch) as executor:
            futures = [
                executor.submit(
                    _measure_once,
                    req.input_tokens,
                    req.generated_tokens,
                )
                for _ in range(current_batch)
            ]

            for f in futures:
                ttft, tpot, itl = f.result()
                ttft_samples.append(ttft)
                tpot_samples.append(tpot)
                itl_samples.extend(itl)

        remaining -= current_batch

    return TTFTITLReport(
        ttft=_stats(ttft_samples),
        tpot=_stats(tpot_samples),
        itl=_stats(itl_samples),
        num_prompts=req.num_prompts,
        batch_size=req.batch_size,
        num_batches=num_batches,
    )
