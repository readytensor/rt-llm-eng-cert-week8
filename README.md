# LLM Engineering & Deployment - Week 8 Code Examples

**Week 8: Inference Frameworks in Practice**  
Part of the LLM Engineering & Deployment Certification Program

This repository contains code examples for deploying and benchmarking LLM inference frameworks. The module covers:

- **Baseline Inference** - Hugging Face Transformers + FastAPI serving
- **vLLM** - High-throughput serving with PagedAttention and continuous batching
- **TGI** - Hugging Face's Text Generation Inference server
- **SGLang** - Structured generation and prefix caching
- **GPU Quantization** - GPTQ and AWQ for faster inference
- **GGUF & llama.cpp** - CPU/local inference with quantized models

---

## Prerequisites

- Python 3.10+
- CUDA-capable GPU (required for vLLM, TGI, SGLang)
- ~8GB+ GPU memory for Llama 3.2 1B experiments
- Docker (optional, for TGI deployment)

---

## Setup

### 1. Environment Setup

Create a virtual environment:

```bash
python -m venv venv
```

Activate the virtual environment:

```bash
# On Windows:
venv\Scripts\activate

# On Mac/Linux:
source venv/bin/activate
```

### 2. Dependency Installation

**For baseline experiments:**

```bash
pip install torch transformers peft accelerate fastapi uvicorn
```

**For vLLM:**

```bash
pip install vllm
```

**For llama.cpp (CPU inference):**

```bash
pip install llama-cpp-python
```

### 3. Hugging Face Authentication

Some models (like Llama) require authentication:

```bash
huggingface-cli login
```

Accept the model license on the Hugging Face model page before downloading.

---

## Running the Code Examples

### Baseline Server

The baseline FastAPI server demonstrates naive inference with Hugging Face Transformers:

```bash
cd code
uvicorn deploy_baseline:app --host 0.0.0.0 --port 8000
```

**Endpoints:**

| Endpoint | Description |
|----------|-------------|
| `POST /generate` | Non-streaming text generation |
| `POST /generate_stream` | Streaming text generation |
| `POST /ttft_itl_batched` | Benchmark TTFT/ITL under load |

**Test generation:**

```bash
curl -X POST http://localhost:8000/generate \
    -H "Content-Type: application/json" \
    -d '{"prompt": "Summarize: The quick brown fox...", "max_new_tokens": 100}'
```

**Run benchmarks:**

```bash
curl -X POST http://localhost:8000/ttft_itl_batched \
    -H "Content-Type: application/json" \
    -d '{"input_tokens": 512, "generated_tokens": 256, "num_prompts": 20, "batch_size": 4}'
```

### vLLM Server

Start a vLLM server with LoRA support:

```bash
vllm serve meta-llama/Llama-3.2-1B-Instruct \
    --enable-lora \
    --lora-modules tuned_model=moo3030/Llama-3.2-1B-QLoRA-Summarizer-adapters \
    --gpu-memory-utilization 0.7 \
    --max-model-len 2048
```

**Test with OpenAI-compatible API:**

```bash
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "meta-llama/Llama-3.2-1B-Instruct",
        "messages": [{"role": "user", "content": "Hello!"}]
    }'
```

**Benchmark with vLLM's built-in tool:**

```bash
vllm bench serve \
    --backend vllm \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --dataset-name random \
    --random-input-len 512 \
    --random-output-len 256 \
    --num-prompts 20 \
    --max-concurrency 4
```

### Notebooks

| Notebook | Description |
|----------|-------------|
| `inference_baseline.ipynb` | Run baseline server and benchmark TTFT/ITL |
| `inference_vllm.ipynb` | Deploy vLLM, test streaming, concurrent requests |

---

## Lessons Overview

### Unit 1: Inference Fundamentals (Theory)

| Lesson | Topic | Key Concepts |
|--------|-------|--------------|
| 1 | Inference Basics | Autoregressive generation, prefill vs decode, bottlenecks |
| 2 | Benchmarking | TTFT, ITL, E2E, throughput, warmup, percentile reporting |
| 3 | KV Cache | Cache structure, memory cost, fragmentation |
| 4 | Attention Optimizations | Flash Attention, Paged Attention |
| 5 | Quantization | INT8/INT4, GPTQ, AWQ, GGUF formats |
| 6 | Scheduling | Continuous batching, speculative decoding |

### Unit 2: Inference Frameworks (Practice)

| Lesson | Topic | Key Concepts |
|--------|-------|--------------|
| 0 | HF Baseline | FastAPI serving, streaming, baseline metrics |
| 1 | vLLM | PagedAttention, continuous batching, OpenAI API |
| 2 | TGI | Hugging Face inference server, Docker deployment |
| 3 | SGLang | RadixAttention, structured output, prefix caching |
| 4 | GPU Quantization | Loading GPTQ/AWQ models in vLLM |
| 5 | GGUF & llama.cpp | CPU inference, quantization levels |
| 6 | NIM Evaluation | End-to-end framework comparison |

---

## Project Structure

```
rt-llm-eng-cert-week8/
├── code/
│   ├── deploy_baseline.py       # Baseline FastAPI server
│   ├── inference_baseline.ipynb # Baseline benchmarking notebook
│   └── inference_vllm.ipynb     # vLLM deployment notebook
├── images/
│   ├── lesson2/                 # TGI vs vLLM diagrams
│   ├── lesson4/                 # GPTQ process diagrams
│   └── lesson5/                 # Quantization diagrams
├── unit1-lessons/               # Theory lessons (Module A)
│   ├── lesson1-inference-basics.md
│   ├── lesson2-benchmarking.md
│   ├── lesson3-kv-cache.md
│   ├── lesson4-attention-optimizations.md
│   ├── lesson5-quantization.md
│   └── lesson6-scheduling-optimizations.md
├── unit2-lessons/               # Practice lessons (Module B)
│   ├── overview.md
│   ├── lesson0-hf-baseline.md
│   ├── lesson1-vllm.md
│   ├── lesson2-tgi.md
│   ├── lesson3-sglang.md
│   ├── lesson4-gpu-quantization.md
│   ├── lesson5-gguf-llamacpp.md
│   └── lesson6-nim-evaluation.md
└── README.md
```

---

## Framework Comparison Reference

| Framework | Best For | Key Feature |
|-----------|----------|-------------|
| **HF Baseline** | Prototyping, simple deployments | Simplicity, flexibility |
| **vLLM** | High-throughput production serving | PagedAttention, continuous batching |
| **TGI** | HuggingFace ecosystem integration | Docker deployment, safety features |
| **SGLang** | Structured output, agentic workloads | RadixAttention, constrained decoding |
| **llama.cpp** | CPU/local inference | GGUF quantization, no GPU required |

---

## Key Metrics Reference

| Metric | What It Measures | What Drives It |
|--------|------------------|----------------|
| **TTFT** | Time to first token | Prefill computation + scheduling |
| **ITL** | Time between tokens | Decode efficiency, memory bandwidth |
| **TPOT** | Time per output token | Average decode latency |
| **Throughput** | Tokens/second | Batching, hardware utilization |
| **RPS** | Requests/second | Overall system capacity |

---

## Hardware Considerations

- **GPU Memory**: Llama 3.2 1B requires ~4GB in FP16. Larger models need more or quantization.
- **vLLM**: Requires CUDA. Use `--gpu-memory-utilization` to control memory allocation.
- **Quantization**: 4-bit models reduce memory ~4x with minimal quality loss.
- **CPU Inference**: llama.cpp with GGUF models works on any hardware but is slower.

---

## License

This work is licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).

**You are free to:**

- Share and adapt this material for non-commercial purposes
- Must give appropriate credit and indicate changes made
- Must distribute adaptations under the same license

See [LICENSE](LICENSE) for full terms.

---

## Contact

For questions or issues related to this repository, please refer to the course materials or contact your instructor.
