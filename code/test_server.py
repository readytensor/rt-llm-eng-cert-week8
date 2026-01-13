import requests

dialogue = """Victoria: God I'm really broke, I spent way to much this month 
Victoria: At least we get paid soon..
Magda: Yeah, don't remind me, I know the feeling
Magda: I just paid my car insurance, I feel robbed 
Victoria: Thankfully mine is paid for the rest of the year"""

response = requests.post(
    "http://localhost:8080/v1/chat/completions",
    json={
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant that summarizes conversations.",
            },
            {"role": "user", "content": f"Summarize this conversation:\n{dialogue}"},
        ],
        "max_tokens": 150,
        "temperature": 0.01,
    },
)

print(response.json()["choices"][0]["message"]["content"])
