from time import perf_counter
import transformers
import torch

model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

start_loading = perf_counter()
pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)
end_loading = perf_counter()
print(f"Loading shards took {end_loading - start_loading:.6f} seconds")

messages = [
    {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
    {"role": "user", "content": "Who are you?"},
]

num_tokens = 256
start_generation = perf_counter()
outputs = pipeline(
    messages,
    max_new_tokens=num_tokens,
)
end_generation = perf_counter()

print(outputs[0]["generated_text"][-1])
print(f"Generating {num_tokens} tokens took {end_generation - start_generation:.6f} seconds.")