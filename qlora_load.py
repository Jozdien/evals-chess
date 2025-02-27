from time import perf_counter
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

# Load the trained LoRA adapter
peft_model_id = "/tmp/qlora-imdb" 
config = PeftConfig.from_pretrained(peft_model_id)

bnb_config = BitsAndBytesConfig(
   load_in_4bit=True,
   bnb_4bit_quant_type="nf4",
   bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
   config.base_model_name_or_path,
   quantization_config=bnb_config,
   device_map="auto"
)
model = PeftModel.from_pretrained(model, peft_model_id)
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

# Test inference
text = "This movie was absolutely"
inputs = tokenizer(text, return_tensors="pt").to("cuda")
num_tokens = 50

start_generation = perf_counter()
outputs = model.generate(**inputs, max_new_tokens=num_tokens, temperature=0.7)
end_generation = perf_counter()

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
print(f"Gernerating {num_tokens} tokens took {end_generation - start_generation:.6f} seconds")