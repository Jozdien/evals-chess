# Haven't tested this code

from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load the trained LoRA adapter
peft_model_id = "/tmp/lora-imdb"
config = PeftConfig.from_pretrained(peft_model_id)
model = AutoModelForCausalLM.from_pretrained(
   config.base_model_name_or_path, 
   device_map="auto",
   torch_dtype=torch.float16
)
model = PeftModel.from_pretrained(model, peft_model_id)
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

# Test inference
text = "This movie was absolutely"
inputs = tokenizer(text, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=50, temperature=0.7)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))