from time import perf_counter
from datasets import load_dataset
from transformers import (
   AutoModelForCausalLM,
   AutoTokenizer,
   TrainingArguments
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
import torch

START_TIME = perf_counter()

# Core hyperparameters
LEARNING_RATE = 2e-4
BATCH_SIZE = 4
GRAD_ACCUMULATION_STEPS = 4
MAX_STEPS = 100
NUM_EPOCHS = 3

# LoRA specific
LORA_R = 8
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

# Model config
MODEL_ID = "meta-llama/Llama-3.1-8B"
OUTPUT_DIR = "/tmp/lora-imdb"  # to test

# Clear GPU memory
torch.cuda.empty_cache()

# Load dataset
dataset = load_dataset("stanfordnlp/imdb", split="train")

# Load base model
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",
    torch_dtype=torch.float16,
    use_cache=False, # use_cache incompatible with gradient checkpointing
    max_memory={0: "40GB"}
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

# LoRA configuration
lora_config = LoraConfig(
   r=LORA_R,
   lora_alpha=LORA_ALPHA,
   target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
   lora_dropout=LORA_DROPOUT,
   bias="none",
   task_type="CAUSAL_LM"
)

# Apply LoRA
model = get_peft_model(model, lora_config)

# Enable gradient computation
for param in model.parameters():
    param.requires_grad = True

# You might also want to print trainable parameters to verify
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Number of trainable parameters: {trainable_params}")

# Training arguments
training_args = TrainingArguments(
   output_dir=OUTPUT_DIR,
   num_train_epochs=NUM_EPOCHS,
   per_device_train_batch_size=BATCH_SIZE,
   gradient_accumulation_steps=GRAD_ACCUMULATION_STEPS,
   gradient_checkpointing=True,
   optim="adamw_torch_fused",
   logging_steps=10,
   learning_rate=LEARNING_RATE,
   bf16=True,
   save_strategy="steps",
   save_steps=50,
   max_steps=MAX_STEPS,
   warmup_ratio=0.05,
   lr_scheduler_type="constant",
)

# Initialize trainer
trainer = SFTTrainer(
   model=model,
   train_dataset=dataset,
   args=training_args,
   tokenizer=tokenizer,
)

# Start training
trainer.train()
trainer.save_model()

END_TIME = perf_counter()
print(f"""Training LoRA with the following hyperparameters:
Learning rate: {LEARNING_RATE}
Batch size: {BATCH_SIZE}
Gradient Accumulation Steps: {GRAD_ACCUMULATION_STEPS}
Max steps: {MAX_STEPS}
Number of epochs: {NUM_EPOCHS}
LORA Rank: {LORA_R}
LORA Alpha: {LORA_ALPHA}
LORA Dropout: {LORA_DROPOUT}
took {END_TIME - START_TIME:.6f} seconds.""")