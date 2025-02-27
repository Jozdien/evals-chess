from time import perf_counter
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, prepare_model_for_kbit_training
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
OUTPUT_DIR = "/tmp/lora-imdb"

# Clear GPU memory
torch.cuda.empty_cache()

dataset = load_dataset("stanfordnlp/imdb", split="train")  # to test

# Quantization configuration
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

# LoRA configuration
lora_config = LoraConfig(
    r=LORA_R,  # Rank
    lora_alpha=LORA_ALPHA,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM"
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

# Prepare model for training
model = prepare_model_for_kbit_training(model)

# Training arguments
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUMULATION_STEPS,
    gradient_checkpointing=True,
    optim="paged_adamw_32bit",
    logging_steps=10,
    learning_rate=LEARNING_RATE,
    bf16=True,
    save_strategy="steps",
    save_steps=50,
    max_steps=MAX_STEPS,  # Adjust based on your needs
    warmup_ratio=0.05,
    lr_scheduler_type="constant",
)

# Initialize trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=training_args,
    tokenizer=tokenizer,
    peft_config=lora_config,
)

# Start training
trainer.train()

# Save the final model
trainer.save_model()

END_TIME = perf_counter()
print(f"""Training QLoRA with the following hyperparameters:
Learning rate: {LEARNING_RATE}
Batch size: {BATCH_SIZE}
Gradient Accumulation Steps: {GRAD_ACCUMULATION_STEPS}
Max steps: {MAX_STEPS}
Number of epochs: {NUM_EPOCHS}
LORA Rank: {LORA_R}
LORA Alpha: {LORA_ALPHA}
LORA Dropout: {LORA_DROPOUT}
took {END_TIME - START_TIME:.6f} seconds.""")