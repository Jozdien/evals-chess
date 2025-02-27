# How to Set Up

Disclaimer: I set this up on a vast.ai server with the Pytorch (Vast) template. This means that a lot of packages aren't yet included in `requirements.txt`.

Run `pip install -r requirements.txt`. Then run the following code to install stockfish if you want to run code that involves that:

```bash
# Download the tar file
wget https://github.com/official-stockfish/Stockfish/releases/latest/download/stockfish-ubuntu-x86-64-avx2.tar

# Extract it (this will create a stockfish file)
tar xf stockfish-ubuntu-x86-64-avx2.tar && rm stockfish-ubuntu-x86-64-avx2.tar

# Make it executable
cd .. && mv evals_chess/stockfish/stockfish-ubuntu-x86-64-avx2 evals_chess/stockfish-temp && rm -r evals_chess/stockfish && cd evals_chess && mv stockfish-temp stockfish
```

# What Code Do

## Chess Evaluation

- `run_base_local.py`: Run chess games with local open-source base models (Qwen2.5-72B, Llama-3.1-70B, etc.)
- `run_base.py`: Run chess games with base models (gpt-4-base)
- `run_chat.py`: Run chess games with chat models (gpt-4o, fine-tuned models)

## LoRA Training and Loading

- `lora_train.py` & `qlora_train.py`: Scripts for training LoRAs and QLoRAs
- `lora_load.py` & `qlora_load.py`: Scripts for loading trained LoRA and QLoRA adapters

## Quick Testing

- `test_base_local.py`: Test script for local open-source base models, just to check that the model is working
- `test_instruct_local.py`: Test script for local open-source instruction models, just to check that the model is working

## Utilities

- `utils.py`: Many, many QoL functions
- `filter_games.py`: Filter games from the Lichess database to create datasets for training
- `generate_datasets.py`: Convert filtered PGN files into fine-tuning format
- `validate_datasets.ipynb`: Notebook for validating dataset formatting
- `file_handler.py`: Utility for uploading files to OpenAI API
- `finetune_handler.py`: Utility for creating and managing fine-tuning jobs with OpenAI