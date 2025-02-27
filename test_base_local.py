from time import perf_counter
import transformers
import torch

# model_id = "meta-llama/Llama-3.1-70B"
model_id = "Qwen/Qwen2.5-72B"
puzzle_prompt = "r2qkb1r/pp2nppp/3p4/2pNN1B1/2BnP3/3P4/PPP2PPP/R2bK2R w KQkq - 1 0\n1. Nf6+ gxf6 2."
game_prompt = """[Event "FIDE Candidates Tournament 2022"]
[Site "Madrid, Spain"]
[Date "2022.06.17"]
[Round "1.3"]
[White "Caruana, Fabiano"]
[Black "Nakamura, Hikaru"]
[Result "1-0"]
[WhiteElo "2783"]
[WhiteTitle "GM"]
[BlackElo "2760"]
[BlackTitle "GM"]
[TimeControl "900+10"]
[Variant "Standard"]

1."""

num_tokens = 15

start_loading = perf_counter()
pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={
        "torch_dtype": torch.bfloat16,
        # "quantization_config": {
        #     "load_in_8bit": True
        # },
    },
    device_map="auto"
)
end_loading = perf_counter()
print(f"Loading shards took {end_loading - start_loading:.6f} seconds")

start_generation = perf_counter()
print(pipeline(puzzle_prompt, max_new_tokens=num_tokens))
end_generation = perf_counter()

print(f"Gernerating {num_tokens} tokens took {end_generation - start_generation:.6f} seconds.")