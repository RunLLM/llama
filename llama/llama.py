from typing import Tuple, List
import os
import sys
import torch
import time
import json

from pathlib import Path

from fairscale.nn.model_parallel.initialize import initialize_model_parallel

from llama_facebook import ModelArgs, Transformer, Tokenizer, LLaMA

def _setup_model_parallel() -> Tuple[int, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))

    torch.distributed.init_process_group("nccl")
    initialize_model_parallel(world_size)
    torch.cuda.set_device(local_rank)

    # seed must be the same in all processes
    torch.manual_seed(1)
    return local_rank, world_size


def _load(
    ckpt_dir: str,
    tokenizer_path: str,
    local_rank: int,
    world_size: int,
    max_seq_len: int,
    max_batch_size: int,
) -> LLaMA:
    start_time = time.time()
    checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
    assert world_size == len(
        checkpoints
    ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {world_size}"
    ckpt_path = checkpoints[local_rank]
    print("Loading")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    with open(Path(ckpt_dir) / "params.json", "r") as f:
        params = json.loads(f.read())

    model_args: ModelArgs = ModelArgs(
        max_seq_len=max_seq_len, max_batch_size=max_batch_size, **params
    )
    tokenizer = Tokenizer(model_path=tokenizer_path)
    model_args.vocab_size = tokenizer.n_words
    torch.set_default_tensor_type(torch.cuda.HalfTensor)
    model = Transformer(model_args)
    torch.set_default_tensor_type(torch.FloatTensor)
    model.load_state_dict(checkpoint, strict=False)

    generator = LLaMA(model, tokenizer)
    print(f"Loaded in {time.time() - start_time:.2f} seconds")
    return generator

def generate(prompts: List[str]):
    ckpt_dir = '/7B/'
    tokenizer_path = '/tokenizer.model'
    temperature = 0.8
    top_p = 0.95
    max_seq_len = 512
    max_batch_size = 8

    local_rank, world_size = _setup_model_parallel()
    if local_rank > 0:
        sys.stdout = open(os.devnull, "w")

    generator = _load(
        ckpt_dir, tokenizer_path, local_rank, world_size, max_seq_len, max_batch_size
    )

    results = []
    for i in range(0, len(prompts) // max_batch_size):
        print("Inference iteration", i+1)
        # Generate results for each full batch of prompts
        prompt_batch = prompts[i * max_batch_size:(i + 1) * max_batch_size]
        prompt_results = generator.generate(
            prompt_batch, max_gen_len=64, temperature=temperature, top_p=top_p
        )
        results += prompt_results

    # Generate results for the remaining prompts (if any)
    if len(prompts) % max_batch_size != 0:
        print("Inference iteration", len(prompts) // max_batch_size + 1)
        prompt_batch = prompts[(len(prompts) // max_batch_size) * max_batch_size:]
        prompt_results = generator.generate(
            prompt_batch, max_gen_len=64, temperature=temperature, top_p=top_p
        )
        results += prompt_results
    
    return results
