import torch
import os
import torch.distributed as dist
import torch.multiprocessing as mp
from emotional_gene import emotional_gene

from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer

model_name = "gpt2-medium"
model = GPT2LMHeadModel.from_pretrained(model_name, output_hidden_states=True)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

def run_inference(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '5554'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    model.to(rank)
    model.eval()

    if dist.get_rank() == 0:
        prompt = "also I"
        affect = None
        topic = 'positive'
        knob = 0.2
    elif dist.get_rank() == 1:
        prompt = "there are"
        affect = "anger"
        topic = 'negative'
        knob = 0.4
    
    generated_text = emotional_gene(Knob=knob, Prompt=prompt, Topic=topic, Affect=affect, model = model, tokenizer=tokenizer)
    print(generated_text)

def main():
    world_size = 2
    mp.spawn(
        run_inference,
        args=(world_size,),
        nprocs=world_size,
        join=True
    )

if __name__ == "__main__":
    main()

    
