import torch
import os
import torch.distributed as dist
import torch.multiprocessing as mp
from emotional_gene import emotional_gene

from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer

model_name = "gpt2-medium"
model = GPT2LMHeadModel.from_pretrained(model_name, output_hidden_states=True)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

def run_inference(rank, knobs, prompts, topics, affects, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '5554'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    model.to(rank)
    model.eval()

    if dist.get_rank() == 0:
        prompt = prompts[0]
        affect = affects[0]
        topic = topics[0]
        knob = knobs[0]
    elif dist.get_rank() == 1:
        prompt = prompts[1]
        affect = affects[1]
        topic = topics[1]
        knob = knobs[1]
    
    generated_text = emotional_gene(Knob=knob, Prompt=prompt, Topic=topic, Affect=affect, model = model, tokenizer=tokenizer, device = rank)
    print(generated_text)

def run_emo_gene(knobs=[0.4, 0.5], prompts=['There are', 'also I'], topics=[None, 'positive'], affects=[None, 'joy']):
    world_size = 2
    mp.spawn(
        run_inference,
        args=(world_size,knobs, prompts, topics, affects, ),
        nprocs=world_size,
        join=True
    )

run_emo_gene()

# if __name__ == "__main__":
#     main()

    
