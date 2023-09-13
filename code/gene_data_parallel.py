import torch
import os
import torch.distributed as dist
import torch.multiprocessing as mp
from emotional_gene import emotional_gene
# from data_utils import get_targets, extract_moseii_from_extraction_universal, extract_meld_from_extraction_universal
# from constants import * 
# from eval_utils import is_float
# from run_utils import get_input_promts

import numpy as np

from transformers import GPT2LMHeadModel, GPT2Tokenizer


def run_inference(rank, prompts, model, tokenizer, result_queue, world_size):

    # dist.init_process_group("nccl", rank=rank, world_size=world_size)
    if model is None or tokenizer is None:
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")
        model = GPT2LMHeadModel.from_pretrained("gpt2-medium", output_hidden_states=True)
    
    # if len(knobs) == 2:
    #     if rank == 0:
    #         prompt = prompts[0]
    #         affect = affects[0]
    #         topic = topics[0]
    #         knob = knobs[0]
    #     elif rank == 1:
    #         prompt = prompts[1]
    #         affect = affects[1]
    #         topic = topics[1]
    #         knob = knobs[1]
    # elif len(knobs) == 1:
    #     prompt = prompts[0]
    #     affect = affects[0]
    #     topic = topics[0]
    #     knob = knobs[0]
    device  = torch.device('cuda', rank)
    model.to(rank)
    model.eval()
    for i in range(len(prompts) - 1):
        if rank == 0:
            knob = prompts[i][0]
            prompt = prompts[i][1]
            topic = prompts[i][2]
            affect = prompts[i][3]  
        elif rank == 1:
            knob = prompts[i+1][0]
            prompt = prompts[i+1][1]
            topic = prompts[i+1][2]
            affect = prompts[i+1][3] 

        
    # print(device)

        generated_text = emotional_gene(Knob=knob, Prompt=prompt, Topic=topic, Affect=affect, model = model, tokenizer=tokenizer, device = device)
        print(generated_text)
        result_queue.put((generated_text))



def run_emo_gene_parallel(
        prompts = [[0.4, 'there are', 'negative', 'anger'], 
               [0, 'also I', None, 'joy'], 
               [0.6, 'objectives', 'positive', 'surprise'], 
               [0.7, 'negative speech', 'negative', 'sadness'],
               [0.7, 'positive speech', 'positive', 'trust'],
               [0.9, 'the hospital', 'negative', 'sadness'],
               [0.3, 'I dont know', 'negative', 'anger'],
               [1, 'negative speech', 'negative', 'sadness']], 
        model = None, 
        tokenizer = None):
    
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '5554'
    world_size = torch.cuda.device_count()

    if model is None or tokenizer is None:
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")
        model = GPT2LMHeadModel.from_pretrained("gpt2-medium", output_hidden_states=True)
    
    
    result_queue = mp.Queue()
    for rank in range(world_size):
        mp.Process(target=run_inference, args=(rank, prompts, model, tokenizer, result_queue, world_size)).start()

    generated_texts = []
    for _ in range(world_size):
        temp_res = result_queue.get()
        generated_texts.append(temp_res)
        del temp_res

    # print(generated_texts[0])
    # print(generated_texts[1])
    return generated_texts

generated_texts = run_emo_gene_parallel()
print(generated_texts)
    
