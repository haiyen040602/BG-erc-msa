import logging
import re
import os
import random
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
from transformers import (AdamW, AutoModelForSeq2SeqLM, GPT2LMHeadModel, GPT2Tokenizer,
                          AutoTokenizer, get_linear_schedule_with_warmup)

from constants import *
from data_utils import (ABSADataset, filter_none, filter_invalid,
                        get_dataset, get_inputs, normalize_augment, get_inputs_and_targets,
                        extract_meld_from_extraction_universal, extract_iemocap_from_extraction_universal,
                        extract_moseii_from_extraction_universal,
                        get_targets)
from model_utils import (prepare_constrained_tokens, prepare_tag_tokens)
from eval_utils import is_float
from emotional_gene import emotional_gene
from gene_data_parallel import run_emo_gene_parallel

logger = logging.getLogger(__name__)


class Prefix_fn_cls():
    def __init__(self, tokenizer, name, special_tokens, input_enc_idxs):
        self.tokenizer=tokenizer
        self.input_enc_idxs=input_enc_idxs
        self.special_ids = [element for l in self.tokenizer(special_tokens, add_special_tokens=False)['input_ids'] for element in l]
        self.special_ids = list(set(self.special_ids))
        self.task = name
        # self.extract = args.extract

    def get(self, batch_id, previous_tokens):
        # get input
        inputs=[]
        # print("Recent task: ", self.task)
        if "gene" in self.task:
            inputs = list(set(self.input_enc_idxs[batch_id].tolist())) + self.special_ids
        else:
            inputs = self.special_ids
        return inputs


def train(args, tokenizer, model, train_dataset, task, epochs, lr, bs, acc_step=None, save_ckpt=False, save_last=False):
    start_info = "#"*20+f" Conduct {task} Training"+"#"*20
    logger.info("#"*len(start_info))
    logger.info(start_info)
    logger.info("#"*len(start_info))

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        { "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": args.weight_decay, },
        { "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0, },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=lr, eps=args.adam_epsilon)

    if acc_step is None:
        acc_step = args.gradient_accumulation_steps

    train_dataloader = DataLoader(train_dataset, batch_size=bs, drop_last=True, shuffle=True, num_workers=4)
    t_total = (
        (len(train_dataloader.dataset) // (bs * max(1, len(args.n_gpu))))
        // acc_step
        * float(epochs)
    )
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)
    train_iterator = trange(int(epochs), dynamic_ncols=True, desc="Epoch")
    criterion = nn.CrossEntropyLoss()
    # visualize input
    logger.info(f"Training examples out of {len(train_dataset)}:")
    for i in range(3):
        logger.info('Input : {}'.format(tokenizer.decode(train_dataset[i]['source_ids'], skip_special_tokens=True)))
        logger.info('Output: {}'.format(tokenizer.decode(train_dataset[i]['target_ids'], skip_special_tokens=True)))

    logger.info(f"Model emb weights of <pad> {model.shared.weight[0][:5]}")
    # start training
    for n_epoch, _ in enumerate(train_iterator):
        epoch_train_loss = 0.0
        epoch_iterator = tqdm(train_dataloader, dynamic_ncols=True, desc="Iteration", disable=True)
        for step, batch in enumerate(epoch_iterator):
            model.train()

            # truth label removed pad tokens
            lm_labels = batch["target_ids"]
            lm_labels[lm_labels[:, :] == tokenizer.pad_token_id] = -100

            outputs = model(
                batch["source_ids"].to(args.device),
                attention_mask=batch["source_mask"].to(args.device),
                labels=lm_labels.to(args.device),
                decoder_attention_mask=batch['target_mask'].to(args.device),
                decoder_input_ids=None,
            )

            loss = outputs[0]
            loss.backward()
            epoch_train_loss += loss.item()

            # loss = criterion(outputs, lm_labels)
            # loss.backward()
            # epoch_train_loss += loss.item()

            ## Không sử dụng điều kiện dùng loss
            if (step+1) % acc_step == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                model.zero_grad()

        if save_ckpt and n_epoch in range(args.num_train_epochs)[-args.save_last_k:]:
            ckpt_dir = os.path.join(args.seed_dir, f"checkpoint-e{n_epoch}")
            if not os.path.exists(ckpt_dir):
                os.makedirs(ckpt_dir)
            model.save_pretrained(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir)
            # json.dump(args, open(f"{ckpt_dir}/args.json", 'w'), indent=4)
            logger.info(f"Save model checkpoint to {ckpt_dir}")

        logger.info(f"Epoch {n_epoch} Avg epoch train loss: {epoch_train_loss / len(epoch_iterator):.5f} lr: {scheduler.get_last_lr()}")

    if save_last:
        ckpt_dir = os.path.join(args.seed_dir, f"{task}-model")
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
        model.save_pretrained(ckpt_dir)
        tokenizer.save_pretrained(ckpt_dir)
        logger.info(f"Save model checkpoint to {ckpt_dir}")

    logger.info("Finish training!")


def aux_training(args, tokenizer, model, train_dataset):

    return_values = {}
    if args.data_gene:
        return_values["data_gene_dataset"] = data_gene(args, tokenizer, model, train_dataset)
    if args.pseudo:
        return_values["pseudo_dataset"] = pseudo_label(args, tokenizer, model, train_dataset)

    return return_values


def infer(args, dataset, model, tokenizer, name, is_constrained=False, constrained_vocab=None, keep_mask=False, **decode_dict):
    dataloader = DataLoader(dataset, batch_size=args.eval_batch_size, num_workers=4)

    if keep_mask:
        print("Keep Mask: ", keep_mask)
        # can't skip special directly, will lose extra_id
        unwanted_tokens = [tokenizer.eos_token, tokenizer.unk_token, tokenizer.pad_token]
        unwanted_ids = tokenizer.convert_tokens_to_ids(unwanted_tokens)
        def filter_decode(ids):
            ids = [i for i in ids if i not in unwanted_ids]
            tokens = tokenizer.convert_ids_to_tokens(ids)
            sentence = tokenizer.convert_tokens_to_string(tokens)
            return sentence

    # inference
    print("Recent task: ", name)
    inputs, outputs, targets = [], [], []
    logger.info(f"Inferencing on {name} ...")
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader):
            if is_constrained:
                prefix_fn_obj = Prefix_fn_cls(tokenizer, name, constrained_vocab, batch['source_ids'].to(args.device))
                prefix_fn = lambda batch_id, sent: prefix_fn_obj.get(batch_id, sent)
            else:
                prefix_fn = None

            outs_dict = model.generate(input_ids=batch['source_ids'].to(args.device),
                                        attention_mask=batch['source_mask'].to(args.device),
                                        # max_length=10,
                                        prefix_allowed_tokens_fn=prefix_fn,
                                        output_scores=True,
                                        return_dict_in_generate=True,
                                        **decode_dict,
                                        )
            outs = outs_dict["sequences"]

            if keep_mask:
                input_ = [filter_decode(ids) for ids in batch["source_ids"]]
                dec = [filter_decode(ids) for ids in outs]
                target = [filter_decode(ids) for ids in batch["target_ids"]]
            else:
                input_ = [tokenizer.decode(ids, skip_special_tokens=True) for ids in batch["source_ids"]]
                dec = [tokenizer.decode(ids, skip_special_tokens=True) for ids in outs]
                target = [tokenizer.decode(ids, skip_special_tokens=True) for ids in batch["target_ids"]]

            inputs.extend(input_)
            outputs.extend(dec)
            targets.extend(target)

    decode_txt = "constrained" if is_constrained else "greedy"
    with open(os.path.join(args.inference_dir, f"{name}_{decode_txt}_output.txt"), "w", encoding='utf-8') as f:
        for i, o in enumerate(outputs):
            f.write(f"{inputs[i]} ===> {o}\n")

    return inputs, outputs, targets


def data_gene(args, tokenizer, model, train_dataset):

    # 1. extract label
    if args.data_gene_extract:
        extract_task = f"extract_{args.task}"
        target_extract_inputs, target_extract_outputs = extract_model(args, tokenizer, model, extract_task)
    else:
        target_extract_inputs = []
        target_extract_outputs = []
        test_dataset = get_dataset(args, task=args.task, data_type="test", tokenizer=tokenizer)
        for data in test_dataset:
            input = tokenizer.decode(data['source_ids'], skip_special_tokens=True)
            label = tokenizer.decode(data['target_ids'], skip_special_tokens=True)
            target_extract_inputs.append(input)
            target_extract_outputs.append(label)
        
        print('Input data for generation process: ', target_extract_inputs[0])
        print('Output data for generation process: ', target_extract_outputs[0] )

    # 2. gene train & infer
    target_gene_aug_inputs, target_gene_aug_outputs = gene_model(
        args, tokenizer, model, target_extract_inputs, target_extract_outputs
    )

    for i in range(3):
        print("Input of sentence generation: ", target_gene_aug_inputs[i])
        print("Output of sentence generation: ", target_gene_aug_outputs[i])

    # 3. postprocess
    # change direction & post process
    target_gene_aug_inputs_processed, target_gene_aug_targets_processed = postprocess_gene_outputs(args, target_gene_aug_inputs, target_gene_aug_outputs)

    # 3.1 use the extract model to do filtering
    if args.model_filter:
        target_gene_aug_inputs_processed, target_gene_aug_targets_processed = model_filter(args, target_gene_aug_inputs_processed, target_gene_aug_targets_processed)
        logger.info(f"Aug num after filtering: {len(target_gene_aug_inputs_processed)}")

    # 3.2 control number of augmentation number
    if args.data_gene_aug_num:
        target_gene_aug_inputs_processed, target_gene_aug_targets_processed = \
            target_gene_aug_inputs_processed[:args.data_gene_aug_num], target_gene_aug_targets_processed[:args.data_gene_aug_num]
    if args.data_gene_aug_ratio:
        aug_num = int(len(target_gene_aug_inputs_processed) * args.data_gene_aug_ratio)
        target_gene_aug_inputs_processed, target_gene_aug_targets_processed = \
            target_gene_aug_inputs_processed[:aug_num], target_gene_aug_targets_processed[:aug_num]

    logger.info(f"Aug num final: {len(target_gene_aug_inputs_processed)}")

    # 4. merge dataset
    target_gene_aug_dataset = ABSADataset(args, tokenizer, inputs=target_gene_aug_inputs_processed, targets=target_gene_aug_targets_processed, name="target_gene_aug")
    train_dataset_merged = ABSADataset(args, tokenizer, dataset_list=[train_dataset, target_gene_aug_dataset])

    return train_dataset_merged


def pseudo_label(args, tokenizer, model, train_dataset):
    # 1. train absa on train
    train(args, tokenizer, model, train_dataset, task=f"pseudo_{args.task}", epochs=args.num_train_epochs, lr=args.learning_rate, bs=args.train_batch_size, save_ckpt=False, save_last=True)

    # 2. inference on target unlabel
    target_dataset = get_dataset(args, task=args.task, data_type="target-unlabel", tokenizer=tokenizer)
    is_constrained = True if args.data_extract_wt_constrained else False
    print ('Pseudo Labels Extraction with constrained decoding: ', is_constrained)
    target_pseudo_inputs, target_pseudo_outputs, _ = infer(
        args, target_dataset, model, tokenizer, name=f"target_pseudo_{args.task}",
        is_constrained=is_constrained, constrained_vocab=prepare_constrained_tokens(tokenizer, args.task, args.paradigm),
    )

    for i in range(3):
        print("Input of pseudo inference: ", i, target_pseudo_inputs[i])
        print("Output of pseudo inference: ", i, target_pseudo_outputs[i])

    if args.pseudo_skip_none:
        target_pseudo_inputs, target_pseudo_outputs = pseudo_filter_none(target_pseudo_inputs, target_pseudo_outputs)

    # 3. merge pseudo labelled data
    target_pseudo_aug_dataset = ABSADataset(args, tokenizer, inputs=target_pseudo_inputs, targets=target_pseudo_outputs, name=f"target_pseudo_{args.task}")
    train_dataset_merged = ABSADataset(args, tokenizer, dataset_list=[train_dataset, target_pseudo_aug_dataset])
    return train_dataset_merged


def pseudo_filter_none(inputs, outputs):
    new_inputs, new_outputs = [], []
    for idx in range(len(outputs)):
        if "none" not in outputs[idx]:
            new_inputs.append(inputs[idx])
            new_outputs.append(outputs[idx])
    return new_inputs, new_outputs


def extract_model(args, tokenizer, model, extract_task):

    # 1. train extract model (trong trường hợp không có model sẵn hay sử dụng lại model cũ)
    if args.extract_model:
        model = AutoModelForSeq2SeqLM.from_pretrained(args.extract_model).to(args.device)
        tokenizer = AutoTokenizer.from_pretrained(args.extract_model, use_fast=False)
        logger.info(f"Model reloaded with {args.extract_model}")
        logger.info(f"Tokenizer len: {len(tokenizer)}")
    elif args.runned_folder:
        model_path = os.path.join(args.runned_folder, f"seed-{args.seed}",
        f"{args.source_domain}-{args.target_domain[0]}", f"extract_{args.task}-model")
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(args.device)
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        logger.info(f"Model reloaded with {model_path}")

    train_extract_dataset = get_dataset(args, task=extract_task, data_type="train", tokenizer=tokenizer)
    train(args, tokenizer, model, train_extract_dataset, task=extract_task, epochs=args.data_gene_extract_epochs, lr=args.learning_rate, bs=args.train_batch_size, save_ckpt=False, save_last=True)

    # 2. infer on target domain
    target_extract_dataset = get_dataset(args, task=extract_task, data_type="target-unlabel", tokenizer=tokenizer)

    is_constrained = True if args.data_extract_wt_constrained else False
    print("Data extraction with constrained decoding: ", is_constrained)

    target_extract_inputs, target_extract_outputs, _ = infer(
        args, target_extract_dataset, model, tokenizer,
        name=f"target_{extract_task}", is_constrained=is_constrained, constrained_vocab=prepare_constrained_tokens(tokenizer, args.task, args.paradigm)
    )
    for i in range(3):
        print("Input of Data Extraction: ", target_extract_inputs[i])
        print("Output of Data Extraction: ", target_extract_outputs[i])
    return target_extract_inputs, target_extract_outputs


def gene_model(args, tokenizer, model, target_extract_inputs, target_extract_outputs):

    if args.data_gene_same_model:
        logger.info(f"Model keep the same.")
    elif args.data_gene_base_model:
        # model_name = args.gene_model
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")
        model = GPT2LMHeadModel.from_pretrained("gpt2-medium")
        logger.info(f"Model reloaded with gpt2-medium")
    elif args.data_gene_affective_model:
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")
        model = GPT2LMHeadModel.from_pretrained("gpt2-medium", output_hidden_states=True)
        logger.info(f"Model reloaded with gpt2-medium")

    logger.info(f"Tokenizer len: {len(tokenizer)}")

    if args.data_gene_same_model:
        ## training with former model in extraction task
        train_gene_dataset = get_dataset(args, task=f"gene_{args.task}", data_type="train", tokenizer=tokenizer)
        train(args, tokenizer, model, train_gene_dataset, task=f"gene_{args.task}", epochs=args.data_gene_epochs, lr=args.learning_rate, bs=args.train_batch_size, save_ckpt=False, save_last=True)

        ## change order of input and output
        target_gene_inputs, target_gene_targets = target_extract_outputs, target_extract_inputs
        target_gene_dataset = ABSADataset(args, tokenizer, inputs=target_gene_inputs, targets=target_gene_targets, name="target_gene")

        ## prepare vocab for constrained decoding
        target_domain_words = prepare_gene_vocab(args)

        ## decoding methods
        decode_dict = {"min_length": args.data_gene_min_length,}
        specific_dict = {
            "greedy": {"do_sample": False},
            "top_p": {"do_sample": True, "top_p": args.data_gene_top_p},
            "beam": {"num_beams": args.data_gene_num_beam, "early_stopping": True},
        }
        if args.data_gene_decode:
            decode_dict.update(specific_dict[args.data_gene_decode])

        is_constrained = True if args.data_gene_wt_constrained else False
        print("Data Generation with constrained decoding: ", is_constrained)
        ## inference
        target_gene_aug_inputs, target_gene_aug_outputs, _ = infer(
            args, target_gene_dataset, model, tokenizer, name="target_gene",
            is_constrained=is_constrained, constrained_vocab=target_domain_words, **decode_dict
        )
    elif args.data_gene_affective_model:
        ## Prepare for input prompts  
        target_gene_inputs, target_gene_targets = target_extract_outputs, target_extract_inputs
        

        num_input_prompt = int( args.num_input_prompt)
        prompts, target_gene_aug_inputs = get_input_promts(args, target_gene_inputs, target_gene_targets, num_input_prompt)

        target_gene_aug_outputs = []
        
        logger.info(f"Starting generating data.")
        logger.info(f"Number of generated prompts: ")
        logger.info(str(len(prompts)))
        
        device = 'cuda'

        if torch.cuda.device_count() == 1:
            for i, prompt in enumerate(prompts):
                # generated_texts = run_emo_gene_parallel(knobs=[])
                model.to(device)
                model.eval()
                generated_text = emotional_gene(Knob=prompt[0], Prompt=prompt[1], Topic=prompt[2], Affect=prompt[3], model=model, tokenizer=tokenizer, device=device)
                target_gene_aug_outputs.append(generated_text)
                
                if i < 5:
                    print("prompt: ", prompt)
                    print("generated text: ", generated_text)
                if (i + 1) == args.data_gene_num_samples:
                    break

        elif torch.cuda.device_count() == 2:
            # for i in range(0, len(prompts) - 1, 2):
            #     knobs = [prompts[i][0], prompts[i+1][0]]
            #     prompt = [prompts[i][1], prompts[i+1][1]]
            #     topics = [prompts[i][2], prompts[i+1][2]]
            #     affects = [prompts[i][3], prompts[i+1][3]]
            #     generated_texts = run_emo_gene_parallel(knobs=knobs, prompts=prompt, topics=topics, affects=affects, model=model, tokenizer=tokenizer)
            prompts = prompts[: args.data_gene_num_samples]
            target_gene_aug_outputs = run_emo_gene_parallel(prompts=prompts, model=model, tokenizer=tokenizer)
            
            for i in range(args.data_gene_num_samples):
                print("prompt: ", prompts[i])
                print("generated text 1: ", target_gene_aug_outputs[0])
                print("generated text 2: ", target_gene_aug_outputs[1])
            


        logger.info("Ending generating data.")
        # with open(os.path.join(args.inference_dir, f"{name}_{decode_txt}_output.txt"), "w") as f:
        #     for i, o in enumerate(target_gene_aug_outputs):
        #         f.write(f"{inputs[i]} ===> {o}\n")
        for i in range(3):
            print("Prompt ", i, prompts[i])
            print("Generated Data Input: ", target_gene_aug_inputs[i])
            print("Generated Data Output: ", target_gene_aug_outputs[i])

        with open(os.path.join(args.inference_dir, f"emotional_gene_output.txt"), "w", encoding='utf-8') as f:
            for i, o in enumerate(target_gene_aug_outputs):
                f.write(f"{target_gene_aug_inputs[i]} ===> {o}\n")
        # target_gene_aug_inputs = target_gene_inputs

    return target_gene_aug_inputs, target_gene_aug_outputs


    

def get_input_promts(args, target_gene_inputs, target_gene_targets, num_input_promts):
    sents = []
    scores = []
    if "meld" in args.task:
        emotions = []
        sent_scores = get_targets(args=args, data_type_file="target-unlabel")
        # print(scores)
        for i in target_gene_inputs:
            emo = extract_meld_from_extraction_universal(i)
            emotions.append(emo[0])

    elif "moseii" in args.task:
        emotions = get_targets(args=args, data_type_file="target-unlabel")
        sent_scores = []
        for i in target_gene_inputs:
            extr = extract_moseii_from_extraction_universal(i)
            sent_scores.append(extr)
    
    error_score_index = []
    for i, sent_score in enumerate(sent_scores):
        if is_float(sent_score[1][0]):
            score = float(sent_score[1][0])
            scores.append(score)
        else:
            scores.append(0)
            error_score_index.append(i)
        
        if sent_score[0][0] in MOSEII_DICT:
                sents.append(sent_score[0][0])
        else:
            sents.append(None)


    ##TODO: normalize score from [-3, 3] to [0, 1]
    scores = np.array(scores)
    scores = list(map(abs, scores))
    nor_scores = (scores-min(scores))/(max(scores)-min(scores))

    prompts = []
    target_gene_aug_inputs = []
    
    neu_count = 0
    for i, input in enumerate(target_gene_targets):
        seqs = input.split()
        len_seqs = len(seqs)
        if num_input_promts <= len_seqs:
            prompt = " ".join(seqs[0:num_input_promts])
        else:
            prompt = " ".join(seqs[0:len_seqs])
        
        if i in error_score_index:
            continue
        
        if scores[i] >= 0 and sents[i] == 'pos' and emotions[i] in MELD_POS:
            sent = 'positive'
        elif scores[i] <= 0 and sents[i] == 'neg' and emotions[i] in MELD_NEG:
            sent = 'negative'
        elif sents[i] == 'pos' and emotions[i] == 'neutral':
            sent = 'positive'
        elif sents[i] == 'neg' and emotions[i] == 'neutral':
            sent = 'negative'
        elif sents[i] == 'neu' and emotions[i] in MELD_DICT_WO_NEUTRAL:
            sent = None
        else:
            continue
            

        if emotions[i] == 'neutral':
            prompts.append([nor_scores[i], prompt, sent, None])
        else:
            prompts.append([nor_scores[i], prompt, sent, emotions[i]])
        target_gene_aug_inputs.append(target_gene_inputs[i])
        
    with open(os.path.join(args.data_dir, f"gene_input_and_prompt.txt"), "w") as f:
        for i, p in enumerate(prompts):
            f.write(f"{target_gene_aug_inputs[i]} ===> {p}\n")
    # target_gene_aug_inputs = target_gene_inputs
    return prompts, target_gene_aug_inputs


def postprocess_gene_outputs(args, target_gene_aug_inputs, target_gene_aug_outputs):
    # process input & output
    target_gene_aug_inputs_processed = target_gene_aug_outputs
    target_gene_aug_targets_processed = [i.strip() for i in target_gene_aug_inputs]

    # normalize and filter
    target_gene_aug_inputs_processed, target_gene_aug_targets_processed = normalize_augment(args, target_gene_aug_inputs_processed, target_gene_aug_targets_processed)
    target_gene_aug_inputs_processed, target_gene_aug_targets_processed = filter_none(target_gene_aug_inputs_processed, target_gene_aug_targets_processed, args.data_gene_none_remove_ratio)
    target_gene_aug_inputs_processed, target_gene_aug_targets_processed = filter_invalid(target_gene_aug_inputs_processed, target_gene_aug_targets_processed)

    return target_gene_aug_inputs_processed, target_gene_aug_targets_processed


def extract_label_words(inputs):
    label_words = []
    for i in inputs:
        i = re.sub("<\w+>", "", i)
        label_words += i.strip().split()
    return label_words
   

def prepare_gene_vocab(args):
    # Get target inputs
    # target_inputs = get_inputs(args, data_type_file="target-unlabel")
    target_inputs = get_inputs(args, data_type_file="train")
    # Create a set to avoid duplicates and improve performance
    target_domain_words = set(" ".join(target_inputs).split())
    # Extend with tag tokens
    target_domain_words.update(prepare_tag_tokens(args))
    # Create a set to avoid duplicates and improve performance
    target_domain_words = set(" ".join(target_inputs).split())
    # Extend with tag tokens

    target_domain_words.update(prepare_tag_tokens(args))
    logger.info(f"{len(target_domain_words)} target domain words")
    target_domain_words = list(target_domain_words)

    # add punctuatio words and stop words
    import string
    target_domain_words.extend(list(string.punctuation))
    target_domain_words.extend(STOP_WORDS)

    return target_domain_words


def model_filter(args, inputs, outputs):

    extract_path = os.path.join(args.seed_dir, f"extract_{args.task}-model")
    model2 = AutoModelForSeq2SeqLM.from_pretrained(extract_path).to(args.device)
    tokenizer2 = AutoTokenizer.from_pretrained(extract_path, use_fast=False)
    logger.info(f"{extract_path} loaded.")
    logger.info(f"Model emb weights of <pad> {model2.shared.weight[0][:5]}")

    filter_dataset = ABSADataset(args, tokenizer2, inputs=inputs, targets=outputs, name="target_filter")
    filter_inputs, filter_outputs, _  = infer(
        args, filter_dataset, model2, tokenizer2,
        name=f"target_filter", is_constrained=True, constrained_vocab=prepare_tag_tokens(args)
    )

    assert len(filter_inputs) == len(inputs)

    removed = []

    new_inputs, new_outputs = [], []
    filter_num = 0
    for i in range(len(outputs)):
        if filter_outputs[i].strip() != outputs[i].strip(): ### loại bỏ whitespace trong string
            # if predict none, then use generated results to allow more tgt domain exploration
            # if args.model_filter_skip_none and args.task in ["ate", "uabsa"]:
            #         if "none" in filter_outputs[i].strip():
            #             new_inputs.append(inputs[i])
            #             new_outputs.append(outputs[i])
            #             continue
            filter_num += 1
            removed.append(' #### '.join([inputs[i], outputs[i], filter_outputs[i]]))
            continue
        else:
            new_inputs.append(inputs[i])
            new_outputs.append(outputs[i])

    logger.info(f"{filter_num} augmentations out of {len(inputs)} are removed by model.")

    with open(os.path.join(args.inference_dir, f"model_filter.txt"), "w") as f:
        for i, o in enumerate(removed):
            f.write(f"{o}\n")

    return new_inputs, new_outputs