import argparse
import logging
import os
import random
from datetime import datetime

from constants import *

logger = logging.getLogger(__name__)

def init_args():
    parser = argparse.ArgumentParser()
    # basic settings
    parser.add_argument("--task", default='meld', type=str, required=True,
                        help="The name of the task, selected from: [xabsa]")
    parser.add_argument("--targets", default=None, type=str,
                        help="The name of the target of task, selected from: [xabsa]")
    parser.add_argument("--device", default='cuda', type=str, required=False)
    parser.add_argument("--dataset", default='cross_domain', type=str, required=True,
                        help="The name of the dataset, selected from: [cross_domain]")
    parser.add_argument("--data_dir", default=None, type=str, help="args.output_dir/data")
    parser.add_argument("--model_name_or_path", default='t5-base', type=str,
                        help="Path to pre-trained model or shortcut name")
    parser.add_argument("--paradigm", default='extraction-universal', type=str, required=True,
                        help="""The way to construct target sentence, selected from:[
                            extraction:             (apple, positive); (banana, negative) //  None
                            extraction-universal:  <pos> apple <pos> orange <neg> banana //   [none]
                            ]""")
    parser.add_argument("--do_train", action='store_false', help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_false', help="Whether to run eval on the dev/test set.")
    parser.add_argument("--name", default='experinments', type=str, help="name of the exp")
    parser.add_argument("--commit", default=None, type=str, help="commit id")
    parser.add_argument("--save_last_k", default=1, type=int, help="save last k")
    parser.add_argument("--n_runs", default=1, type=int, help="run with n seeds")
    parser.add_argument("--clear_model", action='store_true', help="remove saved ckpts")
    parser.add_argument("--save_best", action='store_true', help="save best model only")
    parser.add_argument("--nrows", default=None, type=int, help="# of lines to be read")

    # basic setting
    parser.add_argument("--train_by_pair", action='store_false', help="train a model for each pair")
    parser.add_argument("--target_domain", default=None, type=str)

    # inference
    parser.add_argument("--no_greedy", action='store_true', help="only constrained decoding")
    parser.add_argument("--beam", default=1, type=int)

    # Other parameters
    parser.add_argument("--max_seq_length", default=128, type=int)
    parser.add_argument("--n_gpu", default='0')
    parser.add_argument("--train_batch_size", default=16, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=128, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=3e-4, type=float)
    parser.add_argument("--num_train_epochs", default=10, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")

    # training details
    parser.add_argument("--weight_decay", default=0.0, type=float)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float)
    parser.add_argument("--warmup_steps", default=0.0, type=float)

    # template
    parser.add_argument("--init_tag", default='english', type=str, help="[english]")

    # data genenration
    parser.add_argument("--data_gene", action='store_true', help="enable data generation")
    parser.add_argument("--data_gene_epochs", default=10, type=int, help="epoch of generation model training")
    parser.add_argument("--data_gene_num_samples", default=10, type=int, help="number of samples generated in generation phrase")
    parser.add_argument("--data_gene_extract", action='store_true', help="enable text-to-label stage")
    parser.add_argument("--data_gene_extract_epochs", default=10, type=int, help="epoch of extract model training")
    parser.add_argument("--data_gene_none_remove_ratio", default=0, type=float, help="remove none input for gene model")
    parser.add_argument("--data_gene_none_word_num", default=1, type=int, help="rand word added to generate diverse none examples")
    parser.add_argument("--data_gene_extract_none_remove_ratio", default=0, type=float, help="remove none training sample for extract model training")
    parser.add_argument("--data_gene_same_model", action='store_true', help="extract & gene model share the same model")
    parser.add_argument("--data_gene_base_model", action='store_true', help="enable data generation by gpt base model")
    parser.add_argument("--data_gene_affective_model", action='store_true', help="enable data generation by affective generation")
    parser.add_argument("--num_input_prompt", default=2, type=int)
    parser.add_argument("--max_gene_length", default=50, type=int)
    parser.add_argument("--data_extract_wt_constrained", action='store_true', help="enable constrained decoding during extraction task")
    parser.add_argument("--data_gene_wt_constrained", action='store_true', help="turn off constrained decoding during gene generation")
    parser.add_argument("--use_same_model", action='store_false', help="all stages use the same model")
    parser.add_argument("--model_filter", action='store_true', help="use extract model inference for filtering")
    parser.add_argument("--model_filter_skip_none", action='store_true', help="keep the sample if extract model output none")

    # decode
    parser.add_argument("--data_gene_decode", default=None, type=str, help="[greedy, top_p, beam]")
    parser.add_argument("--data_gene_top_p", default=0.9, type=float)
    parser.add_argument("--data_gene_num_beam", default=1, type=int)
    parser.add_argument("--data_gene_min_length", default=0, type=int)
    parser.add_argument("--extract_model", default=None, type=str, help="path to extract model")
    parser.add_argument("--gene_model", default="gpt2-medium", type=str, help="path to gene model")

    parser.add_argument("--runned_folder", default=None, type=str, help="Load previous trained model for aux training")
    parser.add_argument("--data_gene_aug_num", default=None, type=int, help="how many augmentation samples")
    parser.add_argument("--data_gene_aug_ratio", default=None, type=float, help="how much ratio of augmentation samples")

    # pseudo labelling
    parser.add_argument("--pseudo", action='store_true', help="data extraction")
    parser.add_argument("--pseudo_skip_none", action='store_true', help="use non-none only")

    args = parser.parse_args()

    # Set up output directory
    output_dir = f"../outputs/{args.task}/{args.dataset}"
    os.makedirs(output_dir, exist_ok=True)

    # Create a timestamped directory
    timestamp = datetime.now().strftime("%m%d_%H%M")
    output_dir = os.path.join(output_dir, f"{timestamp}-{args.name}")
    os.makedirs(output_dir, exist_ok=True)

    args.output_dir = output_dir

    return args


def prepare_seeds(seed=42, n_runs=1):
    # prepare n seeds
    random.seed(seed)
    seed_list = random.sample(range(100), n_runs-1)
    seed_list += [seed]
    seed_list = sorted(seed_list)
    print("Seed list: {}".format(seed_list))
    return seed_list


def prepare_pairs(args):
    # Define the mapping between tasks, datasets, and pairs
    task_dataset_pairs = {
        "moseii": {"cross_domain": MOSEII_TRANSFER_PAIRS},
        "meld": {"cross_domain": MELD_TRANSFER_PAIRS},
        "iemocap": {"cross_domain": IEMOCAP_TRANSFER_PAIRS},
        "meld_context": {"cross_domain": MELD_CONTEXT_TRANSFER_PAIRS},
        "iemocap_context": {"cross_domain": IEMOCAP_CONTEXT_TRANSFER_PAIRS}
    }

    task_dataset_pairs_small = {
        "moseii": {"cross_domain_small": MOSEII_TRANSFER_PAIRS},
        "meld": {"cross_domain_small": MELD_TRANSFER_PAIRS},
        "iemocap": {"cross_domain_small": IEMOCAP_TRANSFER_PAIRS},
        "meld_context": {"cross_domain_small": MELD_CONTEXT_TRANSFER_PAIRS},
        "iemocap_context": {"cross_domain_small": IEMOCAP_CONTEXT_TRANSFER_PAIRS}
    }

    # Check if the task and dataset combination is valid and retrieve the corresponding pair
    task = str(args.task)   
    datasets = str(args.dataset)
    if args.targets is not None:
        ls = []
        target = str(args.targets)
        ls.append(target)
        pair_dict = {task : ls}
    elif "small" in datasets:
        pair_dict = task_dataset_pairs_small.get(task, {}).get(datasets)
    else:
        pair_dict = task_dataset_pairs.get(task, {}).get(datasets)
    print(pair_dict)
    if pair_dict is None:
        raise NotImplementedError("The task and/or dataset is not implemented.")

    return pair_dict


