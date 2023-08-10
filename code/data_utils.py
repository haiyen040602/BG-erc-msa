# This file contains all data loading and transformation functions
import logging
import random
import re

from torch.utils.data import Dataset

from constants import *

logger = logging.getLogger(__name__)

senttag2word = {'POS': 'positive', 'NEG': 'negative', 'NEU': 'neutral'}


def filter_invalid(inputs, outputs):
    new_inputs, new_outputs = [], []
    rm_num = 0
    for idx in range(len(inputs)):
        valid = True
        aps = re.findall("(<\w+>)(.*?)(?=<\w+>|$)", outputs[idx])
        for ap in aps:
            for ele in ap:
                # some element is missing
                if len(ele) == 0:
                    valid = False
                    break
            if valid is False:
                break
        if valid:
            new_inputs.append(inputs[idx])
            new_outputs.append(outputs[idx])
        else:
            rm_num += 1
            logger.info(f"Output is invalid: {outputs[idx]}")
    logger.info(f"Filterd out {rm_num} invalid samples")
    return new_inputs, new_outputs


def filter_none(inputs, outputs, ratio=0):
    if ratio > 0:
        new_inputs = []
        new_outputs = []
        rm_num = 0
        none_num = 0
        import random
        for idx in range(len(inputs)):
            if outputs[idx].strip() == NONE_TOKEN:
                none_num += 1
                if random.random() < ratio:
                    rm_num += 1
                    continue
            new_inputs.append(inputs[idx])
            new_outputs.append(outputs[idx])
        logger.info(f"Filtered out {rm_num} [none] samples out of {none_num} [none]")
        return new_inputs, new_outputs
    else:
        return inputs, outputs


def normalize_augment(args, inputs, outputs):
    new_inputs = []
    new_outputs = []
    rm_num = 0
    for idx in range(len(inputs)):
        # rm added random word
        # if args.data_gene_none_word_num > 0 and NONE_TOKEN in outputs[idx]:
        #     if args.task in ["uabsa", "ate"]:
        #         # just keep none token
        #         outputs[idx] = NONE_TOKEN
        new_inputs.append(inputs[idx])
        new_outputs.append(outputs[idx])
    return new_inputs, new_outputs


def read_line_examples_from_file(data_path):
    """
    Read data from file, each line is: sent####labels
    Return List[List[word]], List[Tuple]
    """
    sents, labels = [], []
    with open(data_path, 'r', encoding='UTF-8') as fp:
        words, labels = [], []
        for line in fp:
            line = line.strip()
            if line != '':
                words, tuples = line.split('####')
                ### sent lưu mảng các từ trong câu
                sents.append(words.split())
                labels.append(tuples)
    logger.info(f"{data_path.split('/')[-1]}\tTotal examples = {len(sents)} ")
    return sents, labels



class ABSADataset(Dataset):
    def __init__(self, args, tokenizer, inputs=None, targets=None, dataset_list=None, name=None):
        self.args = args
        self.tokenizer = tokenizer
        self.inputs, self.targets = inputs or [], targets or []
        self.inputs_tensor_list, self.targets_tensor_list = [], []
        self.name = name or []

        # If a dataset_list is provided, merge the datasets
        if dataset_list:
            for d in dataset_list:
                self.inputs += d.inputs
                self.targets += d.targets
                self.inputs_tensor_list += d.inputs_tensor_list
                self.targets_tensor_list += d.targets_tensor_list

                if isinstance(d.name, list):
                    self.name.extend(d.name)
                else:
                    self.name.append(d.name)

            # Write merged data to a file
            processed_path = f"{d.args.data_dir}/{'+'.join(self.name)}_merged_processed.txt"
            with open(processed_path, "w") as f:
                for i in range(len(self.inputs)):
                    f.write(f"{self.inputs[i]} ===> {self.targets[i]}\n")
            logger.info(f"{self.name} is merged.")
            logger.info(f"{processed_path} is written.")
        else:
            # If no dataset_list is provided, behave like ABSADataset
            self.max_len = args.max_seq_length
            self.inputs_tensor_list, self.targets_tensor_list = self.encode(self.inputs, self.targets)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        source_ids = self.inputs_tensor_list[index]["input_ids"].squeeze()
        target_ids = self.targets_tensor_list[index]["input_ids"].squeeze()

        src_mask = self.inputs_tensor_list[index]["attention_mask"].squeeze()
        target_mask = self.targets_tensor_list[index]["attention_mask"].squeeze()

        return {"source_ids": source_ids, "source_mask": src_mask,
                "target_ids": target_ids, "target_mask": target_mask}

    def encode(self, inputs=[], targets=[]):
        inputs_tensor_list, targets_tensor_list = [], []

        for i in range(len(inputs)):
            input_i = ' '.join(inputs[i]) if isinstance(inputs[i], list) else inputs[i]
            target_i = ' '.join(targets[i]) if isinstance(targets[i], list) else targets[i]

            # Tokenize input and target data
            tokenized_input = self.tokenizer.batch_encode_plus(
                [input_i], max_length=self.max_len, pad_to_max_length=True, truncation=True,
                return_tensors="pt",
            )
            tokenized_target = self.tokenizer.batch_encode_plus(
                [target_i], max_length=self.max_len, pad_to_max_length=True, truncation=True,
                return_tensors="pt"
            )

            inputs_tensor_list.append(tokenized_input)
            targets_tensor_list.append(tokenized_target)

        return inputs_tensor_list, targets_tensor_list


def get_inputs(args, data_type_file="train"):
    """
        train_inputs: ["hi", "I love apples."],
    """
    data_path = f"{args.data_dir}/{data_type_file}.txt"
    inputs, _ = read_line_examples_from_file(data_path)
    inputs = [" ".join(i) for i in inputs]
    return inputs


def prepare_moseii_gene(args, data_type_file):
    targets, inputs = prepare_moseii_extraction(args, data_type_file=data_type_file)
    return inputs, targets

def prepare_meld_gene(args, data_type_file):
    targets, inputs = prepare_moseii_extraction(args, data_type_file=data_type_file)
    return inputs, targets

def prepare_iemocap_gene(args, data_type_file):
    targets, inputs = prepare_moseii_extraction(args, data_type_file=data_type_file)
    return inputs, targets

def get_generation_inputs_and_targets(args, task, data_type):
    if task not in ["gene_moseii", "gene_meld", "gene_iemocap"]:
        raise NotImplementedError(f"Task {task} is not supported.")

    if data_type == "train":
        prepare_fn = {
            "gene_moseii": prepare_moseii_gene,
            "gene_meld": prepare_meld_gene,
            "gene_iemocap": prepare_iemocap_gene
        }[task]
        inputs, targets = prepare_fn(args, data_type_file="train")
    else:
        raise NotImplementedError(f"Data type {data_type} is not supported.")

    return inputs, targets

def prepare_moseii_extraction(args, data_type_file="train"):
    data_path = f"{args.data_dir}/{data_type_file}.txt"
    sents, labels = read_line_examples_from_file(data_path)
    inputs = [" ".join(s) for s in sents]

    targets = []
    for i, label in enumerate(labels):
        targets.append(label)

    return inputs, targets

def prepare_meld_extraction(args, data_type_file="train"):
    data_path = f"{args.data_dir}/{data_type_file}.txt"
    sents, labels = read_line_examples_from_file(data_path)
    inputs = [" ".join(s) for s in sents]

    targets = []
    for i, label in enumerate(labels):
        targets.append(label)
    
    return inputs, targets

def prepare_iemocap_extraction(args, data_type_file="train"):    
    data_path = f"{args.data_dir}/{data_type_file}.txt"
    sents, labels = read_line_examples_from_file(data_path)
    inputs = [" ".join(s) for s in sents]

    targets = []
    for i, label in enumerate(labels):
        targets.append(label)
    
    return inputs, targets

def get_extraction_inputs_and_targets(args, task, data_type):
    task_mapping = {
        "moseii": {"extraction": prepare_moseii_extraction, "extraction-universal" : prepare_moseii_extraction},
        "extract_moseii": {"extraction": prepare_moseii_extraction, "extraction-universal" : prepare_moseii_extraction},
        "meld": {"extraction": prepare_meld_extraction, "extraction-universal" : prepare_meld_extraction},
        "extract_meld": {"extraction": prepare_meld_extraction, "extraction-universal" : prepare_meld_extraction},
        "iemocap": {"extraction": prepare_iemocap_extraction, "extraction-universal" : prepare_iemocap_extraction},
        "extract_iemocap": {"extraction": prepare_iemocap_extraction, "extraction-universal" : prepare_iemocap_extraction}
    }

    if task not in task_mapping:
        raise NotImplementedError

    if args.paradigm not in task_mapping[task]:
        raise NotImplementedError

    inputs, targets = task_mapping[task][args.paradigm](args, data_type_file=data_type)

    return inputs, targets



def get_inputs_and_targets(args, task, data_type):

    is_gene = True if "gene" in task else False
    if is_gene:
        inputs, targets = get_generation_inputs_and_targets(args, task, data_type)
    else:
        inputs, targets = get_extraction_inputs_and_targets(args, task, data_type)
        # if data_type == "train" and task in ["extract_ate", "extract_uabsa"]:
        #     # need to filter out too much none, encourage more pairs to be detected
        #     inputs, targets = filter_none(inputs, targets, args.data_gene_extract_none_remove_ratio)

    # rm label to prevent potential leakage
    if data_type == "target-unlabel":
        targets = ["Label removed." for t in targets]
        logger.info("Removed label for target-unlabel.")

    path = f"{args.data_dir}/{data_type}_{task}_processed.txt"
    save_inputs_and_targets(path, inputs, targets)

    return inputs, targets


def save_inputs_and_targets(path, inputs, targets):
    with open(path, "w") as f:
        for i in range(len(inputs)):
            f.write(f"{inputs[i]} ===> {targets[i]}\n")
    logger.info(f"{path} is written.")


def get_dataset(args, task, data_type, tokenizer):
    inputs, targets = get_inputs_and_targets(args, task, data_type)
    # for i in range(3):
    #     print("Input ", i, ": ", inputs[i])
    #     print("Output ", i, ": ", targets[i])
    dataset = ABSADataset(args, tokenizer, inputs=inputs, targets=targets, name=f"{data_type}_{task}")
    return dataset