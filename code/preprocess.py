import os
import re
import logging

from constants import *

logger = logging.getLogger(__name__)


def read_line_from_file(file_path):
    sentences = []
    with open(file_path, 'r', encoding='UTF-8') as fp:
        for line in fp:
            sentences.append(line)
    return sentences

def write_generation(file_paths, split="train", data_dir=None, do_write=True, nrows=None, mode='iemocap'):

    final_sents = []
    count_dict = {}
    for path in file_paths:

        sentences, domain = None, None
    
        sentences = read_line_from_file(path)[:nrows] ### format lại label của tập uabsa
        if split in ["dev", "test"]:
            domain = re.search(f"(\w+)[_-]{split}\.txt", path).group(1)

        count_dict[domain] = len(sentences)
        final_sents.extend(sentences)
        logger.info(f"{split} {path}: {len(sentences)}")

    logger.info(f"{split} total: {len(final_sents)}")
    logger.info(f"{split} count dict: {count_dict}")

    # lưu dữ liệu sang tập mới
    if do_write:
        output_dir = data_dir

        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        output_path = f'{output_dir}/{split}.txt'
        with open (output_path, 'w', encoding="utf-8") as f:
            for s in final_sents:
                f.write (f'{s.strip()}\n')
            logger.info(f"{output_path} is written.")

    return final_sents, count_dict


def preprocess(dataset_dir=None, data_dir=None, source=None, targets=None, do_write=True, nrows=None, unlabel=False, mode='iemocap'):

    if mode:
        train_file_format = "{i}_train.txt"
        dev_file_format = f"{source}_dev.txt"
        test_file_format = "{i}_test.txt"
    else:
        raise ValueError("Invalid mode. Choose either 'uabsa' or 'aste'.")
    
    ## Chia dữ liệu unlabeled như thế nào?
    if unlabel:
        # Process unlabeled data
        train_paths = [os.path.join(dataset_dir, train_file_format.format(i=i)) for i in targets]
        train_sents, _ = write_generation(train_paths, "target-unlabel", data_dir=data_dir, do_write=do_write, nrows=nrows, mode=mode)
        logger.info(f"Cross domain unlabel train_paths: {train_paths}")
        return train_sents

    else:
        # Process labeled data
        train_paths = [os.path.join(dataset_dir, train_file_format.format(i=i)) for i in [source]]
        dev_paths = [os.path.join(dataset_dir, dev_file_format)]
        test_paths = [os.path.join(dataset_dir, test_file_format.format(i=i)) for i in targets]

        logger.info(f"train_paths: {train_paths}")
        logger.info(f"dev_paths: {dev_paths}")
        logger.info(f"test_paths: {test_paths}")

        # Read and preprocess the data using write_generation
        ### Format lại label của raw data
        _, _ = write_generation(train_paths, "train", data_dir=data_dir, do_write=do_write, nrows=nrows, mode=mode)
        _, dev_count_dict = write_generation(dev_paths, "dev", data_dir=data_dir, do_write=do_write, nrows=nrows, mode=mode)
        _, test_count_dict = write_generation(test_paths, "test", data_dir=data_dir, do_write=do_write, nrows=nrows, mode=mode)

        return dev_count_dict, test_count_dict


def prepare_raw_data(args):

    def process_task(preprocessor, mode, dataset_dir=None, data_gene=False, pseudo=False):
        eval_count_dict, test_count_dict = preprocessor(
            dataset_dir=dataset_dir,
            data_dir=args.data_dir,
            source=args.source_domain,
            targets=args.target_domain,
            nrows=args.nrows,
            mode=mode
        )
        if data_gene or pseudo:
            preprocessor(
                dataset_dir=dataset_dir,
                data_dir=args.data_dir,
                source=args.source_domain,
                targets=args.target_domain,
                nrows=args.nrows,
                unlabel=True,
                mode=mode
            )
        return eval_count_dict, test_count_dict

    # ate and uabsa share same dataset
    task_map = {
        "moseii": ('moseii',  "../data/cross_domain/moseii"),
        "meld": ('meld',  "../data/cross_domain/meld"),
        "iemocap": ('iemocap', "../data/cross_domain/iemocap"), 
    }

    task_map_small = {
        "moseii": ('moseii',  "../data/cross_domain_small/moseii"),
        "meld": ('meld',  "../data/cross_domain_small/meld"),
        "iemocap": ('iemocap', "../data/cross_domain_small/iemocap"),
        "meld_context": ('meld',  "../data/cross_domain_small/meld_context"),
        "iemocap_context": ('iemocap', "../data/cross_domain_small/iemocap_context"),
    }
   
    if args.task in task_map and args.dataset == "cross_domain":
        mode, dataset_dir = task_map[args.task]
        return process_task(preprocess, mode, dataset_dir, args.data_gene, args.pseudo)
    elif args.task in task_map_small and args.dataset == "cross_domain_small":
        mode, dataset_dir = task_map_small[args.task]
        return process_task(preprocess, mode, dataset_dir, args.data_gene, args.pseudo)
    else:
        raise NotImplementedError
