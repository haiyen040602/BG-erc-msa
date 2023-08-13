
import logging

import torch

from constants import *

logger = logging.getLogger(__name__)

def prepare_constrained_tokens(tokenizer, task, paradigm):
    special_tokens = [tokenizer.eos_token] # add end token
    if "moseii" in task:
        if paradigm == "extraction-universal":
            # note space will affect tokenization, make sure the special tokens align with processed method considering space
            special_tokens += MOSEII_TAGS
        else:
            raise NotImplementedError
    elif "meld" in task:
        if paradigm == "extraction-universal":
            # special_tokens += MELD_TAGS
            special_tokens += MELD_LABELS
            special_tokens += [SEP_TOKEN]
        else:
            raise NotImplementedError
    elif "iemocap" in task:
        if paradigm == "extraction-universal":
            # special_tokens += IEMOCAP_TAGS
            special_tokens += IEMOCAP_LABELS
            special_tokens += [SEP_TOKEN]
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
    return special_tokens

### Chuẩn bị các tag token cho từng task
def prepare_tag_tokens(args):
    tag_tokens = []
    if "moseii" in args.task:
        if "extraction-universal" in args.paradigm:
            tag_tokens += MOSEII_TAGS
        if args.data_gene: ### nếu làm nhiệm vụ generate label
            tag_tokens += MOSEII_TAGS
            tag_tokens += [SEP_TOKEN] 
    elif "meld" in args.task: ### label: <aspect> a
        if args.paradigm == "extraction-universal":
            # tag_tokens += MELD_TAGS
            tag_tokens += MELD_LABELS
            tag_tokens += [SEP_TOKEN] 
    elif "iemocap" in args.task : ### label: <POS> a <opinion> a
        if "extraction-universal" in args.paradigm:
            # tag_tokens += IEMOCAP_TAGS
            tag_tokens += IEMOCAP_LABELS
            tag_tokens += [SEP_TOKEN] 
    else:
        raise NotImplementedError

    tag_tokens = list(set(tag_tokens))
    logger.info(f"Tag tokens: {tag_tokens}")
    return tag_tokens


def init_tag(args, tokenizer, model, tag_tokens):
    if args.init_tag == "english":
        if args.paradigm == "extraction-universal":
            import re
            map_dict = {"pos": "positive", "neg": "negative", "neu": "neutral", "meld": "label", "iemocap": "label"} #, "meld": "label", "iemocap": "label"
            for tag_word in tag_tokens:
                tag_id = tokenizer.encode(tag_word, add_special_tokens=False)[0]
                init_word = re.sub("\W", "", tag_word).strip()
                print("tag word: ", tag_word, "init_word: ", init_word)
                # map senti
                if init_word in map_dict:
                    init_word = map_dict[init_word]
                # skip sep
                elif init_word == "sep":
                    continue
                init_id = tokenizer.encode(init_word, add_special_tokens=False)[0]
                with torch.no_grad():
                    model.shared.weight[tag_id] = model.shared.weight[init_id]
                logger.info(f"{tokenizer.decode(tag_id)} is init by {tokenizer.decode(init_id)}")
        elif args.paradigm == "extraction":
            pass
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError