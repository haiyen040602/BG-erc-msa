T5_ORI_LEN = 32100

TAG_TO_WORD = {'POS': 'positive', 'NEG': 'negative', 'NEU': 'neutral'}
TAG_WORD_LIST = ['positive', 'negative', 'neutral']

# annotation-bracket
TAG_TO_BRACKET = {"POS": ("((", "))"), "NEG": ("[[", "]]"), "NEU": ("{{", "}}")}
BRACKET_TO_TAG = {"{{": "neutral", "[[": "negative", "((": "positive"}

# annotation-special
NONE_TOKEN = "[none]"
AND_TOKEN = "[and]"
ASPECT_TOKEN = "<aspect>"
OPINION_TOKEN = "<opinion>"
EMPTY_TOKEN = "<empty>"
SEP_TOKEN = "<sep>"
PADDING_TOKEN = "<pad>"
SPECIAL_TOKEN = ["<pad>", "<eos>", "<unk>"]
TAG_TO_SPECIAL = {"POS": ("<pos>", "</pos>"), "NEG": ("<neg>", "</neg>"), "NEU": ("<neu>", "</neu>")}
SPECIAL_TO_TAG = {"<pos>": "positive", "<neg>": "negative", "<neu>": "neutral"}

TARGET_TEST_COUNT_DICT = {
    "laptop": 800,
    "rest": 2158,
    "device": 1279,
    "service": 747,
}

STOP_WORDS = ['about', 'itself', 'so', 'further', 'against', "don't", 'shouldn', 'to', 'didn', 'hers', 'over', 'haven', "it's", 'of', 'have', 'm', 'but', "you've", 'which', 'd', 'most', 'nor', "haven't", "wasn't", 'yourself', 'with', 'am', 'do', 'than', "that'll", "isn't", 'or', "shan't", 'then', 'while', 'did', 'off', 'under', "mustn't", "won't", 'again', 'you', 'its', 'these', 'some', 'he', 'after', 'doesn', 'into', 't', 'more', 'whom', 'his', 'from', 'a', 'at', 'during', 'when', "she's", "aren't", 'was', 'same', 'myself', 'my', 'has', 'aren', 'by', 'before', "needn't", 'yourselves', 'such', 'she', 'is', 'needn', 'here', 'too', 'ourselves', "didn't", 'both', 'i', 'theirs', 'weren', 'be', 'their', 'were', 'because', 'should', "should've", "couldn't", 'will', 'isn', 'all', 'and', 'through', 'won', "weren't", 'y', 'they', 'for', 'until', 'him', 's', 'now', 'those', 'up', 'had', 'that', 'ma', 'couldn', 'been', 'why', 'below', 'own', 'doing', "you'll", 'very', 'above', "shouldn't", 'where', 've', 'if', 'are', 'how', 'wasn', 'it', 'what', 'as', 'hadn', 'hasn', "you'd", "wouldn't", 'don', 'few', 'other', 're', 'ain', "hadn't", "doesn't", 'himself', 'shan', 'the', 'not', 'mustn', 'does', "hasn't", 'll', 'your', 'yours', 'herself', 'in', 'wouldn', 'themselves', 'who', 'there', 'ours', 'out', 'mightn', 'me', 'them', 'once', "mightn't", 'we', 'her', 'this', 'being', 'any', 'can', 'o', 'no', 'having', "you're", 'our', 'on', 'between', 'down', 'only', 'just', 'each', 'an']
# STOP_WORDS = []


IEMOCAP_POS = ['joy', 'excited', 'surprise', 'neutral']
IEMOCAP_NEG = ['sadness', 'angry', 'frustrated', 'surprise', 'anger', 'disgust', 'fear', 'neutral']
IEMOCAP_NEU = ['neutral', 'surprise']

MELD_POS = ['joy', 'surprise', 'neutral']
MELD_NEG = ['sadness', 'angry', 'surprise', 'disgust', 'fear', 'neutral']
MELD_NEU = ['surprise']

MELD_IEMOCAP = ['neutral', 'joy', 'sadness']

MOSEII_TAGS = ["<pos>", "<neg>", "<neu>", "<score>", "</score>"]
MOSEII_DICT = ['neu', 'neg', 'pos']
# MELD_TAGS = ["<emotion>"]
MELD_LABELS = ['<meld>', 'joy', 'sadness', 'fear', 'anger', 'surprise', 'disgust', 'neutral']
MELD_DICT = ['neutral', 'surprise', 'anger', 'disgust', 'fear', 'joy', 'sadness']
MELD_DICT_WO_NEUTRAL = ['surprise', 'anger', 'disgust', 'fear', 'joy', 'sadness']

# IEMOCAP_TAGS =["<emotion>"]
IEMOCAP_LABELS = ['<iemocap>', 'joy', 'sadness', 'angry', 'neutral', 'excited', 'frustrated']
IEMOCAP_DICT = ['neutral', 'excited', 'angry', 'joy', 'sadness', 'frustrated']
MELD_TRANSFER_PAIRS = {
    "meld": ["iemocap", "iemocap_context", "moseii"]
}

MELD_CONTEXT_TRANSFER_PAIRS = {
    "meld_context": ["iemocap", "iemocap_context", "moseii"]
}

MOSEII_TRANSFER_PAIRS = {
    "moseii": ["meld", "meld_context", "iemocap", "iemocap_context"]
}

IEMOCAP_TRANSFER_PAIRS = {
    "iemocap": ["meld", "meld_context", "moseii"]
}

IEMOCAP_CONTEXT_TRANSFER_PAIRS = {
    "iemocap_context": ["meld", "meld_context", "moseii"]
}
