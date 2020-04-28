from enum import IntEnum
from transformers import *
from models.loss import *
from utils.eval_metrics import *

NLP_MODELS = {
    "bert": (BertConfig, BertModel, BertTokenizer, 'bert-base-uncased'),
    "distilbert": (DistilBertConfig, DistilBertModel, DistilBertTokenizer, 'distilbert-base-uncased'),
    "albert": (AlbertConfig, AlbertModel, AlbertTokenizer, 'albert-base-v2'),
    "roberta": (RobertaConfig, RobertaModel, RobertaTokenizer, 'roberta-base'),
    "xlnet" : (XLNetConfig, XLNetModel, XLNetTokenizer, 'xlnet-base-cased'),
    "electra" : (ElectraConfig, ElectraModel, ElectraTokenizer, 'google/electra-small-generator')
}
LOSSES = {
    "crossentropyloss" : CrossEntropyLoss,
    "spanloss" : SpanLoss,
    "nerloss" : NERLoss
}

METRICS = {
    "accuracy": accuracy_,
    "f1_score": f1_score_,
    "ner_accuracy": accuracy_ner,
    "ner_f1_score": f1_score_ner,
    "seq_f1" : seq_f1_score,
    "seq_precision" : seq_precision,
    "seq_recall" : seq_recall
}

class ModelType(IntEnum):
    BERT = 1
    DISTILBERT = 2
    ALBERT = 3
    ROBERTA = 4
    XLNET = 5
    ELECTRA = 6

class TaskType(IntEnum):
    SingleSenClassification = 1
    SentencePairClassification = 2
    NER = 3
    Span = 4

class LossType(IntEnum):
    CrossEntropyLoss = 0
    SpanLoss = 1
    NERLoss = 2

