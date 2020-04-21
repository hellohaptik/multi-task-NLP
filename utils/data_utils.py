from enum import IntEnum
from transformers import *
from models.loss import *

NLP_MODELS = {
    "bert": (BertConfig, BertModel, BertTokenizer, 'bert-base-uncased'),
    "distilbert": (DistilBertConfig, DistilBertModel, DistilBertTokenizer, 'distilbert-base-uncased'),
    "albert": (AlbertConfig, AlbertModel, AlbertTokenizer, 'albert-base-v2'),
    "roberta": (RobertaConfig, RobertaModel, RobertaTokenizer, 'roberta-base'),
    "xlnet" : (XLNetConfig, XLNetModel, XLNetTokenizer, 'xlnet-base-cased')
}
LOSSES = {
    "crossentropyloss" : CrossEntropyLoss,
    "spanloss" : SpanLoss
}

class ModelType(IntEnum):
    BERT = 1
    DISTILBERT = 2
    ALBERT = 3
    ROBERTA = 4
    XLNET = 5

class TaskType(IntEnum):
    SingleSenClassification = 1
    SentencePairClassification = 2
    NER = 3
    Span = 4

class MetricType(IntEnum):
    #accuracy for classification task
    ACC = 0
    # Mathew's correlation coefficient for binary classification
    MCC = 1
    # F1 score
    F1 = 3
    # Exact match/ f1 score for span task
    EmF1 = 4

class LossType(IntEnum):
    CrossEntropyLoss = 0
    SpanLoss = 1

