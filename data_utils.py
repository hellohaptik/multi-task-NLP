# Copyright (c) Microsoft. All rights reserved.

from enum import IntEnum
from transformers import *
from models.loss import *

NLP_MODELS = {
    "bert": (BertConfig, BertModel, BertTokenizer, 'bert-base-uncased'),
    "distilbert": (DistilBertConfig, DistilBertModel, DistilBertTokenizer, 'distilbert-base-uncased')
}
LOSSES = {
    "crossentropyloss" : CrossEntropyLoss,
    "spanloss" : SpanLoss
}

class ModelType(IntEnum):
    BERT = 1
    DISTILBERT = 2

class TaskType(IntEnum):
    SingleSenClassification = 1
    SentencePairClassification = 2
    Span = 3

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

