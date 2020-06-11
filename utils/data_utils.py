from enum import IntEnum
from transformers import *
from models.loss import *
from utils.eval_metrics import *
from utils.tranform_functions import *

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
    "nerloss" : NERLoss
}

METRICS = {
    "classification_accuracy": classification_accuracy,
    "classification_f1_score": classification_f1_score,
    "seqeval_f1_score" : seqeval_f1_score,
    "seqeval_precision" : seqeval_precision,
    "seqeval_recall" : seqeval_recall,
    "snips_f1_score" : snips_f1_score,
    "snips_precision" : snips_precision,
    "snips_recall" : snips_recall,
    "classification_recall" : classification_recall
}

TRANSFORM_FUNCS = {
    "snips_intent_ner_to_tsv" : snips_intent_ner_to_tsv,
    "coNLL_ner_pos_to_tsv" : coNLL_ner_pos_to_tsv,
    "snli_entailment_to_tsv" : snli_entailment_to_tsv,
    "bio_ner_to_tsv" : bio_ner_to_tsv,
    "create_fragment_detection_tsv" : create_fragment_detection_tsv,
    "msmarco_query_type_to_tsv" : msmarco_query_type_to_tsv,
    "imdb_sentiment_data_to_tsv" : imdb_sentiment_data_to_tsv,
    "qqp_query_similarity_to_tsv" : qqp_query_similarity_to_tsv,
    "msmarco_answerability_detection_to_tsv" : msmarco_answerability_detection_to_tsv,
    "clinc_out_of_scope_to_tsv" : clinc_out_of_scope_to_tsv
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

class LossType(IntEnum):
    CrossEntropyLoss = 0
    NERLoss = 1

