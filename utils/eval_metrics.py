"""
File for creating metric functions
"""
from sklearn.metrics import accuracy_score, f1_score
from seqeval.metrics import f1_score as seq_f1
from seqeval.metrics import precision_score, recall_score

def accuracy_(yTrue, yPred):
    """
    Accuracy score for classification tasks using the label provided in file and predictions from multi-task model.
    It takes a batch of predictions and labels.

    To use this metric, add **accuracy** into list of ``metrics`` in task file.

    Args:
        yPred (:obj:`list`) : [0, 2, 1, 3]
        yTrue (:obj:`list`) : [0, 1, 2, 3]

    """
    return accuracy_score(yTrue, yPred)*100

def f1_score_(yTrue, yPred):
    """
    Standard f1 score from sklearn for classification tasks.
    It takes a batch of predictions and labels.

    To use this metric, add **f1_score** into list of ``metrics`` in task file.

    Args:
        yPred (:obj:`list`) : [0, 2, 1, 3]
        yTrue (:obj:`list`) : [0, 1, 2, 3]

    """
    return f1_score(yTrue, yPred, average='micro')

def f1_score_ner(yTrue, yPred):
    """
    yPred = [ [0, 3, 4, 7], [4, 1, 3, 5], ...]
    yTrue = [ [0, 3, 2, 7], [2, 1, 3, 3], ...]
    """
    avgF1 = []
    for pred, actual in zip(yPred, yTrue):
        assert len(pred) == len(actual), "len of true label doesnt match actual label"
        avgF1.append(f1_score(pred, actual, average='micro'))
    avgF1F = sum(avgF1) / len(avgF1)
    return avgF1F

def accuracy_ner(yTrue, yPred):
    """
    yPred = [ [0, 3, 4, 7], [4, 1, 3, 5], ...]
    yTrue = [ [0, 3, 2, 7], [2, 1, 3, 3], ...]
    """
    avgAcc = []
    for pred, actual in zip(yPred, yTrue):
        assert len(pred) == len(actual), "len of true label doesnt match actual label"
        avgAcc.append(accuracy_score(pred, actual))
    avgAccF = sum(avgAcc) / len(avgAcc)
    return avgAccF*100

def seq_f1_score(yTrue, yPred):
    """
    f1 score for NER/sequence labelling tasks taken from the `seqeval <https://github.com/chakki-works/seqeval>`_ library.
    
    To use this metric, add **seq_f1** into list of ``metrics`` in task file.

    Args:
        y_true (:obj:`list of list`) : [['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        y_pred (:obj:`list of list`) : [['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
    """
    return seq_f1(yTrue, yPred)

def seq_precision(yTrue, yPred):
    """
    Precision score for NER/sequence labelling tasks taken from the `seqeval <https://github.com/chakki-works/seqeval>`_ library.
    
    To use this metric, add **seq_precision** into list of ``metrics`` in task file.

    Args:
        y_true (:obj:`list of list`) : [['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        y_pred (:obj:`list of list`) : [['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
    """
    return precision_score(yTrue, yPred)

def seq_recall(yTrue, yPred):

    """
    Recall score for NER/sequence labelling tasks taken from the `seqeval <https://github.com/chakki-works/seqeval>`_ library.
    
    To use this metric, add **seq_recall** into list of ``metrics`` in task file.

    Args:
        y_true (:obj:`list of list`) : [['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        y_pred (:obj:`list of list`) : [['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
    """
    return recall_score(yTrue, yPred)
