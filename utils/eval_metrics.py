"""
File for creating metric functions
"""
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import recall_score as class_recall_score
from seqeval.metrics import f1_score as seq_f1
from seqeval.metrics import precision_score, recall_score

def classification_accuracy(yTrue, yPred):
    """
    Accuracy score for classification tasks using the label provided in file and predictions from multi-task model.
    It takes a batch of predictions and labels.

    To use this metric, add **classification_accuracy** into list of ``metrics`` in task file.

    Args:
        yPred (:obj:`list`) : [0, 2, 1, 3]
        yTrue (:obj:`list`) : [0, 1, 2, 3]

    """
    return accuracy_score(yTrue, yPred)*100

def classification_f1_score(yTrue, yPred):
    """
    Standard f1 score from sklearn for classification tasks.
    It takes a batch of predictions and labels.

    To use this metric, add **classification_f1_score** into list of ``metrics`` in task file.

    Args:
        yPred (:obj:`list`) : [0, 2, 1, 3]
        yTrue (:obj:`list`) : [0, 1, 2, 3]

    """
    return f1_score(yTrue, yPred, average='micro')

def classification_recall(yTrue, yPred):
    """
    Standard recall score from sklearn for classification tasks.
    It takes a batch of predictions and labels.

    To use this metric, add **classification_f1_score** into list of ``metrics`` in task file.

    Args:
        yPred (:obj:`list`) : [0, 2, 1, 3]
        yTrue (:obj:`list`) : [0, 1, 2, 3]

    """
    return class_recall_score(yTrue, yPred, average='micro')

def seqeval_f1_score(yTrue, yPred):
    """
    f1 score for NER/sequence labelling tasks taken from the `seqeval <https://github.com/chakki-works/seqeval>`_ library.
    
    To use this metric, add **seqeval_f1_score** into list of ``metrics`` in task file.

    Args:
        yTrue (:obj:`list of list`) : [['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        yPred (:obj:`list of list`) : [['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
    """
    return seq_f1(yTrue, yPred)

def seqeval_precision(yTrue, yPred):
    """
    Precision score for NER/sequence labelling tasks taken from the `seqeval <https://github.com/chakki-works/seqeval>`_ library.
    
    To use this metric, add **seqeval_precision** into list of ``metrics`` in task file.

    Args:
        yTrue (:obj:`list of list`) : [['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        yPred (:obj:`list of list`) : [['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
    """
    return precision_score(yTrue, yPred)

def seqeval_recall(yTrue, yPred):

    """
    Recall score for NER/sequence labelling tasks taken from the `seqeval <https://github.com/chakki-works/seqeval>`_ library.
    
    To use this metric, add **seqeval_recall** into list of ``metrics`` in task file.

    Args:
        yTrue (:obj:`list of list`) : [['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        yPred (:obj:`list of list`) : [['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
    """
    return recall_score(yTrue, yPred)


# compute f1 score is modified from conlleval.pl
def __startOfChunk(prevTag, tag, prevTagType, tagType, chunkStart = False):
    if prevTag == 'B' and tag == 'B':
        chunkStart = True
    if prevTag == 'I' and tag == 'B':
        chunkStart = True
    if prevTag == 'O' and tag == 'B':
        chunkStart = True
    if prevTag == 'O' and tag == 'I':
        chunkStart = True

    if prevTag == 'E' and tag == 'E':
        chunkStart = True
    if prevTag == 'E' and tag == 'I':
        chunkStart = True
    if prevTag == 'O' and tag == 'E':
        chunkStart = True
    if prevTag == 'O' and tag == 'I':
        chunkStart = True

    if tag != 'O' and tag != '.' and prevTagType != tagType:
        chunkStart = True
    return chunkStart

def __endOfChunk(prevTag, tag, prevTagType, tagType, chunkEnd = False):
    if prevTag == 'B' and tag == 'B':
        chunkEnd = True
    if prevTag == 'B' and tag == 'O':
        chunkEnd = True
    if prevTag == 'I' and tag == 'B':
        chunkEnd = True
    if prevTag == 'I' and tag == 'O':
        chunkEnd = True

    if prevTag == 'E' and tag == 'E':
        chunkEnd = True
    if prevTag == 'E' and tag == 'I':
        chunkEnd = True
    if prevTag == 'E' and tag == 'O':
        chunkEnd = True
    if prevTag == 'I' and tag == 'O':
        chunkEnd = True

    if prevTag != 'O' and prevTag != '.' and prevTagType != tagType:
        chunkEnd = True
    return chunkEnd

def __splitTagType(tag):
    s = tag.split('-')
    if len(s) > 2 or len(s) == 0:
        raise ValueError('tag format wrong. it must be B-xxx.xxx')
    if len(s) == 1:
        tag = s[0]
        tagType = ""
    else:
        tag = s[0]
        tagType = s[1]
    return tag, tagType

def computeF1Score(correct_slots, pred_slots):

    correctChunk = {}
    correctChunkCnt = 0
    foundCorrect = {}
    foundCorrectCnt = 0
    foundPred = {}
    foundPredCnt = 0
    correctTags = 0
    tokenCount = 0
    for correct_slot, pred_slot in zip(correct_slots, pred_slots):
        inCorrect = False
        lastCorrectTag = 'O'
        lastCorrectType = ''
        lastPredTag = 'O'
        lastPredType = ''
        for c, p in zip(correct_slot, pred_slot):
            correctTag, correctType = __splitTagType(c)
            predTag, predType = __splitTagType(p)

            if inCorrect == True:
                if __endOfChunk(lastCorrectTag, correctTag, lastCorrectType, correctType) == True and \
                   __endOfChunk(lastPredTag, predTag, lastPredType, predType) == True and \
                   (lastCorrectType == lastPredType):
                    inCorrect = False
                    correctChunkCnt += 1
                    if lastCorrectType in correctChunk:
                        correctChunk[lastCorrectType] += 1
                    else:
                        correctChunk[lastCorrectType] = 1
                elif __endOfChunk(lastCorrectTag, correctTag, lastCorrectType, correctType) != \
                     __endOfChunk(lastPredTag, predTag, lastPredType, predType) or \
                     (correctType != predType):
                    inCorrect = False

            if __startOfChunk(lastCorrectTag, correctTag, lastCorrectType, correctType) == True and \
               __startOfChunk(lastPredTag, predTag, lastPredType, predType) == True and \
               (correctType == predType):
                inCorrect = True

            if __startOfChunk(lastCorrectTag, correctTag, lastCorrectType, correctType) == True:
                foundCorrectCnt += 1
                if correctType in foundCorrect:
                    foundCorrect[correctType] += 1
                else:
                    foundCorrect[correctType] = 1

            if __startOfChunk(lastPredTag, predTag, lastPredType, predType) == True:
                foundPredCnt += 1
                if predType in foundPred:
                    foundPred[predType] += 1
                else:
                    foundPred[predType] = 1

            if correctTag == predTag and correctType == predType:
                correctTags += 1

            tokenCount += 1

            lastCorrectTag = correctTag
            lastCorrectType = correctType
            lastPredTag = predTag
            lastPredType = predType

        if inCorrect == True:
            correctChunkCnt += 1
            if lastCorrectType in correctChunk:
                correctChunk[lastCorrectType] += 1
            else:
                correctChunk[lastCorrectType] = 1

    if foundPredCnt > 0:
        precision = 100*correctChunkCnt/foundPredCnt
    else:
        precision = 0

    if foundCorrectCnt > 0:
        recall = 100*correctChunkCnt/foundCorrectCnt
    else:
        recall = 0

    if (precision+recall) > 0:
        f1 = (2*precision*recall)/(precision+recall)
    else:
        f1 = 0

    return f1, precision, recall

def snips_f1_score(yTrue, yPred):
    
    """
    f1 score for SNIPS NER/Slot filling task taken from the `MiuLab <https://github.com/MiuLab/SlotGated-SLU/blob/master/utils.py>`_ library.
    
    To use this metric, add **snips_f1_score** into list of ``metrics`` in task file.

    Args:
        yTrue (:obj:`list of list`) : [['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        yPred (:obj:`list of list`) : [['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        
    """
    
    snipsF1, _, _ = computeF1Score(yTrue, yPred)
    return snipsF1

def snips_precision(yTrue, yPred):
    """
    Precision score for SNIPS NER/Slot filling task taken from the `MiuLab <https://github.com/MiuLab/SlotGated-SLU/blob/master/utils.py>`_ library.
    
    To use this metric, add **snips_precision** into list of ``metrics`` in task file.

    Args:
        yTrue (:obj:`list of list`) : [['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        yPred (:obj:`list of list`) : [['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        
    """
    
    _, snipsPrecision, _ = computeF1Score(yTrue, yPred)
    return snipsPrecision
    
def snips_recall(yTrue, yPred):
    
    """
    Recall score for SNIPS NER/Slot filling task taken from the `MiuLab <https://github.com/MiuLab/SlotGated-SLU/blob/master/utils.py>`_ library.
    
    To use this metric, add **snips_recall** into list of ``metrics`` in task file.

    Args:
        yTrue (:obj:`list of list`) : [['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        yPred (:obj:`list of list`) : [['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        
    """
    _, _, snipsRecall = computeF1Score(yTrue, yPred)
    return snipsRecall

