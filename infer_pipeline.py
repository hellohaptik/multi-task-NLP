"""
Pipeline for inference on batch for multi-task
"""
from utils.task_utils import TasksParam
from utils.data_utils import TaskType, ModelType, NLP_MODELS
from models.eval import evaluate
from models.model import multiTaskModel
from data_preparation import * 
from models.data_manager import allTasksDataset, Batcher, batchUtils
from torch.utils.data import Dataset, DataLoader, BatchSampler
import argparse
import os
import torch
import logging
logger = logging.getLogger("multi_task")

class inferPipeline:

    """
    For running inference on samples using a trained model for say TaskA, TaskB and TaskC,
    you can import this class and load the corresponding multi-task model by making an 
    object of this class with the following arguments

    Args:
        modelPath (:obj:`str`) : Path to the trained multi-task model for required tasks.
        maxSeqLen (:obj:`int`, defaults to :obj:`128`) : maximum sequence length to be considered for samples.
         Truncating and padding will happen accordingly.
        
    Example::

        >>> from infer_pipeline import inferPipeline
        >>> pipe = inferPipeline(modelPath = 'sample_out_dir/multi_task_model.pt', maxSeqLen = 50)
    
    """

    def __init__(self, modelPath, maxSeqLen = 128):

        device = torch.device('cpu')
        if torch.cuda.is_available():
            device = torch.device('cuda')

        self.maxSeqLen = maxSeqLen
        self.modelPath = modelPath
        assert os.path.exists(self.modelPath), "saved model not present at {}".format(self.modelPath)

        loadedDict = torch.load(self.modelPath, map_location=device)
        self.taskParams = loadedDict['task_params']
        logger.info('Task Params loaded from saved model.')

        modelName = self.taskParams.modelType.name.lower()
        _, _ , tokenizerClass, defaultName = NLP_MODELS[modelName]
        configName = self.taskParams.modelConfig
        if configName is None:
            configName = defaultName
        #making tokenizer for model
        self.tokenizer = tokenizerClass.from_pretrained(configName)
        logger.info('{} model tokenizer loaded for config {}'.format(modelName, configName))
    
        allParams = {}
        allParams['task_params'] = self.taskParams
        allParams['gpu'] = torch.cuda.is_available()
        # dummy values
        allParams['num_train_steps'] = 10
        allParams['warmup_steps'] = 0
        allParams['learning_rate'] = 2e-5
        allParams['epsilon'] = 1e-8

        #making and loading model
        self.model = multiTaskModel(allParams)
        self.model.load_multi_task_model(loadedDict)

    def make_feature_samples(self, dataList, taskType, taskName):
        allData = []
        for i, sample in enumerate(dataList):
            if taskType == TaskType.SingleSenClassification:
                inputIds, typeIds, inputMask = standard_data_converter(self.maxSeqLen, self.tokenizer, sample[0])
                features = {
                    'uid': i,
                    'label': 0,
                    'token_id': inputIds,
                    'type_id': typeIds,
                    'mask': inputMask}

            elif taskType == TaskType.SentencePairClassification:
                inputIds, typeIds, inputMask = standard_data_converter(self.maxSeqLen, self.tokenizer, sample[0], sample[1])
                features = {
                    'uid': i,
                    'label': 0,
                    'token_id': inputIds,
                    'type_id': typeIds,
                    'mask': inputMask}

            elif taskType == TaskType.NER:

                splitSample = sample[0].split()
                label = ["O"]*len(splitSample)
                tempTokens = ['[CLS]']
                tempLabels = ['[CLS]']
                for word, label in zip(splitSample, label):
                    tokens = self.tokenizer.tokenize(word)
                    for m, token in enumerate(tokens):
                        tempTokens.append(token)
                        #only first piece would be marked with label
                        if m==0:
                            tempLabels.append(label)
                        else:
                            tempLabels.append('X')
                # adding [SEP] at end
                tempTokens.append('[SEP]')
                tempLabels.append('[SEP]')

                out = self.tokenizer.encode_plus(text = tempTokens, add_special_tokens=False,
                                        truncation_strategy ='only_first',
                                        max_length = self.maxSeqLen, pad_to_max_length=True)
                typeIds = None
                inputMask = None
                tokenIds = out['input_ids']
                if 'token_type_ids' in out.keys():
                    typeIds = out['token_type_ids']
                if 'attention_mask' in out.keys():
                    inputMask = out['attention_mask']

                labelMap = self.taskParams.labelMap[taskName]
                tempLabelsEnc = pad_sequences([ [labelMap[l] for l in tempLabels] ], 
                                    maxlen=self.maxSeqLen, value=labelMap["O"], padding="post",
                                    dtype="long", truncating="post").tolist()[0]
                #print(tempLabelsEnc)
                assert len(tempLabelsEnc) == len(tokenIds), "mismatch between processed tokens and labels"
                features = {
                'uid': i,
                'label': tempLabelsEnc,
                'token_id': tokenIds,
                'type_id': typeIds,
                'mask': inputMask}
            else:
                raise ValueError(taskType)

            allData.append(features)

        return allData
    def format_ner_output(self, sample, result):
        assert len(sample) == len(result), "length of sample and result list not same"
        returnList = []
        for i, (sam, res) in enumerate(zip(sample, result)):
            if res not in ["O", "[CLS]", "[SEP]", "X"]:
                curr = res.split('-')[-1]
                if len(returnList)>0:
                    if curr == returnList[len(returnList)-1][0]:
                        returnList[len(returnList)-1].append(sam)
                    else:
                        returnList.append([curr, sam])
                else:
                    returnList.append([curr, sam])
                    #print(returnList)
        outList = []
        for finalSam in returnList:
            #print(finalSam)
            outS = ' '.join(finalSam[1:])
            #print(outS)
            outList.append((finalSam[0], outS))
            #print('{} : {}'.format(finalSam[0], outS))

        return outList

    def format_output(self, dataList, allIds, allPreds, allScores):
        returnList = []
        for sampleId in range(len(dataList)):
            resDict = {}
            #print("\nInput Sample : ", dataList[sampleId])
            resDict['Query'] = dataList[sampleId]
            for i in range(len(allIds)):
                taskName = self.taskParams.taskIdNameMap[i]
                taskType = self.taskParams.taskTypeMap[taskName]
                if allPreds[i] == []:
                    continue

                if taskType == TaskType.NER:
                    result = allPreds[i][sampleId]
                    inpp = dataList[sampleId][0].split()
                    #print("{} : ".format(taskName))
                    result = self.format_ner_output(inpp, result)
                else:
                    result = [allPreds[i][sampleId], allScores[i][sampleId]]

                resDict[taskName] = result
                #else:
                    #print("{} : {}".format(taskName, result))
            returnList.append(resDict)
        #print(returnList)
        return returnList
                

    def infer(self, dataList, taskNamesList, batchSize = 8, seed=42):

        """
        This is the function which can be called to get the predictions for input samples
        for the mentioned tasks.

        - Samples can be packed in a ``list of lists`` manner as the function processes inputs in batch.
        - In case, an input sample requires sentence pair, the two sentences can be kept as elements of the list.
        - In case of single sentence classification or NER tasks, only the first element of a sample will be used.
        - For NER, the infer function automatically splits the sentence into tokens.
        - All the tasks mentioned in ``taskNamesList`` are performed for all the input samples.

        Args:

            dataList (:obj:`list of lists`) : A batch of input samples. For eg.
                
                [
                    [<sentenceA>, <sentenceB>],
                    
                    [<sentenceA>, <sentenceB>],

                ]

                or in case all the tasks just require single sentence inputs,
                
                [
                    [<sentenceA>],

                    [<sentenceA>],

                ]

            taskNamesList (:obj:`list`) : List of tasks to be performed on dataList samples. For eg.

                ['TaskA', 'TaskB', 'TaskC']

                You can choose the tasks you want to infer. For eg.

                ['TaskB']

            batchSize (:obj:`int`, defaults to :obj:`8`) : Batch size for running inference.


        Return:

            outList (:obj:`list of objects`) :
                List of dictionary objects where each object contains one corresponding input sample and it's tasks outputs. The task outputs
                can also contain the confidence scores. For eg.

                [
                    {'Query' : [<sentence>],

                    'TaskA' : <TaskA output>,

                    'TaskB' : <TaskB output>,

                    'TaskC' : <TaskC output>},

                ]

        Example::

            >>> samples = [ ['sample_sentence_1'], ['sample_sentence_2'] ]
            >>> tasks = ['TaskA', 'TaskB']
            >>> pipe.infer(samples, tasks)

        """
        #print(dataList)
        #print(taskNamesList)
        allTasksList = []
        for taskName in taskNamesList:
            assert taskName in self.taskParams.taskIdNameMap.values(), "task Name not in task names for loaded model"
            taskId = [taskId for taskId, tName in self.taskParams.taskIdNameMap.items() if tName==taskName][0]
            taskType = self.taskParams.taskTypeMap[taskName]

            taskData = self.make_feature_samples(dataList, taskType, taskName)
            #print('task data :', taskData)

            tasksDict = {"data_task_id" : int(taskId),
                        "data_" : taskData,
                        "data_task_type" : taskType,
                        "data_task_name" : taskName}
            allTasksList.append(tasksDict)

        allData = allTasksDataset(allTasksList, pipeline=True)
        batchSampler = Batcher(allData, batchSize=batchSize, seed =seed,
                             shuffleBatch=False, shuffleTask=False)
        # VERY IMPORTANT TO TURN OFF BATCH SHUFFLE IN INFERENCE. ELSE PREDICTION SCORES
        # WILL GET JUMBLED

        batchSamplerUtils = batchUtils(isTrain = False, modelType= self.taskParams.modelType,
                                  maxSeqLen = self.maxSeqLen)
        inferDataLoader = DataLoader(allData, batch_sampler=batchSampler,
                                    collate_fn=batchSamplerUtils.collate_fn,
                                    pin_memory=torch.cuda.is_available())

        with torch.no_grad():
            allIds, allPreds, allScores = evaluate(allData, batchSampler, inferDataLoader, self.taskParams,
                    self.model, gpu=torch.cuda.is_available(), evalBatchSize=batchSize, needMetrics=False, hasTrueLabels=False,
                    returnPred=True)

            finalOutList = self.format_output(dataList, allIds, allPreds, allScores)
            #print(finalOutList)
            return finalOutList
