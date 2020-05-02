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

    def __init__(self, modelPath, maxSeqLen):
        self.maxSeqLen = maxSeqLen
        self.modelPath = modelPath
        assert os.path.exists(self.modelPath), "saved model not present at {}".format(self.modelPath)

        loadedDict = torch.load(self.modelPath)
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
    def format_output(self, dataList, allIds, allPreds):
        for sampleId in allIds[0]:
            print("\nInput Sample : ", dataList[sampleId])
            for i in range(len(allIds)):
                taskName = self.taskParams.taskIdNameMap[i]
                result = allPreds[i][sampleId]
                print("{} : {}".format(taskName, result))

    def infer(self, dataList, taskNamesList, batchSize = 8, seed=42):
        '''
        dataList :- list(batch) of list input. 
                    eg. [ [<sentence1>, <sentence2>], [<sentece>, <sentence>],... ]
                        [ [<sentence>], [<sentence>], [<sentence>] ....]

        taskNamesList :- list of tasks to perform on data
                    eg. [<taskName1>, <taskName2> ,..]
        '''
        allTasksList = []
        for taskName in taskNamesList:
            assert taskName in self.taskParams.taskIdNameMap.values(), "task Name not in task names for loaded model"
            taskId = [taskId for taskId, tName in self.taskParams.taskIdNameMap.items() if tName==taskName][0]
            taskType = self.taskParams.taskTypeMap[taskName]

            taskData = self.make_feature_samples(dataList, taskType, taskName)
            tasksDict = {"data_task_id" : int(taskId),
                        "data_" : taskData,
                        "data_task_type" : taskType,
                        "data_task_name" : taskName}
            allTasksList.append(tasksDict)

        allData = allTasksDataset(allTasksList, pipeline=True)
        batchSampler = Batcher(allData, batchSize=batchSize, seed =seed)
        batchSamplerUtils = batchUtils(isTrain = False, modelType= self.taskParams.modelType,
                                  maxSeqLen = self.maxSeqLen)
        inferDataLoader = DataLoader(allData, batch_sampler=batchSampler,
                                    collate_fn=batchSamplerUtils.collate_fn,
                                    pin_memory=torch.cuda.is_available())

        with torch.no_grad():
            allIds, allPreds = evaluate(allData, batchSampler, inferDataLoader, self.taskParams,
                    self.model, gpu=torch.cuda.is_available(), evalBatchSize=batchSize, needMetrics=False, hasTrueLabels=False,
                    returnPred=True)
            self.format_output(dataList, allIds, allPreds)


            




