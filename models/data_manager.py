'''
Script to manage datasets for multiple tasks
'''
from torch.utils.data import Dataset, DataLoader, BatchSampler
from utils.data_utils import TaskType, ModelType
import torch
import random
import logging
import json
logger = logging.getLogger("multi_task")

class allTasksDataset(Dataset):
    '''
    class to make pytorch dataset of the processed data for a specific task
    taskDict :- list of dictionaries. Each dictioanry belong to the details of a 
                dataset to be created for a task
                [ {"data_task_id" : "", "data_path" : "", "data_task_type" : ""},
                 ...]
    '''
    def __init__(self, taskDict, pipeline = False):
        self.taskDict = taskDict
        self.pipeline = pipeline
        self.allTasksData, self.taskIdTypeMap = self.make_all_datasets()

    def read_data(self, readPath):
        with open(readPath, 'r', encoding = 'utf-8') as file:
            logger.info('Reading data from file {}'.format(readPath))
            taskData = []
            for i, line in enumerate(file):
                #if i >=1000:
                    #continue
                sample = json.loads(line)
                taskData.append(sample)
        return taskData

    def make_all_datasets(self):
        '''
        For each dataset entry in the taskDict, this function makes them into corresponding dataset 
        and returns a dictionary mapping like {<task_id> : <dataset>,}
        '''
        allTasksData = {}
        taskIdTypeMap = {} # mapping from task id to task type
        for task in self.taskDict:
            if self.pipeline:
                logger.info('Reading data for pipeline')
                data = task["data_"]
            else:
                data = self.read_data(task["data_path"])
            allTasksData[task["data_task_id"]] = data
            taskIdTypeMap[task["data_task_id"]] = task["data_task_type"]
            logger.info('Read Data for Task Id: {} Task Name: {}. Samples {}'.format(task["data_task_id"], task["data_task_name"], len(data)))
        return allTasksData, taskIdTypeMap

    # some standard functions which need to be overridden from Dataset
    #class for item, len etc..
    def __len__(self):
        return sum(len(v) for k, v in self.allTasksData.items())

    # get item will be used to fetch a sample when required for the corresponding task id. 
    def __getitem__(self, idx):
        taskId, sampleId = idx
        out = {"task": {"task_id": taskId, "task_type": self.taskIdTypeMap[taskId]},
                "sample": self.allTasksData[taskId][sampleId]}
        return out

class Batcher(BatchSampler):
    def __init__(self, dataObj, batchSize, shuffleTask = True, shuffleBatch = True, seed = 42):
        '''
        dataObj :- An instance of allTasksDataset containing data for all tasks
        '''
        self.dataObj = dataObj
        self.allTasksData = dataObj.allTasksData
        self.batchSize = batchSize
        # to shuffle the indices in a batch
        self.shuffleBatch = shuffleBatch
        # to shuffle the samples picked up among all the tasks
        self.shuffleTask = shuffleTask
        self.seed = seed
        
        self.allTasksDataBatchIdxs = []
        self.taskIdxId = []
        for taskId, data in self.allTasksData.items():
            self.allTasksDataBatchIdxs.append(self.make_batches(len(data)))
            self.taskIdxId.append(taskId)

    def make_batches(self, dataSize):
        batchIdxs = [list(range(i, min(i+self.batchSize, dataSize))) for i in range(0, dataSize, self.batchSize)]
        if self.shuffleBatch:
            random.seed(self.seed)
            random.shuffle(batchIdxs)
        return batchIdxs

    def make_task_idxs(self):
        '''
        This fn makes task indices for which a corresponding batch is created
        eg. [0, 0, 1, 3, 0, 2, 3, 1, 1, ..] if task ids are 0,1,2,3
        '''
        taskIdxs = []
        for i in range(len(self.allTasksDataBatchIdxs)):
            taskIdxs += [i]*len(self.allTasksDataBatchIdxs[i])
        if self.shuffleTask:
            random.seed(self.seed)
            random.shuffle(taskIdxs)
        return taskIdxs

    #over riding BatchSampler functions to generate iterators for all tasks
    # and iterate
    def __len__(self):
        return sum(len(data) for taskId, data in self.allTasksData.items())

    def __iter__(self):
        allTasksIters = [iter(item) for item in self.allTasksDataBatchIdxs]
        #all_iters = [iter(item) for item in self._train_data_list]
        allIdxs = self.make_task_idxs()
        for taskIdx in allIdxs:
            # this batch belongs to a specific task id
            batchTaskId = self.taskIdxId[taskIdx]
            batch = next(allTasksIters[taskIdx])
            yield [(batchTaskId, sampleIdx) for sampleIdx in batch]
            
    def patch_data(self, batch_info, batch_data, gpu = None):
        if gpu:
            for i, part in enumerate(batch_data):
                if part is not None:
                    if isinstance(part, torch.Tensor):
                        batch_data[i] = part.pin_memory().cuda(non_blocking=True)
                    elif isinstance(part, tuple):
                        batch_data[i] = tuple(sub_part.pin_memory().cuda(non_blocking=True) for sub_part in part)
                    elif isinstance(part, list):
                        batch_data[i] = [sub_part.pin_memory().cuda(non_blocking=True) for sub_part in part]
                    else:
                        raise TypeError("unknown batch data type at %s: %s" % (i, part))

        return batch_info, batch_data

class batchUtils:
    '''
    This class is supposed to perform function which will help complete the batch data
    when DataLoader creates batch using allTasksDataset and Batcher.
    Main function would be
    1. A function to make get the various components of input in batch samples and make them into 
    Pytorch Tensors like token_id, type_ids, masks.

    2. Collater function :- This function will use the above function to convert the batch into 
    pytorch tensor inputs. As converting all the data into pytorch tensors before might not be a good 
    idea due to space, hence this custom function will be used to convert the batches into tensors on the fly
    by acting as custom collater function to DataLoader
    '''

    def __init__(self, isTrain, modelType, maxSeqLen, dropout = 0.005):
        self.isTrain = isTrain
        self.modelType = modelType
        self.maxSeqLen = maxSeqLen
        #self.dropout = dropout

    def check_samples_len(self, batch):
        #function to check whether all samples are having the maxSeqLen mentioned
        for samp in batch:
            assert len(samp['token_id']) == self.maxSeqLen, "token_id len doesn't match max seq len"
            # for multiple encoders
            if samp['type_id'] is not None:
                assert len(samp['type_id']) == self.maxSeqLen, "type_id len doesn't match max seq len"
            if samp['mask'] is not None:
                assert len(samp['mask']) == self.maxSeqLen, "mask len doesn't match max seq len"

    def make_batch_to_input_tensor(self, batch):
        #check len in batch data
        self.check_samples_len(batch)
        batchSize = len(batch)

        hasTypeIds = True
        hasAttnMasks = True
        if batch[0]['type_id'] is None:
            hasTypeIds = False
        if batch[0]['mask'] is None:
            hasAttnMasks = False

        #initializing token id, type id, attention mask tensors for this batch
        tokenIdsBatchTensor = torch.LongTensor(batchSize, self.maxSeqLen).fill_(0)
        typeIdsBatchTensor = torch.LongTensor(batchSize, self.maxSeqLen).fill_(0)
        masksBatchTensor = torch.LongTensor(batchSize, self.maxSeqLen).fill_(0)

        #fillling in data from sample
        for i, sample in enumerate(batch):
            tokenIdsBatchTensor[i] = torch.LongTensor(sample['token_id'])
            if hasTypeIds:
                typeIdsBatchTensor[i] = torch.LongTensor(sample['type_id'])
            if hasAttnMasks:
                masksBatchTensor[i] = torch.LongTensor(sample['mask'])

        # meta deta will store more things like task id, task type etc. 
        batchMetaData = {"token_id_pos" : 0, "type_id_pos" : 1, "mask_pos" : 2}
        batchData = [tokenIdsBatchTensor, None, None]  #None, None in case type ids, attnMasks not required by model
        if hasTypeIds:
            batchData[1] = typeIdsBatchTensor
        if hasAttnMasks:
            batchData[2] = masksBatchTensor
        return batchMetaData, batchData

    def collate_fn(self, batch):
        '''
        This function will be used by DataLoader to return batches
        '''
        taskId = batch[0]["task"]["task_id"]
        taskType = batch[0]["task"]["task_type"]

        orgBatch = []
        labels = []
        for sample in batch:
            assert sample["task"]["task_id"] == taskId
            assert sample["task"]["task_type"] == taskType
            orgBatch.append(sample["sample"])
            labels.append(sample["sample"]["label"])
            
        batch = orgBatch
        #making tensor batch data
        batchMetaData, batchData = self.make_batch_to_input_tensor(batch)
        batchMetaData['task_id'] = taskId
        batchMetaData['task_type'] = taskType

        #adding label tensor when training (as they'll used for loss calculatoion and update)
        # and in evaluation, it won't go with batch data, rather will keep it with meta data for metrics
        if self.isTrain:

            if taskType in (TaskType.SingleSenClassification, TaskType.SentencePairClassification, TaskType.NER):
                batchData.append(torch.LongTensor(labels))

            #position for label
            batchMetaData['label_pos'] = len(batchData) - 1
        else:
            # for test/eval labels won't be added into batch, but kept in meta data
            # so metric evaluation can be done
            #batchData :- [tokenIdsBatchTensor, typeIdsBatchTensor, MasksBatchTensor]
            batchMetaData['label'] = labels

        batchMetaData['uids'] = [sample['uid'] for sample in batch]  # used in scoring
        return batchMetaData, batchData
