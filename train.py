'''
Final Training script to run traininig for multi-task
'''
import argparse
import random
import numpy as np
import logging
import torch
import os

from task_utils import TasksParam
from data_manager import allTasksDataset, Batcher, batchUtils
from torch.utils.data import Dataset, DataLoader, BatchSampler
from logger_ import make_logger

def make_arguments(parser):
    parser.add_argument('--data_dir', type = str, required=True,
                        help='path to directory where prepared data is present')
    parser.add_argument('--task_file', type = str, required = True,
                        help = 'path to the yml task file')
    parser.add_argument('--out_dir', type = str, required=True,
                        help = 'path to save the model')
    parser.add_argument('--train_batch_size', type = int, default=8,
                        help='batch size to use for training')
    parser.add_argument('--eval_batch_size', type = int, default = 32,
                        help = "batch size to use during evaluation")
    parser.add_argument('--debug_mode', default = False, type = bool,
                        help = "record logs for debugging if True")
    parser.add_argument('--log_file', default='multi_task_logs.txt', type = str,
                        help = "name of log file to store")
    parser.add_argument('--seed', default=42, type = int,
                        help = "seed to set for modules")
    parser.add_argument('--max_seq_len', default=384, type =int,
                        help = "max seq length used for model at time of data preparation")
    return parser
    
    
parser = argparse.ArgumentParser()
parser = make_arguments(parser)
args = parser.parse_args()

#setting logging
logger = make_logger(name = __name__, debugMode=args.debug_mode,
                    logFile=args.log_file)
                    
#setting seed
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)

assert os.path.isdir(args.data_dir), "data_dir doesn't exists"
assert os.path.exists(arg.task_file), "task_file doesn't exists"
if not os.path.exists(args.out_dir):
    os.makedirs(args.out_dir)


def make_data_handlers(taskParams, mode, isTrain):
    '''
    This function makes the allTaskDataset, Batch Sampler, Collater function
    and DataLoader for train, dev and test files as per mode.
    In order of task file, 
    train file is at 0th index
    dev file is at 1st index
    test file at 2nd index
    '''
    modePosMap = {"train" : 0, "dev" : 1, "test" : 2}
    modeIdx = modePosMap[mode]
    allTaskslist = []
    for taskId, taskName in taskParams.taskIdNameMap.items():
        taskType = taskParams.taskTypeMap[taskName]
        if mode == "test":
            assert len(taskParams.fileNamesMap[taskName])==3, "test file is required along with train, dev" 
        taskDataPath = os.path.join(args.data_dir, taskParams.fileNamesMap[taskName][modeIdx])
        assert os.path.exists(taskDataPath), "{} doesn't exist".format(taskDataPath)
        taskDict = {"data_task_id" : int(taskId),
                    "data_path" : taskDataPath,
                    "data_task_type" : taskType}
        allTaskslist.append(taskDict)

    allData = allTasksDataset(allTaskslist)
    batchSampler = Batcher(allData, batchSize=args.batch_size, seed = args.seed)
    batchSamplerUtils = batchUtils(isTrain = isTrain, modelType= taskParams.modelType,
                                  maxSeqLen = args.max_seq_len)
    multiTaskDataLoader = DataLoader(allData, batch_sampler = batchSampler,
                                collate_fn=batchSamplerUtils.collate_fn,
                                pin_memory=torch.cuda.is_available())

    return allData, batchSampler, multiTaskDataLoader

def main():
    #making task file into object
    taskParams = TasksParam(args.task_file)
    logger.info("task params object created from task file...")

    allParams = vars(args)
    allParams['task_params'] = taskParams

    # making handlers for train 
    allDataTrain, BatchSamplerTrain, multiTaskDataLoaderTrain = make_data_handlers(taskParams,
                                                                                "train", isTrain=True)
    # if evaluation on dev set is required during training. Labels are required
    # It will occur at the end of each epoch
    if args.eval_while_train:
        allDataDev, BatchSamplerDev, multiTaskDataLoaderDev = make_data_handlers(taskParams,
                                                                                "dev", isTrain=False)
    # if evaluation on test set is required during training. Labels are required
    # It will occur at the end of each epoch
    if args.test_while_train:
        allDataTest, BatchSamplerTest, multiTaskDataLoaderTest = make_data_handlers(taskParams,
                                                                                "test", isTrain=False)   
           










