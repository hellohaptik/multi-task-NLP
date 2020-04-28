"""
File for making inference on a testing file with a saved multi-task model over 
The input data file here has to be the data prepared file for the corresponding test file

For getting inference on a test file, (say test.tsv),

1. Prepare the data for the file using data_preparation.py. You can use the same
tasks_file used for training the corresponding model by changing the filenames with 
the name of the test file along with the 'has_labels' argument to True/ False depending
on the file has labels or not.

2. 
"""
from utils.task_utils import TasksParam
from utils.data_utils import TaskType
from models.data_manager import allTasksDataset, Batcher, batchUtils
from torch.utils.data import Dataset, DataLoader, BatchSampler
import argparse
import os
import torch
import logging
logger = logging.getLogger("multi_task")
 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_file_path', type=str, required=True,
                        help="path to the prepared file on which predictions to be made")
    parser.add_argument('--task_file', type = str, required = True,
                        help = 'path to the yml task file')
    parser.add_argument('--out_dir', type = str, required=True,
                        help="path to save the predictions")
    parser.add_argument('--has_labels', type=str, default=False,
                        help = "If labels are not present in file then False")
    parser.add_argument('--task_name', type=str, required = True,
                        help = "task name for which prediction is required.")
    parser.add_argument('--saved_model_path', type=str, required = True,
                        help = "path to the trained model to load")
    parser.add_argument('--eval_batch_size', type=int, default = 32,
                        help = "batch size for prediction")
    parser.add_argument('--seed', type=int, default = 42,
                        help = "seed")
    args = parser.parse_args()

    assert os.path.exists(args.saved_model_path), "saved model not present at {}".format(args.saved_model_path)
    assert os.path.exists(args.pred_file_path), "prediction file not present at {}".format(args.pred_file_path)

    loadedDict = torch.load(args.saved_model_path)
    taskParamsModel = loadedDict['task_params']
    logger.info('Task Params loaded from saved model.')
    
    taskParamsFile = TasksParam(args.task_file)
    logger.info("Task params object created from task file...")    
    
    assert args.task_name in taskParamsModel.taskIdNameMap.values(), "task Name not in task names for loaded model"
    
    taskId = [for taskId, taskName in taskParamsModel.taskIdNameMap.items() if taskName==args.task_name][0]
    taskType = taskParamsModel.taskTypeMap[taskName]
    allTaskslist = [ 
        {"data_task_id" : int(taskId),
         "data_path" : args.pred_file_path,
         "data_task_type" : taskType,
         "data_task_name" : taskName}
        ]
    allData = allTasksDataset(allTaskslist)
    batchSampler = Batcher(allData, batchSize=args.eval_batch_size, seed = args.seed)
    batchSamplerUtils = batchUtils(isTrain = isTrain, modelType= taskParams.modelType,
                                  maxSeqLen = args.max_seq_len)
    for taskId, taskName in taskParamsModel.taskIdNameMap.items():
        if taskName == args.task_name:
            

            dataFileName =  '{}.json'.format(taskParams.fileNamesMap[taskName][modeIdx].split('.')[0])
            taskDataPath = os.path.join(args.data_dir, dataFileName)
            assert os.path.exists(taskDataPath), "{} doesn't exist".format(taskDataPath)
            taskDict = {"data_task_id" : int(taskId),
                        "data_path" : taskDataPath,
                        "data_task_type" : taskType,
                        "data_task_name" : taskName}
            allTaskslist.append(taskDict)

    allData = allTasksDataset(allTaskslist)
    if mode == "train":
        batchSize = args.train_batch_size
    else:
        batchSize = args.eval_batch_size

    batchSampler = Batcher(allData, batchSize=batchSize, seed = args.seed)
    batchSamplerUtils = batchUtils(isTrain = isTrain, modelType= taskParams.modelType,
                                  maxSeqLen = args.max_seq_len)
    multiTaskDataLoader = DataLoader(allData, batch_sampler = batchSampler,
                                collate_fn=batchSamplerUtils.collate_fn,
                                pin_memory=gpu)

    return allData, batchSampler, multiTaskDataLoader