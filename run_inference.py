"""
File for making inference on a testing file with a saved multi-task model over 
The input data file here has to be the data prepared file for the corresponding test file

For getting inference on a test file, (say test.tsv) 
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
device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_file_path', type=str, required=True,
                        help="path to the tsv file on which predictions to be made")
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
    parser.add_argument('--max_seq_len', type=int, 
                        help = "max seq len used during training of model")
    parser.add_argument('--seed', type=int, default = 42,
                        help = "seed")
    args = parser.parse_args()

    allParams = vars(args)
    assert os.path.exists(args.saved_model_path), "saved model not present at {}".format(args.saved_model_path)
    assert os.path.exists(args.pred_file_path), "prediction tsv file not present at {}".format(args.pred_file_path)
    loadedDict = torch.load(args.saved_model_path, map_location=device)
    taskParamsModel = loadedDict['task_params']
    logger.info('Task Params loaded from saved model.')

    assert args.task_name in taskParamsModel.taskIdNameMap.values(), "task Name not in task names for loaded model"
    
    taskId = [taskId for taskId, taskName in taskParamsModel.taskIdNameMap.items() if taskName==args.task_name][0]
    taskType = taskParamsModel.taskTypeMap[args.task_name]

    # preparing data from tsv file
    rows = load_data(args.pred_file_path, taskType, hasLabels = args.has_labels)

    modelName = taskParamsModel.modelType.name.lower()
    _, _ , tokenizerClass, defaultName = NLP_MODELS[modelName]
    configName = taskParamsModel.modelConfig
    if configName is None:
        configName = defaultName
    
    #making tokenizer for model
    tokenizer = tokenizerClass.from_pretrained(configName)
    logger.info('{} model tokenizer loaded for config {}'.format(modelName, configName))
    
    dataPath = os.path.join(args.out_dir, '{}_prediction_data'.format(configName))
    if not os.path.exists(dataPath):
        os.makedirs(dataPath)
    wrtFile = os.path.join(dataPath, '{}.json'.format(args.pred_file_path.split('/')[-1].split('.')[0]))
    print('Processing Started...')
    create_data_multithreaded(rows, wrtFile, tokenizer, taskParamsModel, args.task_name,
                            args.max_seq_len, multithreaded = True)
    print('Data Processing done for {}. File saved at {}'.format(args.task_name, wrtFile))

    allTaskslist = [ 
        {"data_task_id" : int(taskId),
         "data_path" : wrtFile,
         "data_task_type" : taskType,
         "data_task_name" : args.task_name}
        ]
    allData = allTasksDataset(allTaskslist)
    batchSampler = Batcher(allData, batchSize=args.eval_batch_size, seed = args.seed)
    batchSamplerUtils = batchUtils(isTrain = False, modelType= taskParamsModel.modelType,
                                  maxSeqLen = args.max_seq_len)
    inferDataLoader = DataLoader(allData, batch_sampler=batchSampler,
                                collate_fn=batchSamplerUtils.collate_fn,
                                pin_memory=torch.cuda.is_available())

    allParams['task_params'] = taskParamsModel
    allParams['gpu'] = torch.cuda.is_available()
    # dummy values
    allParams['num_train_steps'] = 10
    allParams['warmup_steps'] = 0
    allParams['learning_rate'] = 2e-5
    allParams['epsilon'] = 1e-8

    #making and loading model
    model = multiTaskModel(allParams)
    model.load_multi_task_model(loadedDict)

    with torch.no_grad():
        wrtPredFile = 'predictions.tsv'
        evaluate(allData, batchSampler, inferDataLoader, taskParamsModel,
                model, gpu=allParams['gpu'], evalBatchSize=args.eval_batch_size, needMetrics=False, hasTrueLabels=False,
                wrtDir=args.out_dir, wrtPredPath=wrtPredFile)

if __name__ == "__main__":
    main()
