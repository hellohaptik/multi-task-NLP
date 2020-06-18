'''
Final Training script to run traininig for multi-task
'''
import argparse
import random
import numpy as np
import pandas as pd
import logging
import torch
import os
import math
from datetime import datetime
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from utils.task_utils import TasksParam   
from utils.data_utils import METRICS, TaskType
from models.data_manager import allTasksDataset, Batcher, batchUtils
from torch.utils.data import Dataset, DataLoader, BatchSampler
from logger_ import make_logger
from models.model import multiTaskModel
from models.eval import evaluate

def make_arguments(parser):
    parser.add_argument('--data_dir', type = str, required=True,
                        help='path to directory where prepared data is present')
    parser.add_argument('--task_file', type = str, required = True,
                        help = 'path to the yml task file')
    parser.add_argument('--out_dir', type = str, required=True,
                        help = 'path to save the model')
    parser.add_argument('--epochs', type = int, required=True,
                        help = 'number of epochs to train')
    parser.add_argument('--freeze_shared_model', default=False, action='store_true',
                        help = "True to freeze the loaded pre-trained shared model and only finetune task specific headers")
    parser.add_argument('--train_batch_size', type = int, default=8,
                        help='batch size to use for training')
    parser.add_argument('--eval_batch_size', type = int, default = 32,
                        help = "batch size to use during evaluation")
    parser.add_argument('--grad_accumulation_steps', type =int, default = 1,
                        help = "number of steps to accumulate gradients before update")
    parser.add_argument('--num_of_warmup_steps', type=int, default = 0,
                        help = "warm-up value for scheduler")
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                       help = "learning rate for optimizer")
    parser.add_argument('--epsilon', type=float, default=1e-8,
                       help="epsilon value for optimizer")
    parser.add_argument('--grad_clip_value', type = float, default=1.0,
                        help = "gradient clipping value to avoid gradient overflowing" )
    parser.add_argument('--log_file', default='multi_task_logs.log', type = str,
                        help = "name of log file to store")
    parser.add_argument('--log_per_updates', default = 10, type = int,
                        help = "number of steps after which to log loss")
    parser.add_argument('--seed', default=42, type = int,
                        help = "seed to set for modules")
    parser.add_argument('--max_seq_len', default=128, type =int,
                        help = "max seq length used for model at time of data preparation")
    parser.add_argument('--save_per_updates', default = 0, type = int,
                        help = "to keep saving model after this number of updates")
    parser.add_argument('--limit_save', default = 10, type = int,
                        help = "max number recent checkpoints to keep saved")
    parser.add_argument('--load_saved_model', type=str, default=None,
                        help="path to the saved model in case of loading from saved")
    parser.add_argument('--eval_while_train', default = False, action = 'store_true',
                        help = "if evaluation on dev set is required during training.")
    parser.add_argument('--test_while_train', default=False, action = 'store_true',
                        help = "if evaluation on test set is required during training.")
    parser.add_argument('--resume_train', default=False, action = 'store_true',
                        help="Set for resuming training from a saved model")
    parser.add_argument('--finetune', default= False, action = 'store_true',
                        help = "If only the shared model is to be loaded with saved pre-trained multi-task model.\
                            In this case, you can specify your own tasks with task file and use the pre-trained shared model\
                            to finetune upon.")
    parser.add_argument('--debug_mode', default = False, action = 'store_true', 
                        help = "record logs for debugging if True")
    parser.add_argument('--silent', default = False, action = 'store_true', 
                        help = "Only write logs to file if True")
    return parser
    
parser = argparse.ArgumentParser()
parser = make_arguments(parser)
args = parser.parse_args()

#setting logging
now = datetime.now()
logDir = now.strftime("%d_%m-%H_%M")
if not os.path.isdir(logDir):
    os.makedirs(logDir)

logger = make_logger(name = "multi_task", debugMode=args.debug_mode,
                    logFile=os.path.join(logDir, args.log_file), silent=args.silent)
logger.info("logger created.")

device = torch.device('cpu')       
#setting seed
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device('cuda')

assert os.path.isdir(args.data_dir), "data_dir doesn't exists"
assert os.path.exists(args.task_file), "task_file doesn't exists"
if not os.path.exists(args.out_dir):
    os.makedirs(args.out_dir)


def make_data_handlers(taskParams, mode, isTrain, gpu):
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
        #dataFileName =  '{}.json'.format(taskParams.fileNamesMap[taskName][modeIdx].split('.')[0])
        dataFileName = '{}.json'.format(taskParams.fileNamesMap[taskName][modeIdx].lower().replace('.tsv',''))
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

    

def main():
    allParams = vars(args)
    logger.info('ARGS : {}'.format(allParams))
    # loading if load_saved_model
    if args.load_saved_model is not None:
        assert os.path.exists(args.load_saved_model), "saved model not present at {}".format(args.load_saved_model)
        loadedDict = torch.load(args.load_saved_model, map_location=device)
        logger.info('Saved Model loaded from {}'.format(args.load_saved_model))

        if args.finetune is True:
            '''
            NOTE :- 
            In finetune mode, only the weights from the shared encoder (pre-trained) from the model will be used. The headers
            over the model will be made from the task file. You can further finetune for training the entire model.
            Freezing of the pre-trained moddel is also possible with argument 
            '''
            logger.info('In Finetune model. Only shared Encoder weights will be loaded from {}'.format(args.load_saved_model))
            logger.info('Task specific headers will be made according to task file')
            taskParams = TasksParam(args.task_file)

        else:
            '''
            NOTE : -
            taskParams used with this saved model must also be stored. THE SAVED TASK PARAMS 
            SHALL BE USED HERE TO AVOID ANY DISCREPENCIES/CHANGES IN THE TASK FILE.
            Hence, if changes are made to task file after saving this model, they shall be ignored
            '''
            taskParams = loadedDict['task_params']
            logger.info('Task Params loaded from saved model.')
            logger.info('Any changes made to task file except the data \
                        file paths after saving this model shall be ignored')
            tempTaskParams = TasksParam(args.task_file)
            #transfering the names of file in new task file to loaded task params
            for taskId, taskName in taskParams.taskIdNameMap.items():
                assert taskName in tempTaskParams.taskIdNameMap.values(), "task names changed in task file given.\
                tasks supported for loaded model are {}".format(list(taskParams.taskIdNameMap.values()))

                taskParams.fileNamesMap[taskName] = tempTaskParams.fileNamesMap[taskName]
    else:
        taskParams = TasksParam(args.task_file)
        logger.info("Task params object created from task file...")
        

    allParams['task_params'] = taskParams
    allParams['gpu'] = torch.cuda.is_available()
    logger.info('task parameters:\n {}'.format(taskParams.taskDetails))

    tensorboard = SummaryWriter(log_dir = os.path.join(logDir, 'tb_logs'))
    logger.info("Tensorboard writing at {}".format(os.path.join(logDir, 'tb_logs')))

    # making handlers for train
    logger.info("Creating data handlers for training...")
    allDataTrain, BatchSamplerTrain, multiTaskDataLoaderTrain = make_data_handlers(taskParams,
                                                                                "train", isTrain=True,
                                                                                gpu = allParams['gpu'])
    # if evaluation on dev set is required during training. Labels are required
    # It will occur at the end of each epoch
    if args.eval_while_train:
        logger.info("Creating data handlers for dev...")
        allDataDev, BatchSamplerDev, multiTaskDataLoaderDev = make_data_handlers(taskParams,
                                                                                "dev", isTrain=False,
                                                                                gpu=allParams['gpu'])
    # if evaluation on test set is required during training. Labels are required
    # It will occur at the end of each epoch
    if args.test_while_train:
        logger.info("Creating data handlers for test...")
        allDataTest, BatchSamplerTest, multiTaskDataLoaderTest = make_data_handlers(taskParams,
                                                                                "test", isTrain=False,
                                                                                gpu=allParams['gpu'])
    #making multi-task model
    allParams['num_train_steps'] = math.ceil(len(multiTaskDataLoaderTrain)/args.train_batch_size) *args.epochs // args.grad_accumulation_steps
    allParams['warmup_steps'] = args.num_of_warmup_steps
    allParams['learning_rate'] = args.learning_rate
    allParams['epsilon'] = args.epsilon
    
    logger.info("NUM TRAIN STEPS: {}".format(allParams['num_train_steps']))
    logger.info("len of dataloader: {}".format(len(multiTaskDataLoaderTrain)))
    logger.info("Making multi-task model...")
    model = multiTaskModel(allParams)
    #logger.info('################ Network ###################')
    #logger.info('\n{}\n'.format(model.network))

    if args.load_saved_model:
        if args.finetune is True:
            model.load_shared_model(loadedDict, args.freeze_shared_model)
            logger.info('shared model loaded for finetune from {}'.format(args.load_saved_model))
        else:
            model.load_multi_task_model(loadedDict)
            logger.info('saved model loaded with global step {} from {}'.format(model.globalStep,
                                                                            args.load_saved_model))
        if args.resume_train:
            logger.info("Resuming training from global step {}. Steps before it will be skipped".format(model.globalStep))

    # training 
    resCnt = 0
    for epoch in range(args.epochs):
        logger.info('\n####################### EPOCH {} ###################\n'.format(epoch))
        totalEpochLoss = 0
        text = "Epoch: {}".format(epoch)
        tt = int(allParams['num_train_steps']*args.grad_accumulation_steps/args.epochs)
        with tqdm(total = tt, position=epoch, desc=text) as progress:
            for i, (batchMetaData, batchData) in enumerate(multiTaskDataLoaderTrain):
                batchMetaData, batchData = BatchSamplerTrain.patch_data(batchMetaData,batchData, gpu = allParams['gpu'])
                if args.resume_train and args.load_saved_model and resCnt*args.grad_accumulation_steps < model.globalStep:
                    '''
                    NOTE: - Resume function is only to be used in case the training process couldnt
                    complete or you wish to extend the training to some more epochs.
                    Please keep the gradient accumulation step the same for exact resuming.
                    '''
                    resCnt += 1
                    progress.update(1)
                    continue
                model.update_step(batchMetaData, batchData)
                totalEpochLoss += model.taskLoss.item()

                if model.globalStep % args.log_per_updates == 0 and (model.accumulatedStep+1 == args.grad_accumulation_steps):
                    taskId = batchMetaData['task_id']
                    taskName = taskParams.taskIdNameMap[taskId]
                    #avgLoss = totalEpochLoss / ((i+1)*args.train_batch_size)
                    avgLoss = totalEpochLoss / (i+1)
                    logger.info('Steps: {} Task: {} Avg.Loss: {} Task Loss: {}'.format(model.globalStep,
                                                                                    taskName,
                                                                                    avgLoss,
                                                                                    model.taskLoss.item()))
                    
                    tensorboard.add_scalar('train/avg_loss', avgLoss, global_step= model.globalStep)
                    tensorboard.add_scalar('train/{}_loss'.format(taskName),
                                            model.taskLoss.item(),
                                            global_step=model.globalStep)
                
                if args.save_per_updates > 0 and  ( (model.globalStep+1) % args.save_per_updates)==0 and (model.accumulatedStep+1==args.grad_accumulation_steps):
                    savePath = os.path.join(args.out_dir, 'multi_task_model_{}_{}.pt'.format(epoch,
                                                                                            model.globalStep))
                    model.save_multi_task_model(savePath)

                    # limiting the checkpoints save, remove checkpoints if beyond limit
                    if args.limit_save > 0:
                        stepCkpMap = {int(ckp.rstrip('.pt').split('_')[-1]) : ckp for ckp in os.listdir(args.out_dir) if ckp.endswith('.pt') }
                        
                        #sorting based on global step
                        stepToDel = sorted(list(stepCkpMap.keys()))[:-args.limit_save]

                        for ckpStep in stepToDel:
                            os.remove(os.path.join(args.out_dir, stepCkpMap[ckpStep]))
                            logger.info('Removing checkpoint {}'.format(stepCkpMap[ckpStep]))

                progress.update(1)

            #saving model after epoch
            if args.resume_train and args.load_saved_model and resCnt*args.grad_accumulation_steps < model.globalStep:
                pass
            else:
                savePath = os.path.join(args.out_dir, 'multi_task_model_{}_{}.pt'.format(epoch, model.globalStep))  
                model.save_multi_task_model(savePath)

            if args.eval_while_train:
                logger.info("\nRunning Evaluation on dev...")
                with torch.no_grad():
                    evaluate(allDataDev, BatchSamplerDev, multiTaskDataLoaderDev, taskParams,
                            model, gpu=allParams['gpu'],evalBatchSize=args.eval_batch_size, hasTrueLabels=True, needMetrics=True)

            if args.test_while_train:
                logger.info("\nRunning Evaluation on test...")
                wrtPredpath = "test_predictions_{}.tsv".format(epoch)
                with torch.no_grad():
                    evaluate(allDataTest, BatchSamplerTest, multiTaskDataLoaderTest, taskParams,
                            model, gpu=allParams['gpu'], evalBatchSize = args.eval_batch_size, needMetrics=True, hasTrueLabels=True,
                            wrtDir=args.out_dir, wrtPredPath=wrtPredpath)

if __name__ == "__main__":
    main()