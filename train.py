'''
Final Training script to run traininig for multi-task
'''
import argparse
import random
import numpy as np
import logging
import torch
import os
from datetime import datetime
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from task_utils import TasksParam
from data_manager import allTasksDataset, Batcher, batchUtils
from torch.utils.data import Dataset, DataLoader, BatchSampler
from logger_ import make_logger
from models.model import multiTaskModel

def make_arguments(parser):
    parser.add_argument('--data_dir', type = str, required=True,
                        help='path to directory where prepared data is present')
    parser.add_argument('--task_file', type = str, required = True,
                        help = 'path to the yml task file')
    parser.add_argument('--out_dir', type = str, required=True,
                        help = 'path to save the model')
    parser.add_argument('--epochs', type = int, required=True,
                        help = 'number of epochs to train')
    parser.add_argument('--train_batch_size', type = int, default=8,
                        help='batch size to use for training')
    parser.add_argument('--eval_batch_size', type = int, default = 32,
                        help = "batch size to use during evaluation")
    parser.add_argument('--eval_while_train', type = bool, default= False,
                        help = "if evaluation on dev set is required during training.")
    parser.add_argument('--test_while_train', type = bool, default = False,
                        help = "if evaluation on test set is required during training.")
    parser.add_argument('--grad_accumulation_steps', type =int, default = 1,
                        help = "number of steps to accumulate gradients before update")
    parser.add_argument('--num_of_warmup_steps', type=int, default = 0,
                        help = "warm-up value for scheduler")
    parser.add_argument('--grad_clip_value', type = float, default=1.0,
                        help = "gradient clipping value to avoid gradient overflowing" )
    parser.add_argument('--debug_mode', default = False, type = bool,
                        help = "record logs for debugging if True")
    parser.add_argument('--log_file', default='multi_task_logs.log', type = str,
                        help = "name of log file to store")
    parser.add_argument('--log_per_updates', default = 10, type = int,
                        help = "number of steps after which to log loss")
    parser.add_argument('--seed', default=42, type = int,
                        help = "seed to set for modules")
    parser.add_argument('--max_seq_len', default=384, type =int,
                        help = "max seq length used for model at time of data preparation")
    parser.add_argument('--tensorboard', default=True, type = bool,
                        help = "To create tensorboard logs")
    parser.add_argument('--save_per_updates', default = 0, type = int,
                        help = "to keep saving model after this number of updates")
    parser.add_argument('--load_saved_model', type=str, default=None,
                        help="path to the saved model in case of loading from saved")
    parser.add_argument('--resume_train', type=bool, default=False, 
                        help="True for resuming training from a saved model")
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
                    logFile=os.path.join(logDir, args.log_file))
logger.info("logger created.")
                    
#setting seed
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)

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
        dataFileName =  '{}.json'.format(taskParams.fileNamesMap[taskName][modeIdx].split('.')[0])
        taskDataPath = os.path.join(args.data_dir, dataFileName)
        assert os.path.exists(taskDataPath), "{} doesn't exist".format(taskDataPath)
        taskDict = {"data_task_id" : int(taskId),
                    "data_path" : taskDataPath,
                    "data_task_type" : taskType,
                    "data_task_name" : taskName}
        allTaskslist.append(taskDict)

    allData = allTasksDataset(allTaskslist)
    batchSampler = Batcher(allData, batchSize=args.train_batch_size, seed = args.seed)
    batchSamplerUtils = batchUtils(isTrain = isTrain, modelType= taskParams.modelType,
                                  maxSeqLen = args.max_seq_len)
    multiTaskDataLoader = DataLoader(allData, batch_sampler = batchSampler,
                                collate_fn=batchSamplerUtils.collate_fn,
                                pin_memory=gpu)

    return allData, batchSampler, multiTaskDataLoader

def main():
    allParams = vars(args)
    # loading if load_saved_model
    if args.load_saved_model is not None:
        assert os.path.exists(args.load_saved_model), "saved model not present at {}".format(args.load_saved_model)
        loadedDict = torch.load(args.load_saved_model)
        logger.info('Saved Model loaded from {}'.format(args.load_saved_model))
        logger.info('Any changes made to task file after saving this model shall be ignored')
        '''
        NOTE : -
        taskParams used with this saved model must also be stored. THE SAVED TASK PARAMS 
        SHALL BE USED HERE TO AVOID ANY DISCREPENCIES/CHANGES IN THE TASK FILE.
        Hence, if changes are made to task file after saving this model, they shall be ignored
        '''
        taskParams = loadedDict['task_params']
        logger.info('Task Params loaded from saved model.')
    else:
        taskParams = TasksParam(args.task_file)
        logger.info("Task params object created from task file...")

    allParams['task_params'] = taskParams
    allParams['gpu'] = torch.cuda.is_available()
    #logger.info('task parameters:\n {}'.format(taskParams.taskDetails))

    if args.tensorboard:
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
    allParams['num_train_steps'] = len(multiTaskDataLoaderTrain) *args.epochs // args.grad_accumulation_steps
    allParams['warmup_steps'] = args.num_of_warmup_steps

    logger.info("Making multi-task model...")
    model = multiTaskModel(allParams)
    #logger.info('################ Network ###################')
    #logger.info('\n{}\n'.format(model.network))

    if args.load_saved_model:
        model.load_multi_task_model(loadedDict)
        logger.info('saved model loaded with global step {} from {}'.format(model.globalStep,
                                                                            args.load_saved_model))
        if args.resume_train:
            logger.info("Resuming training from global step {}. Steps before it will be skipped".format(model.globalStep))

    # training 
    resCnt = 0
    for epoch in tqdm(range(args.epochs)):
        logger.info('\n####################### EPOCH {} ###################\n'.format(epoch))
        totalEpochLoss = 0
        for i, (batchMetaData, batchData) in enumerate(multiTaskDataLoaderTrain):
            batchMetaData, batchData = BatchSamplerTrain.patch_data(batchMetaData,batchData, gpu = allParams['gpu'])
            if args.resume_train and args.load_saved_model and resCnt*args.grad_accumulation_steps < model.globalStep:
                '''
                NOTE: - Resume function is only to be used in case the training process couldnt
                complete or you wish to extend the training to some more epochs.
                Please keep the gradient accumulation step the same for exact resuming.
                '''
                resCnt += 1
                continue
            model.update_step(batchMetaData, batchData)
            totalEpochLoss += model.taskLoss.item()

            if model.globalStep % args.log_per_updates == 0 and (model.accumulatedStep+1 == args.grad_accumulation_steps):
                taskId = batchMetaData['task_id']
                taskName = taskParams.taskIdNameMap[taskId]
                avgLoss = totalEpochLoss / ((i+1)*args.train_batch_size) 
                logger.info('Steps: {} Task: {} Avg.Loss: {} Task Loss: {}'.format(model.globalStep,
                                                                                taskName,
                                                                                avgLoss,
                                                                                model.taskLoss.item()))
                if args.tensorboard:
                    tensorboard.add_scalar('train/avg_loss', avgLoss, global_step= model.globalStep)
                    tensorboard.add_scalar('train/{}_loss'.format(taskName),
                                            model.taskLoss.item(),
                                            global_step=model.globalStep)

            if args.save_per_updates > 0 and (model.globalStep % args.save_per_updates)==0 and (model.accumulatedStep+1==args.grad_accumulation_steps):
                savePath = os.path.join(args.out_dir, 'multi_task_model_{}_{}.pt'.format(epoch,
                                                                                        model.globalStep))
                model.save_multi_task_model(savePath)

    #saving model after epoch
    savePath = os.path.join(args.out_dir, 'multi_task_model_{}_{}.pt'.format(epoch, model.globalStep))  
    model.save_multi_task_model(savePath)

if __name__ == "__main__":
    main()
                
