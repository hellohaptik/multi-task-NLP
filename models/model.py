import torch
import torch.nn as nn
import logging
from models.dropout import DropoutWrapper
from data_utils import ModelType, NLP_MODELS, TaskType, LOSSES
from transformers import AdamW, get_linear_schedule_with_warmup
logger = logging.getLogger("multi_task")

class multiTaskNetwork(nn.Module):
    def __init__(self, params):
        super(multiTaskNetwork, self).__init__()
        self.params = params
        self.taskParams = self.params['task_params']
        assert self.taskParams.modelType in ModelType._value2member_map_, "Model Type is not recognized, check in data_utils"
        self.modelType = self.taskParams.modelType

        # making shared base encoder model
        # Initializing with a config file does not load the weights associated with the model,
        # only the configuration. Check out the from_pretrained() method to load the model weights.
        #modelName = ModelType(self.modelType).name.lower()
        modelName = self.modelType.name.lower()
        configClass, modelClass, tokenizerClass, defaultName = NLP_MODELS[modelName]
        if self.taskParams.modelConfig is not None:
            logger.info('Making shared model from given config name {}'.format(self.taskParams.modelConfig))
            self.sharedModel = modelClass.from_pretrained(self.taskParams.modelConfig)
        else:
            logger.info("Making shared model with default config")
            self.sharedModel = modelClass.from_pretrained(defaultName)
        self.hiddenSize = self.sharedModel.config.hidden_size
        
        #making headers
        self.allDropouts, self.allHeaders = self.make_multitask_heads()
        self.initialize_headers()
    def make_multitask_heads(self):
        '''
        Function to make task specific headers for all tasks.
        hiddenSize :- Size of the encoder hidden state output from shared model. Hidden state will be 
        used as input to headers. Hence size is required to make input layer for header.

        The final layer output size depends on the number of classes expected in the output
        '''
        allHeaders = nn.ModuleList()
        allDropouts = nn.ModuleList()

        # taskIdNameMap is orderedDict, it will preserve the order of tasks
        for taskId, taskName in self.taskParams.taskIdNameMap.items():
            taskType = self.taskParams.taskTypeMap[taskName]
            numClasses = int(self.taskParams.classNumMap[taskName])
            dropoutValue = self.taskParams.dropoutProbMap[taskName]
            if taskType == TaskType.Span:
                assert numClasses == 2, " Span required num classes to be 2"

            dropoutLayer = DropoutWrapper(dropoutValue)
            outLayer = nn.Linear(self.hiddenSize, numClasses)
            allDropouts.append(dropoutLayer)
            allHeaders.append(outLayer)
            

        return allDropouts, allHeaders

    def initialize_headers(self):
        def init_weights(module):
            if isinstance(module, (nn.Linear, nn.Embedding)):
                # Slightly different from the TF version which uses truncated_normal for initialization
                # cf https://github.com/pytorch/pytorch/pull/5617
                module.weight.data.normal_(mean=0.0, std=0.02 * 1.0)
            if isinstance(module, nn.Linear):
                if module.bias is not None:
                    module.bias.data.zero_()

        self.apply(init_weights)

    def forward(self, tokenIds, typeIds, attentionMasks, taskId):

        # taking out output from shared encoder. 
        outputs = self.sharedModel(input_ids = tokenIds,
                                token_type_ids = typeIds,
                                attention_mask = attentionMasks)
        sequenceOutput = outputs[0]
        pooledOutput = outputs[1]
        taskType = self.taskParams.taskTypeMap[self.taskParams.taskIdNameMap[taskId]]
        if taskType == TaskType.Span:
            #adding dropout layer after shared output
            sequenceOutput = self.allDropouts[taskId](sequenceOutput)
            #adding task specific header
            finalOutLogits = self.allHeaders[taskId](sequenceOutput)
            #as this is span, logits will have 2 entries, one for start and other for end
            startLogits, endLogits = finalOutLogits.split(1, dim=1)
            startLogits = startLogits.squeeze(-1)
            endLogits = endLogits.squeeze(-1)

            return startLogits, endLogits
        else:
            #adding dropout layer after shared output
            pooledOutput = self.allDropouts[taskId](pooledOutput)
            #adding task specific header
            logits = self.allHeaders[taskId](pooledOutput)
            
            return logits

class multiTaskModel:
    '''
    This is the model helper class which is responsible for building the 
    model architecture and training. It has following functions
    1. Make the multi-task network
    2. set optimizer with linear scheduler and warmup
    3. Multi-gpu support
    4. Task specific loss function
    5. Model update for training
    6. Predict function for inference
    '''
    def __init__(self, params):
        self.params = params
        self.taskParams = self.params['task_params']

        self.globalStep = 0
        self.accumulatedStep = 0

        # making model
        if torch.cuda.device_count() > 1:
            logger.info("Using number of gpus: {}".format(torch.cuda.device_count()))
            self.network = nn.DataParallel(multiTaskNetwork(params))
        else:
            self.network = multiTaskNetwork(params)
            logger.info('Using single GPU')

        # transfering to gpu if available
        if self.params['gpu']:
            self.network.cuda()

        #optimizer and scheduler
        self.optimizer, self.scheduler = self.make_optimizer(numTrainSteps=self.params['num_train_steps'],
                                                            warmupSteps=self.params['warmup_steps'])
        #loss class list
        self.lossClassList = self.make_loss_list()


    def make_optimizer(self, numTrainSteps, lr = 2e-5, eps = 1e-8, warmupSteps=0):
        # we will use AdamW optimizer from huggingface transformers. This optimizer is 
        #widely used with BERT. It is modified form of Adam which is used in Tensorflow 
        #implementations
        optimizer = AdamW(self.network.parameters(), lr=lr, eps = eps)

        # lr scheduler
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=warmupSteps,
                                                    num_training_steps=numTrainSteps)
        logger.debug("optimizer and scheduler created")
        return optimizer, scheduler

    def make_loss_list(self):
        # making loss class list according to task id
        lossClassList = []
        for taskId, taskName in self.taskParams.taskIdNameMap.items():
            lossName = self.taskParams.lossMap[taskName].name.lower()
            lossClass = LOSSES[lossName]()
            lossClassList.append(lossClass)
        return lossClassList

    def _to_cuda(self, tensor):
        # Function directly taken from MT-DNN 
        if tensor is None: return tensor

        if isinstance(tensor, list) or isinstance(tensor, tuple):
            y = [e.cuda(non_blocking=True) for e in tensor]
            for e in y:
                e.requires_grad = False
        else:
            y = tensor.cuda(non_blocking=True)
            y.requires_grad = False
        return y 

    def update_step(self, batchMetaData, batchData):
        #changing to train mode
        self.network.train()
        target = batchData[batchMetaData['label_pos']]

        #transfering label to gpu if present
        if self.params['gpu']:
            target = self._to_cuda(target)

        taskType = batchMetaData['task_type']
        taskId = batchMetaData['task_id']
        logger.debug('task id for batch {}'.format(taskId))
        #making forward pass
        #batchData: [tokenIdsBatchTensor, typeIdsBatchTensor, masksBatchTensor, labelsTensor]
        #model forward function input [tokenIdsBatchTensor, typeIdsBatchTensor, masksBatchTensor, taskId]
        # we are not going to send labels in batch
        logger.debug('len of batch data {}'.format(len(batchData)))
        logger.debug('label position in batch data {}'.format(batchMetaData['label_pos']))

        modelInputs = batchData[:batchMetaData['label_pos']]
        modelInputs += [taskId]

        logger.debug('size of model inputs {}'.format(len(modelInputs)))
        logits = self.network(*modelInputs)
        #calculating task loss
        self.taskLoss = 0
        logger.debug('size of model output logits {}'.format(logits.size()))
        logger.debug('size of target {}'.format(target.size()))
        if self.lossClassList[taskId] and (target is not None):
            self.taskLoss = self.lossClassList[taskId](logits, target)
            #tensorboard details
            if self.params['tensorboard']:
                self.tbTaskId = taskId
                self.tbTaskLoss = self.taskLoss.item()
        taskLoss = self.taskLoss / self.params['grad_accumulation_steps']
        taskLoss.backward()
        self.accumulatedStep += 1

        #gradients will be updated only when accumulated steps become
        #mentioned number in grad_acc_steps (one global update)
        if self.accumulatedStep == self.params['grad_accumulation_steps']:
            logging.debug('model updated.')
            if self.params['grad_clip_value'] > 0:
                torch.nn.utils.clip_grad_norm_(self.network.parameters(),
                                                self.params['grad_clip_value'])

            self.optimizer.step()
            self.scheduler.step()

            self.optimizer.zero_grad()
            self.globalStep += 1
            #resetting accumulated steps
            self.accumulatedStep = 0

    def save_multi_task_model(self, savePath):
        '''
        We will save the model parameters with state dict.
        Also the current optimizer state.
        Along with the current global_steps and epoch which would help
        for resuming training
        Plus we will save the task parameters object created from the task file.
        The same object shall be used for this saved model
        '''
        modelStateDict = {k : v.cpu() for k,v in self.network.state_dict().items()}
        toSave = {'model_state_dict' :modelStateDict,
                'optimizer_state' : self.optimizer.state_dict(),
                'scheduler_state' : self.scheduler.state_dict(),
                'global_step' : self.globalStep,
                'task_params' : self.taskParams}
        torch.save(toSave, savePath)
        logger.info('model saved in {} global step at {}'.format(self.globalStep, savePath))

    def load_multi_task_model(self, loadedDict):
        self.network.load_state_dict(loadedDict['model_state_dict'])
        self.optimizer.load_state_dict(loadedDict['optimizer_state'])
        self.scheduler.load_state_dict(loadedDict['scheduler_state'])
        self.globalStep = loadedDict['global_step']