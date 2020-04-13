import torch
import torch.nn as nn
from dropout import DropoutWrapper
from data_utils import ModelType, NLP_MODELS, TaskType

class multiTaskModel(nn.module):
    def __init__(self, params):
        super(multiTaskModel, self).__init__()
        self.params = params
        self.allTasks = self.params['task_params']
        assert self.params['model_type'] in ModelType._value2member_map_, "Model Type is recognized, check in data_utils"
        self.modelType = self.params['model_type']

        #making shared base encoder model
        # Initializing with a config file does not load the weights associated with the model,
        # only the configuration. Check out the from_pretrained() method to load the model weights.
        modelName = ModelType(self.modelType).name.lower()
        configClass, modelClass, tokenizerClass, defaultName = NLP_MODELS[modelName]
        if self.params["config_name"] is not None:
            print('Making shared model from given config name {}'.format(self.params["config_name"]))
            self.sharedModel = modelClass.from_pretrained(self.params["config_name"])
        else:
            print("Making shared model with default config")
            self.sharedModel = modelClass.from_pretrained(defaultName)
        self.hiddenSize = self.sharedModel.config.hidden_size
        
        #making headers
        self.allDropouts, self.allHeaders = make_multitask_heads()

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
        for taskId, taskName in self.allTasks.taskIdNameMap.items():
            taskType = self.allTasks.taskTypeMap[taskName]
            numClasses = int(self.allTasks.classNumMap[taskName])
            dropoutValue = self.allTasks.dropoutProbMap[taskName]
            if taskType == TaskType.Span:
                assert numClasses == 2, " Span required num classes to be 2"

            dropoutLayer = DropoutWrapper(dropoutValue)
            outLayer = nn.Linear(self.hiddenSize, numClasses)
            allDropouts.append(dropoutLayer)
            allHeaders.append(outLayer)
            

        return allDropouts, allHeaders

    def initialize_headers(self):

    def forward(self, tokenIds, typeIds, attentionMasks, taskId):

        # taking out output from shared encoder. 
        outputs = self.sharedModel(input_ids = tokenIds,
                                token_type_ids = typeIds,
                                attention_mask = attentionMasks)
        sequenceOutput = outputs[0]
        pooledOutput = outputs[1]
        taskType = self.allTasks.taskTypeMap[self.allTasks.taskIdNameMap[taskId]]
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






