
import yaml
import os
from collections import OrderedDict
from data_utils import TaskType, MetricType, ModelType, LossType

class TasksParam:
    '''
    This class keeps the details mentioned in the tasks yml file as attributes.
    This class is taken from MT-DNN with some modifications 
    '''
    def __init__(self, taskFilePath):
        # dictioanry holding all the tasks details with 
        # task name as key.
        self.taskDetails = yaml.safe_load(open(taskFilePath))
        self.modelType = self.validity_checks()

        classNumMap = {}
        taskTypeMap = {}
        taskNameIdMap = {}
        taskIdNameMap = OrderedDict()
        metricsMap = {}
        dropoutProbMap = {}
        lossMap = {}
        lossWeightMap = {}
        fileNamesMap = {}

        for i, (taskName, taskVals) in enumerate(self.taskDetails.items()):
            classNumMap[taskName] = taskVals["class_num"]
            taskNameIdMap[taskName] = i
            taskIdNameMap[i] = taskName
            taskTypeMap[taskName] = TaskType[taskVals["task_type"]]
            metricsMap[taskName] = tuple(MetricType[metric_name] for metric_name in taskVals["metrics"])
            fileNamesMap[taskName] = list(taskVals["file_names"])

            if "config_name" in taskVals:
                modelConfig = taskVals["config_name"]
            else:
                modelConfig = None

            if "dropout_prob" in taskVals:
                dropoutProbMap[taskName] = taskVals["dropout_prob"]
            else:
                dropoutProbMap[taskName] = 0.05
            # loss map
            if "loss_type" in taskVals:
                lossMap[taskName] = LossType[taskVals["loss_type"]]
            else:
                lossMap[taskName] = None


            if "loss_weight" in taskVals:
                '''
                loss weight for individual task. This factor 
                will be multiplied directly to the loss calculated
                for backpropagation
                '''
                lossWeightMap[taskName] = float(taskVals["loss_weight"])
            else:
                lossWeightMap[taskName] = float(1.0)

        self.classNumMap = classNumMap
        self.taskTypeMap = taskTypeMap
        self.taskNameIdMap = taskNameIdMap
        self.taskIdNameMap = taskIdNameMap
        self.modelConfig = modelConfig
        self.metricsMap = metricsMap
        self.fileNamesMap = fileNamesMap
        self.dropoutProbMap = dropoutProbMap
        self.lossMap = lossMap
        self.lossWeightMap = lossWeightMap

    def validity_checks(self):
        '''
        Check if the yml has correct form or not.
        '''
        requiredParams = {"class_num", "task_type", "metrics", "loss_type", "file_names"}
        uniqueModel = set()
        uniqueConfig = set()
        for taskName, taskVals in self.taskDetails.items():
            # check task name
            assert taskName.isalpha(), "only alphabets are allowed in task name. No special chracters/numbers/whitespaces allowed. Task Name: %s" % taskName

            # check all required arguments
            assert len(requiredParams.intersection(set(taskVals.keys()))) == len(requiredParams), "following parameters are required {}".format(requiredParams)

            #check is loss, metric. model type is correct
            try:
                LossType[taskVals["loss_type"]]
                ModelType[taskVals["model_type"]]
                [MetricType[m] for m in taskVals["metrics"]]
            except:
                print("allowed loss {}".format(list(LossType)))
                print("allowed model type {}".format(list( ModelType)))
                print("allowed metric type {}".format(list(MetricType)))
                raise

            # check model type, only one model type is allowed for all tasks
            uniqueModel.add(ModelType[taskVals["model_type"]])
            if "config_name" in taskVals:
                uniqueConfig.add(taskVals["config_name"])

            #check if all data files exists for task
            #for fileName in taskVals['file_names']:
                #assert os.path.exists(fileName)

        assert len(uniqueModel) == 1, "Only one type of model can be shared across all tasks"
        assert len(uniqueConfig) <= 1, "Model config has to be same across all shared tasks"

        #return model type from here
        return list(uniqueModel)[0]


