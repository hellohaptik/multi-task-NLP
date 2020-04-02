
import yaml
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
        metricsMap = {}
        dropoutpMap = {}
        lossMap = {}
        lossWeightMap = {}

        for taskName, taskVals in self.taskDetails.items():
            classNumMap[taskName] = taskVals["class_num"]
            taskTypeMap[taskName] = TaskType[taskVals["task_type"]]
            metricsMap[taskName] = tuple(MetricType[metric_name] for metric_name in taskVals["metrics"])
            fileNamesMap[taskName] = list(taskVals["file_names"])

            if "dropout_p" in taskVals:
                dropoutpMap[taskName] = taskVals["dropout_p"]
            # loss map
            if "loss_type" in taskVals:
                lossMap[taskName] = LossType[taskVals["loss"]]
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
        self.metricsMap = metricsMap
        self.fileNamesMap = fileNamesMap
        self.dropoutpMap = dropoutpMap
        self.lossMap = lossMap
        self.lossWeightMap = lossWeightMap

    def validity_checks(self):
        '''
        Check if the yml has correct form or not.
        '''
        requiredParams = {"class_num", "task_type", "metrics", "loss_type", "file_names"}
        uniqueModel = set()
        for taskName, taskVals in self.taskDetails:
            # check task name
            assert taskName.isalpha(), "only alphabets are allowed in task name. No special chracters/numbers/whitespaces allowed. Task Name: %s" % taskName

            # check all required arguments
            assert len(requiredParams.intersection(set(taskVals.keys()))) == len(requiredParams),
            "following parameters are required {}".format(requiredParams)

            #check is loss, metric. model type is correct
            try:
                LossType[taskVals["loss_type"]]
                ModelType[taskVals["model_type"]]
                MetricType[taskVals["metrics"]]
            except KeyError:
                print("allowed loss {}".format(list(LossType)))
                print("allowed model type {}".format(list( ModelType)))
                print("allowed metric type {}".format(list(MetricType)))

            # check model type, only one model type is allowed for all tasks
            uniqueModel.add(ModelType[taskVals["model_type"]])

        assert len(uniqueModel) == 1, "Only one type of model can be shared across all tasks"

        #return model type from here
        return list(uniqueModel)[0]


