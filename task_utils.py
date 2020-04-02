
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

        #global_map = {}
        classNumMap = {}
        #data_type_map = {}
        taskTypeMap = {}
        metricsMap = {}
        #enable_san_map = {}
        dropoutpMap = {}
        #encoderType_map = {}
        lossMap = {}
        #kd_loss_map = {}
        lossWeightMap = {}

        #uniq_encoderType = set()
        for taskName, taskVals in self.taskDetails.items():
            classNumMap[taskName] = taskVals["class_num"]
            #data_format = DataFormat[task_def["data_format"]]
            #data_type_map[task] = data_format
            taskTypeMap[taskName] = TaskType[taskVals["task_type"]]
            metricsMap[taskName] = tuple(MetricType[metric_name] for metric_name in taskVals["metrics"])
            #enable_san_map[task] = task_def["enable_san"]
            #uniq_encoderType.add(EncoderModelType[task_def["encoder_type"]])
            fileNamesMap[taskName] = list(taskVals["file_names"])
            '''
            if "labels" in task_def:
                labels = task_def["labels"]
                label_mapper = Vocabulary(True)
                for label in labels:
                    label_mapper.add(label)
                global_map[task] = label_mapper
            '''
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

            '''
            if "kd_loss" in task_def:
                t_loss = task_def["kd_loss"]
                loss_crt = LossCriterion[t_loss]
                kd_loss_map[task] = loss_crt
            else:
                kd_loss_map[task] = None
            '''
        #assert len(uniq_encoderType) == 1, 'The shared encoder has to be the same.'
        #self.global_map = global_map
        self.classNumMap = classNumMap
        #self.data_type_map = data_type_map
        self.taskTypeMap = taskTypeMap
        self.metricsMap = metricsMap
        #self.enable_san_map = enable_san_map
        self.fileNamesMap = fileNamesMap
        self.dropoutpMap = dropoutpMap
        #self.modelType = uniq_encoderType.pop()
        self.lossMap = lossMap
        #self.kd_loss_map = kd_loss_map
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


