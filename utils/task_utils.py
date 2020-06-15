import yaml
import os
import joblib
from collections import OrderedDict
from utils.data_utils import TaskType, ModelType, LossType, METRICS

class TasksParam:
    '''
    This class keeps the details mentioned in the tasks yml file as attributes.
    '''
    def __init__(self, taskFilePath):
        # dictioanry holding all the tasks details with 
        # task name as key.
        #The idea to store, retrieve task information in yaml file and process using dictionary maps and IntEnum classes
        # is inspired from Microsoft's mt-dnn <https://github.com/namisan/mt-dnn>
        
        self.taskDetails = yaml.safe_load(open(taskFilePath))
        self.modelType = self.validity_checks()

        classNumMap = {}
        taskTypeMap = {}
        taskNameIdMap = {}
        taskIdNameMap = OrderedDict()
        metricsMap = {}
        dropoutProbMap = {}
        lossMap = {}
        labelMap = {}
        lossWeightMap = {}
        fileNamesMap = {}

        for i, (taskName, taskVals) in enumerate(self.taskDetails.items()):
            taskNameIdMap[taskName] = i
            taskIdNameMap[i] = taskName
            taskTypeMap[taskName] = TaskType[taskVals["task_type"]]
            fileNamesMap[taskName] = list(taskVals["file_names"])

            modelConfig = None
            dropoutProbMap[taskName] = 0.05
            lossMap[taskName] = None
            lossWeightMap[taskName] = float(1.0)
            labelMap[taskName] = None
            metricsMap[taskName] = None

            if "class_num" in taskVals:
                classNumMap[taskName] = taskVals["class_num"]

            if "config_name" in taskVals:
                modelConfig = taskVals["config_name"]

            if "dropout_prob" in taskVals:
                dropoutProbMap[taskName] = taskVals["dropout_prob"]
            
            if "metrics" in taskVals:
                metricsMap[taskName] = [m.lower() for m in taskVals["metrics"]]

            # loss map
            if "loss_type" in taskVals:
                lossMap[taskName] = LossType[taskVals["loss_type"]]

            if "label_map_or_file" in taskVals:
                '''
                Label Map is the list of label names (or tag names in NER) which are
                present in the data. We make it into dict. This dict will be used to create the label to index
                map and hence is important to maintain order. It is required in case of 
                NER. For classification tasks, if the labels are already numeric in data,
                label map is not required, but if not, then required.

                DO NOT ADD ANY EXTRA SPECIAL TOKEN LIKE ['CLS'], 'X', ['SEP'] IN LABEL MAP OR COUNT IN CLASS NUMBER

                It can also take the generated label map joblib file from data transformations
                '''
                if type(taskVals["label_map_or_file"]) == list:
                    labelMap[taskName] = {lab:i for i, lab in enumerate(taskVals["label_map_or_file"])}

                elif type(taskVals["label_map_or_file"]) == str:
                    labelMap[taskName] = joblib.load(taskVals["label_map_or_file"])

                else:
                    raise ValueError("label_map_or_file not recognized")
                
                if taskTypeMap[taskName] == TaskType.NER:
                    labelMap[taskName]['[CLS]'] = len(labelMap[taskName])
                    labelMap[taskName]['[SEP]'] = len(labelMap[taskName])
                    labelMap[taskName]['X'] = len(labelMap[taskName])
                    if "O" not in labelMap[taskName]:
                        labelMap[taskName]["O"] = len(labelMap[taskName])
                        
                classNumMap[taskName] = len(labelMap[taskName])

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
        self.labelMap =labelMap
        self.lossWeightMap = lossWeightMap

    def validity_checks(self):
        '''
        Check if the yml has correct form or not.
        '''
        requiredParams = {"task_type", "loss_type", "file_names"}
        uniqueModel = set()
        uniqueConfig = set()
        for taskName, taskVals in self.taskDetails.items():
            # check task name
            assert taskName.isalpha(), "only alphabets are allowed in task name. No special chracters/numbers/whitespaces allowed. Task Name: %s" % taskName

            # check all required arguments
            assert len(requiredParams.intersection(set(taskVals.keys()))) == len(requiredParams), "following parameters are required {}".format(requiredParams)

            #check is loss,  model type is correct
            try:
                LossType[taskVals["loss_type"]]
                ModelType[taskVals["model_type"]]
            except:
                print("allowed loss {}".format(list(LossType)))
                print("allowed model type {}".format(list( ModelType)))
                raise

            # check metric if present
            if "metrics" in taskVals:
                for m in taskVals["metrics"]:
                    assert m.lower() in METRICS, "allowed metrics are {}".format(METRICS.keys())

            # check model type, only one model type is allowed for all tasks
            uniqueModel.add(ModelType[taskVals["model_type"]])
            if "config_name" in taskVals:
                uniqueConfig.add(taskVals["config_name"])

            #check if all data files exists for task
            #for fileName in taskVals['file_names']:
                #assert os.path.exists(fileName)

            #either label map/file is required or class_num is required.
            assert "label_map_or_file" in taskVals or "class_num" in taskVals, "either class_num or label_map_or_file is required"

            # we definitely require label mapping for NER task
            if taskVals["task_type"] == 'NER':
                assert "label_map_or_file" in taskVals, "Unique Tags/Labels or map file needs to be mentioned in label_map_or_file for NER"

        assert len(uniqueModel) == 1, "Only one type of model can be shared across all tasks"
        assert len(uniqueConfig) <= 1, "Model config has to be same across all shared tasks"

        #return model type from here
        return list(uniqueModel)[0]


