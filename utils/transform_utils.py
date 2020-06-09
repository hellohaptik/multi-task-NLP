import yaml
import os
import joblib
from utils.data_utils import TRANSFORM_FUNCS

class TransformParams:
    '''
    This class keeps the details mentioned in transform yaml file for the 
    case when data transformations is required to be performed.
    '''
    def __init__(self, transformFilePath):

        self.transformDetails = yaml.safe_load(open(transformFilePath))
        self.validity_checks()
        transformFnMap = {}
        transformParamsMap = {}
        readFileNamesMap = {}
        readDirMap = {}
        saveDirMap = {}

        for i, (transformName, transformVals) in enumerate(self.transformDetails.items()):
            transformFnMap[transformName] = transformVals['transform_func']
            transformParamsMap[transformName] = {}
            readFileNamesMap[transformName] = list(transformVals['read_file_names'])
            readDirMap[transformName] = transformVals['read_dir']
            saveDirMap[transformName] = transformVals['save_dir']

            if 'transform_params' in transformVals:
                transformParamsMap[transformName] = dict(transformVals['transform_params'])

        self.transformFnMap = transformFnMap
        self.transformParamsMap = transformParamsMap
        self.readFileNamesMap = readFileNamesMap
        self.readDirMap = readDirMap
        self.saveDirMap = saveDirMap

    def validity_checks(self):
        '''
        Check if the transform yml is correct or not
        '''
        requiredParams = {"transform_func", "read_dir", "read_file_names", "save_dir"}
        for i, (transformName, transformVals) in enumerate(self.transformDetails.items()):
            # check all required arguments
            assert len(requiredParams.intersection(set(transformVals.keys()))) == len(requiredParams), "following parameters are required {}".format(requiredParams)

            #check if transform functions is in the defined transform function
            assert transformVals['transform_func'] in TRANSFORM_FUNCS.keys(), "{} transform fn is not in following defined functions {}".format(transformVals['transform_func'],
                                                                                                                                                TRANSFORM_FUNCS.keys())



            




