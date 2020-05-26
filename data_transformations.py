'''
For transforming the raw data in different formats to standard tsv format 
to be consumed for multi-task
'''
import argparse
import os
from utils.transform_utils import TransformParams
from utils.data_utils import TRANSFORM_FUNCS

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--transform_file', type=str, required=True,
                        default='transform_file.yml', help="path to the yml tranform file")
    args = parser.parse_args()
    #making transform params
    transformParams = TransformParams(args.transform_file)

    for transformName, transformFn in transformParams.transformFnMap.items():
        transformParameters = transformParams.transformParamsMap[transformName]
        dataDir = transformParams.readDirMap[transformName]
        assert os.path.exists(dataDir), "{} doesnt exist".format(dataDir)
        saveDir = transformParams.saveDirMap[transformName]
        if not os.path.exists(saveDir):
            os.makedirs(saveDir)
        isTrain = True
        for file in transformParams.readFileNamesMap[transformName]:
            #calling respective transform function over file
            TRANSFORM_FUNCS[transformFn](dataDir = dataDir, readFile=file,
                                        wrtDir=saveDir, transParamDict=transformParameters,
                                        isTrainFile=isTrain)
            # only the first file will be considered as train file for making label map
            isTrain = False
        
        
if __name__ == "__main__":
    main()