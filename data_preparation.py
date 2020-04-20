
import argparse
import os
import json
import multiprocessing as mp
from data_utils import TaskType, ModelType, NLP_MODELS
from task_utils import TasksParam
from tqdm import tqdm, trange

def load_data(dataPath, dataType):
    '''
    This fn loads data from tsv file in according to the format in dataType
    dataPath - path/name of file to read
    dataType - Type of task for which format will be set. Can be 
                Single Sentence Classification
                Sentence Pait Classification
                Span Prediction (MRC)

    Function taken from MT_DNN with modification
    '''
    allData = []
    for line in open(dataPath):
        cols = line.strip("\n").split("\t")

        if dataType == TaskType.SingleSenClassification:
            #print(fields)
            assert len(cols) == 3, "Data is not in Single Sentence Classification format"
            row = {"uid": cols[0], "label": int(cols[1]), "sentenceA": cols[2]}

        elif dataType == TaskType.SentencePairClassification:
            #print(fields)
            assert len(cols) == 4, "Data is not in Sentence Pair Classification format"
            row = {
                "uid": cols[0], "label": cols[1],"sentenceA": cols[2], "sentenceB": cols[3]}

        elif dataType == TaskType.Span:
            assert len(cols) == 4, "Data is not in Span format"
            row = {
                "uid": cols[0],
                "label": cols[1],
                "sentenceA": cols[2],
                "sentenceB": cols[3]}
        else:
            raise ValueError(dataType)

        allData.append(row)

    return allData

def standard_data_converter(maxSeqLen, tokenizer, senA, senB = None):
    '''
    If the data is sentence Pair, -> [CLS]senA[SEP]senB[SEP]
    If data is single sentence -> [CLS]senA[SEP]

    Truncation stategy will truncate the 2nd sentence only (that is passage.
    This would be helpful as we don't want to truncate the query). The strategy can 
    be changed to 'longest_first' or other if required.

    Different model encoders require different inputs. Some encoders doesn't support type_ids, some 
    doesn't support attention_mask. Hence, to support multiple encoders, typeIds and mask would be intially kept None
    '''
    typeIds = None
    mask = None
    if senB:
        out = tokenizer.encode_plus(senA, senB, add_special_tokens = True,
                                    truncation_strategy = 'only_second', max_length = maxSeqLen,
                                    pad_to_max_length = True)
    else:
        out = tokenizer.encode_plus(senA, add_special_tokens=True,
                                    truncation_strategy ='only_first',
                                    max_length = maxSeqLen, pad_to_max_length=True)

    tokenIds = out['input_ids']
    if 'token_type_ids' in out.keys():
        typeIds = out['token_type_ids']
    if 'attention_mask' in out.keys():
        mask = out['attention_mask']
        
    return tokenIds, typeIds, mask


def create_data_single_sen_classification(data, chunkNumber, tempList, maxSeqLen, tokenizer):
    name = 'single_sen_{}.json'.format(str(chunkNumber))
    with open(name, 'w') as wf:
        with tqdm(total = len(data), position = chunkNumber) as progress:
            for idx, sample in enumerate(data):
                ids = sample['uid']
                senA = sample['sentenceA']
                label = sample['label']
            
                inputIds, typeIds, inputMask = standard_data_converter(maxSeqLen, tokenizer, senA)
                features = {
                    'uid': ids,
                    'label': label,
                    'token_id': inputIds,
                    'type_id': typeIds,
                    'mask': inputMask}
                wf.write('{}\n'.format(json.dumps(features)))
                progress.update(1)
            tempList.append(name)

def create_data_sentence_pair_classification(data, chunkNumber, tempList, maxSeqLen, tokenizer):
    name = 'sentence_pair_{}.json'.format(str(chunkNumber))
    with open(name, 'w') as wf:
        with tqdm(total = len(data), position = chunkNumber) as progress:    
            for idx, sample in enumerate(data):
                ids = sample['uid']
                senA = sample['sentenceA']
                senB = sample['sentenceB']
                label = sample['label']
            
                inputIds, typeIds, inputMask = standard_data_converter(maxSeqLen, tokenizer, senA, senB)
                features = {
                    'uid': ids,
                    'label': label,
                    'token_id': inputIds,
                    'type_id': typeIds,
                    'mask': inputMask}

                wf.write('{}\n'.format(json.dumps(features)))
                progress.update(1)
        tempList.append(name)

def create_data_span_prediction(data, chunkNumber, tempList, maxSeqLen, tokenizer):
    name = 'span_prediction_{}.json'.format(str(chunkNumber))
    '''
    with open(name, 'w') as wf:
        unique_id = 1000000000 
        for example_index, sample in enumerate(data):
            ids = sample['uid']
            doc = sample['sentenceA']
            query = sample['sentenceB']
            label = sample['label']
            doc_tokens, cw_map = squad_utils.token_doc(doc)
            answer_start, answer_end, answer, is_impossible = squad_utils.parse_squad_label(label)
            answer_start_adjusted, answer_end_adjusted = squad_utils.recompute_span(answer, answer_start, cw_map)
            is_valid = squad_utils.is_valid_answer(doc_tokens, answer_start_adjusted, answer_end_adjusted, answer)
            if not is_valid: continue
            feature_list = squad_utils.mrc_feature(tokenizer,
                                    unique_id,
                                    example_index,
                                    query,
                                    doc_tokens,
                                    answer_start_adjusted,
                                    answer_end_adjusted,
                                    is_impossible,
                                    max_seq_len,
                                    MAX_QUERY_LEN,
                                    DOC_STRIDE,
                                    answer_text=answer,
                                    is_training=True)
            unique_id += len(feature_list)
            for feature in feature_list:
                so = json.dumps({'uid': ids,
                            'token_id' : feature.input_ids,
                            'mask': feature.input_mask,
                            'type_id': feature.segment_ids,
                            'example_index': feature.example_index,
                            'doc_span_index':feature.doc_span_index,
                            'tokens': feature.tokens,
                            'token_to_orig_map': feature.token_to_orig_map,
                            'token_is_max_context': feature.token_is_max_context,
                            'start_position': feature.start_position,
                            'end_position': feature.end_position,
                            'label': feature.is_impossible,
                            'doc': doc,
                            'doc_offset': feature.doc_offset,
                            'answer': [answer]})
                writer.write('{}\n'.format(so))

            tempList.append(name)
    '''
            
def create_data_multithreaded(data, wrtPath, tokenizer, taskType, maxSeqLen, multithreaded):
    '''
    This function uses multi-processing to create the data in the required format
    for base models as per the task. Utilizing multiple Cores help in processing
    huge data with speed
    '''
    man = mp.Manager()

    # shared list to store all temp files written by processes
    tempFilesList = man.list()
    numProcess = 1
    if multithreaded:
        numProcess = mp.cpu_count() - 1

    '''
    Dividing the entire data into chunks which can be sent to different processes.
    Each process will write its chunk into a file. 
    After all processes are done writing, we will combine all the files into one
    '''
    chunkSize = int(len(data) / (numProcess))
    print('Data Size: ', len(data))
    print('number of threads: ', numProcess)

    processes = []
    for i in range(numProcess):
        dataChunk = data[chunkSize*i : chunkSize*(i+1)]

        if taskType == TaskType.SingleSenClassification:
            p = mp.Process(target = create_data_single_sen_classification, args = (dataChunk, i, tempFilesList, maxSeqLen, tokenizer))

        if taskType == TaskType.SentencePairClassification:
            p = mp.Process(target = create_data_sentence_pair_classification, args = (dataChunk, i, tempFilesList, maxSeqLen, tokenizer))
        
        if taskType == TaskType.Span:
            p = mp.Process(target = create_data_span_prediction, args = (dataChunk, i, tempFilesList, maxSeqLen, tokenizer))

        p.start()
        #print('Process started: ', p)
        processes.append(p)
        
    for pr in processes:
        pr.join()
    
    # combining the files written by multiple processes into a single final file
    with open(wrtPath, 'w') as f:
        for file in tempFilesList:
            with open(file, 'r') as r:
                for line in r:
                    sample =  json.loads(line)
                    f.write('{}\n'.format(json.dumps(sample)))
            os.remove(file)

def main():

    # taking in arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_file', type=str, default="tasks_file.yml")
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--do_lower_case', type=bool, default=True)
    parser.add_argument('--max_seq_len', type=int, default = 384,
                        help = "max sequence length for making data for model")
    parser.add_argument('--multithreaded', type = bool, default = True,
                        help = "use multiple threads for processing data with speed")
    args = parser.parse_args()

    tasks = TasksParam(args.task_file)
    print('task object created from task file...')
    assert os.path.exists(args.data_dir), "data dir doesnt exist"
    modelName = tasks.modelType.name.lower()
    configClass, modelClass, tokenizerClass, defaultName = NLP_MODELS[modelName]
    configName = tasks.modelConfig
    if configName is None:
        configName = defaultName
    
    #making tokenizer for model
    tokenizer = tokenizerClass.from_pretrained(configName)
    print('{} model tokenizer loaded for config {}'.format(modelName, configName))
    dataPath = os.path.join(args.data_dir, '{}_prepared_data'.format(configName))
    if not os.path.exists(dataPath):
        os.mkdir(dataPath)

    for taskId, taskName in tasks.taskIdNameMap.items():
        for file in tasks.fileNamesMap[taskName]:
            print('Loading raw data for task {} from {}'.format(taskName, os.path.join(args.data_dir, file)))
            rows = load_data(os.path.join(args.data_dir, file), tasks.taskTypeMap[taskName])
            wrtFile = os.path.join(dataPath, '{}.json'.format(file.split('.')[0]))
            print('Processing Started...')
            create_data_multithreaded(rows, wrtFile, tokenizer, tasks.taskTypeMap[taskName],
                                    args.max_seq_len, args.multithreaded)
            print('Data Processing done for {}. File saved at {}'.format(taskName, wrtFile))
            
if __name__ == "__main__":
    main()