import joblib
import argparse
import os
import re
import json
import random
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from statistics import median
from sklearn.model_selection import train_test_split
SEED = 42

def bio_ner_to_tsv(dataDir, readFile, wrtDir, transParamDict, isTrainFile=False):

    """
    This function transforms the BIO style data and transforms into the tsv format required
    for NER. Following transformed files are written at wrtDir,

    - NER transformed tsv file.
    - NER label map joblib file.

    For using this transform function, set ``transform_func`` : **bio_ner_to_tsv** in transform file.

    Args:
        dataDir (:obj:`str`) : Path to the directory where the raw data files to be read are present..
        readFile (:obj:`str`) : This is the file which is currently being read and transformed by the function.
        wrtDir (:obj:`str`) : Path to the directory where to save the transformed tsv files.
        transParamDict (:obj:`dict`, defaults to :obj:`None`): Dictionary requiring the following parameters as key-value
            
            - ``save_prefix`` (defaults to 'bio_ner') : save file name prefix.
            - ``col_sep`` : (defaults to " ") : separator for columns
            - ``tag_col`` (defaults to 1) : column number where label NER tag is present for each row. Counting starts from 0.
            - ``sen_sep`` (defaults to " ") : end of sentence separator. 
    
    """

    transParamDict.setdefault("save_prefix", "bio_ner")
    transParamDict.setdefault("tag_col", 1)
    transParamDict.setdefault("col_sep", " ")
    transParamDict.setdefault("sen_sep", "\n")

    f = open(os.path.join(dataDir,readFile))

    nerW = open(os.path.join(wrtDir, '{}_{}.tsv'.format(transParamDict["save_prefix"], 
                                                        readFile.split('.')[0])), 'w')
    labelMapNer = {}
    sentence = []
    senLens = []
    labelNer = []
    uid = 0
    print("Making data from file {} ...".format(readFile))
    for i, line in enumerate(f):
        if i%5000 == 0:
            print("Processing {} rows...".format(i))

        line = line.strip(' ') #don't use strip empty as it also removes \n
        wordSplit = line.rstrip('\n').split(transParamDict["col_sep"])
        if len(line)==0 or line[0]==transParamDict["sen_sep"]:
            if len(sentence) > 0:
                nerW.write("{}\t{}\t{}\n".format(uid, labelNer, sentence))
                senLens.append(len(sentence))
                #print("len of sentence :", len(sentence))
                sentence = []
                labelNer = []
                uid += 1
            continue
        sentence.append(wordSplit[0])
        labelNer.append(wordSplit[int(transParamDict["tag_col"])])
        if isTrainFile:
            if wordSplit[int(transParamDict["tag_col"])] not in labelMapNer:
                # ONLY TRAIN FILE SHOULD BE USED TO CREATE LABEL MAP FILE.
                labelMapNer[wordSplit[-1]] = len(labelMapNer)
    
    print("NER File Written at {}".format(wrtDir))
    #writing label map
    if labelMapNer != {} and isTrainFile:
        print("Created NER label map from train file {}".format(readFile))
        print(labelMapNer)
        labelMapNerPath = os.path.join(wrtDir, "{}_{}_label_map.joblib".format(transParamDict["save_prefix"], readFile.split('.')[0]) )
        joblib.dump(labelMapNer, labelMapNerPath)
        print("label Map NER written at {}".format(labelMapNerPath))


    f.close()
    nerW.close()

    print('Max len of sentence: ', max(senLens))
    print('Mean len of sentences: ', sum(senLens)/len(senLens))
    print('Median len of sentences: ', median(senLens))    

def snips_intent_ner_to_tsv(dataDir, readFile, wrtDir, transParamDict, isTrainFile=False):

    """
    This function transforms the data present in snips_data/. 
    Raw data is in BIO tagged format with the sentence intent specified at the end of each sentence.
    The transformation function converts the each raw data file into two separate tsv files,
    one for intent classification task and another for NER task. Following transformed files are written at wrtDir

    - NER transformed tsv file.
    - NER label map joblib file.
    - intent transformed tsv file.
    - intent label map joblib file.

    For using this transform function, set ``transform_func`` : **snips_intent_ner_to_tsv** in transform file.

    Args:
        dataDir (:obj:`str`) : Path to the directory where the raw data files to be read are present..
        readFile (:obj:`str`) : This is the file which is currently being read and transformed by the function.
        wrtDir (:obj:`str`) : Path to the directory where to save the transformed tsv files.
        transParamDict (:obj:`dict`, defaults to :obj:`None`): Dictionary of function specific parameters. Not required for this transformation function.

    """

    f = open(os.path.join(dataDir,readFile))

    nerW = open(os.path.join(wrtDir, 'ner_{}.tsv'.format(readFile.split('.')[0])), 'w')
    intW = open(os.path.join(wrtDir, 'intent_{}.tsv'.format(readFile.split('.')[0]) ), 'w')

    labelMapNer = {}
    labelMapInt = {}

    sentence = []
    label = []
    uid = 0
    print("Making data from file {} ...".format(readFile))
    for i, line in enumerate(f):
        if i%5000 == 0:
            print("Processing {} rows...".format(i))

        line = line.strip(' ') #don't use strip empty as it also removes \n
        wordSplit = line.rstrip('\n').split(' ')
        if line == '\n':

            nerW.write("{}\t{}\t{}\n".format(uid, label, sentence))
            sentence = []
            label = []
            uid += 1

        elif len(wordSplit) == 1:
            intent = wordSplit[0]
            query = ' '.join(sentence)
            intW.write("{}\t{}\t{}\n".format(uid, intent, query))
            if isTrainFile and intent not in labelMapInt:
                labelMapInt[intent] = len(labelMapInt)
        else:
            sentence.append(wordSplit[0])
            label.append(wordSplit[-1])
            if isTrainFile and wordSplit[-1] not in labelMapNer:
                # ONLY TRAIN FILE SHOULD BE USED TO CREATE LABEL MAP FILE.
                labelMapNer[wordSplit[-1]] = len(labelMapNer)

    print("NER File Written at {}".format(wrtDir))
    print("Intent File Written at {}".format(wrtDir))

    #writing label map
    if labelMapNer != {} and isTrainFile:
        print("Created NER label map from train file {}".format(readFile))
        print(labelMapNer)
        labelMapNerPath = os.path.join(wrtDir, "ner_{}_label_map.joblib".format(readFile.split('.')[0]))
        joblib.dump(labelMapNer, labelMapNerPath)
        print("label Map NER written at {}".format(labelMapNerPath))

    if labelMapInt != {} and isTrainFile:
        print("Created Intent label map from train file {}".format(readFile))
        print(labelMapInt)
        labelMapIntPath = os.path.join(wrtDir, "int_{}_label_map.joblib".format(readFile.split('.')[0]))
        joblib.dump(labelMapInt, labelMapIntPath)
        print("label Map Intent written at {}".format(labelMapIntPath))

    f.close()
    nerW.close()
    intW.close()
    

def coNLL_ner_pos_to_tsv(dataDir, readFile, wrtDir, transParamDict, isTrainFile=False):
    
    """
    This function transforms the data present in coNLL_data/. 
    Raw data is in BIO tagged format with the POS and NER tags separated by space.
    The transformation function converts the each raw data file into two separate tsv files,
    one for POS tagging task and another for NER task. Following transformed files are written at wrtDir

    - NER transformed tsv file.
    - NER label map joblib file.
    - POS transformed tsv file.
    - POS label map joblib file.

    For using this transform function, set ``transform_func`` : **snips_intent_ner_to_tsv** in transform file.

    Args:
        dataDir (:obj:`str`) : Path to the directory where the raw data files to be read are present..
        readFile (:obj:`str`) : This is the file which is currently being read and transformed by the function.
        wrtDir (:obj:`str`) : Path to the directory where to save the transformed tsv files.
        transParamDict (:obj:`dict`, defaults to :obj:`None`): Dictionary of function specific parameters. Not required for this transformation function.

    """

    
    f = open(os.path.join(dataDir, readFile))

    nerW = open(os.path.join(wrtDir, 'ner_{}.tsv'.format(readFile.split('.')[0])), 'w')
    posW = open(os.path.join(wrtDir, 'pos_{}.tsv'.format(readFile.split('.')[0])), 'w')

    labelMapNer = {}
    labelMapPos = {}

    sentence = []
    senLens = []
    labelNer = []
    labelPos = []
    uid = 0
    print("Making data from file {} ...".format(readFile))
    for i, line in enumerate(f):
        if i%5000 == 0:
            print("Processing {} rows...".format(i))

        line = line.strip(' ') #don't use strip empty as it also removes \n
        wordSplit = line.rstrip('\n').split(' ')
        if len(line)==0 or line.startswith('-DOCSTART') or line[0]=="\n":
            if len(sentence) > 0:
                nerW.write("{}\t{}\t{}\n".format(uid, labelNer, sentence))
                posW.write("{}\t{}\t{}\n".format(uid, labelPos, sentence))
                senLens.append(len(sentence))
                #print("len of sentence :", len(sentence))

                sentence = []
                labelNer = []
                labelPos = []
                uid += 1
            continue
            
        sentence.append(wordSplit[0])
        labelPos.append(wordSplit[-2])
        labelNer.append(wordSplit[-1])
        if isTrainFile:
            if wordSplit[-1] not in labelMapNer:
                # ONLY TRAIN FILE SHOULD BE USED TO CREATE LABEL MAP FILE.
                labelMapNer[wordSplit[-1]] = len(labelMapNer)
            if wordSplit[-2] not in labelMapPos:
                labelMapPos[wordSplit[-2]] = len(labelMapPos)
    
    print("NER File Written at {}".format(wrtDir))
    print("POS File Written at {}".format(wrtDir))
    #writing label map
    if labelMapNer != {} and isTrainFile:
        print("Created NER label map from train file {}".format(readFile))
        print(labelMapNer)
        labelMapNerPath = os.path.join(wrtDir, "ner_{}_label_map.joblib".format(readFile.split('.')[0]))
        joblib.dump(labelMapNer, labelMapNerPath)
        print("label Map NER written at {}".format(labelMapNerPath))

    if labelMapPos != {} and isTrainFile:
        print("Created POS label map from train file {}".format(readFile))
        print(labelMapPos)
        labelMapPosPath = os.path.join(wrtDir, "pos_{}_label_map.joblib".format(readFile.split('.')[0]))
        joblib.dump(labelMapPos, labelMapPosPath)
        print("label Map POS written at {}".format(labelMapPosPath))

    f.close()
    nerW.close()
    posW.close()

    print('Max len of sentence: ', max(senLens))
    print('Mean len of sentences: ', sum(senLens)/len(senLens))
    print('Median len of sentences: ', median(senLens))

def snli_entailment_to_tsv(dataDir, readFile, wrtDir, transParamDict, isTrainFile=False):

    """
    This function transforms the SNLI entailment data available at `SNLI <https://nlp.stanford.edu/projects/snli/snli_1.0.zip>`_ 
    for sentence pair entailment task. Contradiction and neutral labels are mapped to 0 representing non-entailment scenario. Only 
    entailment label is mapped to 1, representing an entailment scenario. Following transformed files are written at wrtDir

    - Sentence pair transformed tsv file for entailment task

    For using this transform function, set ``transform_func`` : **snli_entailment_to_tsv** in transform file.

    Args:
        dataDir (:obj:`str`) : Path to the directory where the raw data files to be read are present..
        readFile (:obj:`str`) : This is the file which is currently being read and transformed by the function.
        wrtDir (:obj:`str`) : Path to the directory where to save the transformed tsv files.
        transParamDict (:obj:`dict`, defaults to :obj:`None`): Dictionary of function specific parameters. Not required for this transformation function.


    """

    mapping = {"contradiction" : 0, "neutral" : 0, "entailment" : 1}
    f = open(os.path.join(dataDir, readFile))
    w = open(os.path.join(wrtDir, 'entailment_{}.tsv'.format(readFile.lower().replace('.jsonl', ''))), 'w')
    print('Making data from file {}...'.format(readFile))
    posCnt = 0
    for i, line in enumerate(f):
        if i % 5000 == 0:
            print("Processing {} rows...".format(i))
        row = json.loads(line)
        if row["gold_label"] == '-':
            # means the annotation was not confident, so dropping sample
            continue
        label = mapping[row["gold_label"]]
        posCnt += label
        w.write("{}\t{}\t{}\t{}\n".format(row["pairID"], label, row["sentence1"], row["sentence2"]))
    
    print('File written at: ', os.path.join(wrtDir, 'entailment_{}.tsv'.format(readFile.lower().replace('.jsonl', ''))))
    
    print('total number of samples: {}'.format(i+1))
    print('number of positive samples: {}'.format(posCnt))
    print("number of negative samples: {}".format(i+1 - posCnt))

    f.close()
    w.close()

def generate_ngram_sequences(data, seq_len_right, seq_len_left):

        #train_data = pd.read_csv(train_data_file)
        #query_list= list(train_data['query'])
        sequence_dict = {}
        for sentence in data:
            word_list = sentence.split()
            len_seq = len(word_list)
            for seq_len in range(seq_len_right):
                i = 0
                while i + seq_len < len_seq:
                    if len_seq - i - seq_len - 1 >= seq_len_right:
                        right_seq = seq_len_right
                    else:
                        right_seq = len_seq - i - seq_len - 1

                    if i >= seq_len_left:
                        left_seq = seq_len_left
                    else:
                        left_seq = i

                    key = " ".join(word_list[i:i + seq_len + 1])

                    if sequence_dict.get(key, None) != None:
                        sequence_dict[key] = min(sequence_dict[key], left_seq + right_seq)
                    else:
                        sequence_dict[key] = left_seq + right_seq
                    i += 1
        return sequence_dict

def validate_sequences(sequence_dict, seq_len_right, seq_len_left):
    micro_sequences = []
    macro_sequences = {}

    for key in sequence_dict.keys():
        score = sequence_dict[key]

        if score < 1 and len(key.split()) <= seq_len_right:
            micro_sequences.append(key)
        else:
            macro_sequences[key] = score

    non_frag_sequences = []
    macro_sequences_copy = macro_sequences.copy()

    for sent in tqdm(micro_sequences, total = len(micro_sequences)):
        for key in macro_sequences.keys():
            if sent in key:
                non_frag_sequences.append(key)
                del macro_sequences_copy[key]

        macro_sequences = macro_sequences_copy.copy()

    for sent in non_frag_sequences:
        macro_sequences[sent] = 0

    for sent in micro_sequences:
        macro_sequences[sent] = 0

    return macro_sequences
    
def create_fragment_detection_tsv(dataDir, readFile, wrtDir, transParamDict, isTrainFile=False):

    """
    This function transforms data for fragment detection task (detecting whether a sentence is incomplete/fragment or not).
    It takes data in single sentence classification format and creates fragment samples from the sentences.
    In the transformed file, label 1 and 0 represent fragment and non-fragment sentence respectively.
    Following transformed files are written at wrtDir

    - Fragment transformed tsv file containing fragment/non-fragment sentences and labels


    For using this transform function, set ``transform_func`` : **create_fragment_detection_tsv** in transform file.
    Args:
        dataDir (:obj:`str`) : Path to the directory where the raw data files to be read are present..
        readFile (:obj:`str`) : This is the file which is currently being read and transformed by the function.
        wrtDir (:obj:`str`) : Path to the directory where to save the transformed tsv files.
        transParamDict (:obj:`dict`, defaults to :obj:`None`): Dictionary requiring the following parameters as key-value
            
            - ``data_frac`` (defaults to 0.2) : Fraction of data to consider for making fragments.
            - ``seq_len_right`` : (defaults to 3) : Right window length for making n-grams.
            - ``seq_len_left`` (defaults to 2) : Left window length for making n-grams.
            - ``sep`` (defaults to "\t") : column separator for input file.
            - ``query_col`` (defaults to 2) : column number containing sentences. Counting starts from 0.

    """

    transParamDict.setdefault("data_frac", 0.2)
    transParamDict.setdefault("seq_len_right", 3)
    transParamDict.setdefault("seq_len_left", 2)
    transParamDict.setdefault("sep", "\t")
    transParamDict.setdefault("query_col", 2)

    allDataDf = pd.read_csv(os.path.join(dataDir, readFile), sep=transParamDict["sep"], header=None)
    sampledDataDf = allDataDf.sample(frac = float(transParamDict['data_frac']), random_state=42)

    #2nd column is considered to have queries in dataframe, 0th uid, 1st label
    # making n-gram with left and right window
    seqDict = generate_ngram_sequences(data = list(sampledDataDf.iloc[:, int(transParamDict["query_col"])]),
                                    seq_len_right = transParamDict['seq_len_right'],
                                    seq_len_left = transParamDict['seq_len_left'])

    fragDict = validate_sequences(seqDict, seq_len_right = transParamDict['seq_len_right'],
                                seq_len_left = transParamDict['seq_len_left'])

    finalDf = pd.DataFrame({'uid' : [i for i in range(len(fragDict)+len(allDataDf))],
                            'label' : [1]*len(fragDict)+[0]*len(allDataDf),
                            'query' : list(fragDict.keys())+list(allDataDf.iloc[:, int(transParamDict["query_col"]) ]) })

    print('number of fragment samples : ', len(fragDict))
    print('number of non-fragment samples : ', len(allDataDf))
    # saving
    print('writing fragment file for {} at {}'.format(readFile, wrtDir))

    finalDf.to_csv(os.path.join(wrtDir, 'fragment_{}.tsv'.format(readFile.split('.')[0])), sep='\t',
                index=False, header=False)
    

def msmarco_answerability_detection_to_tsv(dataDir, readFile, wrtDir, transParamDict, isTrainFile=False):
    """
    This function transforms the MSMARCO triples data available at `triples <https://msmarco.blob.core.windows.net/msmarcoranking/triples.train.small.tar.gz>`_ 
    
    The data contains triplets where the first entry is the query, second one is the context passage from which the query can be
    answered (positive passage) , while the third entry is a context passage from which the query cannot be answered (negative passage).
    Data is transformed into sentence pair classification format, with query-positive context pair labeled as 1 (answerable) 
    and query-negative context pair labeled as 0 (non-answerable)
    
    Following transformed files are written at wrtDir

    - Sentence pair transformed downsampled file.
    - Sentence pair transformed train tsv file for answerability task
    - Sentence pair transformed dev tsv file for answerability task
    - Sentence pair transformed test tsv file for answerability task

    For using this transform function, set ``transform_func`` : **msmarco_answerability_detection_to_tsv** in transform file.

    Args:
        dataDir (:obj:`str`) : Path to the directory where the raw data files to be read are present..
        readFile (:obj:`str`) : This is the file which is currently being read and transformed by the function.
        wrtDir (:obj:`str`) : Path to the directory where to save the transformed tsv files.
        transParamDict (:obj:`dict`, defaults to :obj:`None`): Dictionary of function specific parameters. Not required for this transformation function.

            - ``data_frac`` (defaults to 0.01) : Fraction of data to keep in downsampling as the original data size is too large.
    """
    transParamDict.setdefault("data_frac", 0.01)
    sampleEvery = int(1/float(transParamDict["data_frac"]))
    startId = 0
    print('Making data from file {} ....'.format(readFile))
    rf = open(os.path.join(dataDir, readFile))
    sf = open(os.path.join(wrtDir, 'msmarco_triples_sampled.tsv'), 'w')
    
    # reading the big file line by line
    for i, row in enumerate(rf):
        # sampling
        if i % 100000 == 0:
            print("Processing {} rows...".format(i))
            
        if i % sampleEvery == 0:
            rowData = row.split('\t')
            posRowData = str(startId)+'\t'+str(1)+'\t'+ rowData[0]+'\t'+rowData[1]
            negRowData = str(startId+1)+'\t'+str(0)+'\t'+ rowData[0]+'\t'+rowData[2].rstrip('\n')

            #AN IMPORTANT POINT HERE IS TO STRIP THE row ending '\n' present after the negative 
            # passage, otherwise it will hamper the dataframe.

            #print(negRowData)
            # writing the positive and negative into new sampled data file
            sf.write(posRowData+'\n')
            sf.write(negRowData+'\n')

            #increasing id count
            startId += 2
    print('Total Number of rows in original data: ', i)
    print('Number of answerable samples in downsampled data: ', int(startId / 2))
    print('Number of non-answerable samples in downsampled data: ', int(startId / 2))
    print('Downsampled msmarco triples tsv saved at: {}'.format(os.path.join(wrtDir, 'msmarco_triples_sampled.tsv')))
    
    #making train, test, dev split
    sampledDf = pd.read_csv(os.path.join(wrtDir, 'msmarco_triples_sampled.tsv'), sep='\t', header=None)
    trainDf, testDf = train_test_split(sampledDf, shuffle=True, random_state=SEED,
                                          test_size=0.02)
    trainDf.to_csv(os.path.join(wrtDir, 'msmarco_answerability_train.tsv'), sep='\t', index=False, header=False)
    print('Train file written at: ', os.path.join(wrtDir, 'msmarco_answerability_train.tsv'))
    
    devDf, testDf = train_test_split(testDf, shuffle=True, random_state=SEED,
                                          test_size=0.5)
    devDf.to_csv(os.path.join(wrtDir, 'msmarco_answerability_dev.tsv'), sep='\t', index=False, header=False)
    print('Dev file written at: ', os.path.join(wrtDir, 'msmarco_answerability_dev.tsv'))
    
    devDf.to_csv(os.path.join(wrtDir, 'msmarco_answerability_test.tsv'), sep='\t', index=False, header=False)
    print('Test file written at: ', os.path.join(wrtDir, 'msmarco_answerability_test.tsv'))
    
def msmarco_query_type_to_tsv(dataDir, readFile, wrtDir, transParamDict, isTrainFile=False):

    """
    This function transforms the MSMARCO QnA data available at `MSMARCO_QnA <https://microsoft.github.io/msmarco/>`_ 
    for query-type detection task (given a query sentence, detect what type of answer is expected). Queries are divided
    into 5 query types - NUMERIC, LOCATION, ENTITY, DESCRIPTION, PERSON. The function transforms the json data
    to standard single sentence classification type tsv data. Following transformed files are written at wrtDir

    - Query type transformed tsv data file.
    - Query type label map joblib file.

    For using this transform function, set ``transform_func`` : **msmarco_query_type_to_tsv** in transform file.

    Args:
        dataDir (:obj:`str`) : Path to the directory where the raw data files to be read are present..
        readFile (:obj:`str`) : This is the file which is currently being read and transformed by the function.
        wrtDir (:obj:`str`) : Path to the directory where to save the transformed tsv files.
        transParamDict (:obj:`dict`, defaults to :obj:`None`): Dictionary requiring the following parameters as key-value
            
            - ``data_frac`` (defaults to 0.05) : Fraction of data to consider for downsampling.

    """
    
    transParamDict.setdefault("data_frac", 0.05)

    print('Reading data file {}'.format(readFile))
    allDataDf = pd.read_json(os.path.join(dataDir, readFile))
    #allDataDf = allDataDf.drop(['answers', 'passages', 'wellFormedAnswers'], axis = 1)

    #downsampling
    print('number of samples before downsampling ', len(allDataDf))
    _, dfKeep = train_test_split(allDataDf, test_size = float(transParamDict["data_frac"]),
                                        shuffle=True, random_state=SEED)
    dfKeep = dfKeep[['query_id', 'query_type', 'query' ]]
    #saving
    print('number of samples in final data : ', len(dfKeep))
    print('writing for file {} at {}'.format(readFile, wrtDir))
    dfKeep.to_csv(os.path.join(wrtDir, 'querytype_{}.tsv'.format(readFile.lower().replace('.json', ''))), sep='\t',
                            index=False, header=False)
    if isTrainFile:
        allClasses = dfKeep['query_type'].unique()
        labelMap = {lab : i for i, lab in enumerate(allClasses)}
        labelMapPath = os.path.join(wrtDir, 'querytype_{}_label_map.joblib'.format(readFile.lower().replace('.json', '')))
        joblib.dump(labelMap, labelMapPath)
        print('Created label map file at', labelMapPath)

        
def imdb_sentiment_data_to_tsv(dataDir, readFile, wrtDir, transParamDict, isTrainFile=False):
    
    """
    This function transforms the IMDb moview review data available at `IMDb <https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews/data>`_ after accepting the terms.
    
    The data is having total 50k samples labeled as `positive` or `negative`. The reviews have some html tags which are cleaned
    by this function. Following transformed files are written at wrtDir


    - IMDb train transformed tsv file for sentiment analysis task
    - IMDb test transformed tsv file for sentiment analysis task
    
    For using this transform function, set ``transform_func`` : **imdb_sentiment_data_to_tsv** in transform file.

    Args:
        dataDir (:obj:`str`) : Path to the directory where the raw data files to be read are present..
        readFile (:obj:`str`) : This is the file which is currently being read and transformed by the function.
        wrtDir (:obj:`str`) : Path to the directory where to save the transformed tsv files.
        transParamDict (:obj:`dict`, defaults to :obj:`None`): Dictionary of function specific parameters. Not required for this transformation function.
        
            - ``train_frac`` (defaults to 0.05) : Fraction of data to consider for train/test split.
    """
    transParamDict.setdefault("train_frac", 0.9)
    print('Making data from file ', readFile)
    df = pd.read_csv(os.path.join(dataDir, readFile))
    
    #cleaning review text
    tt = re.compile('\t')
    cleanr = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
    
    df['review'] = [re.sub(tt, ' ', review) for review in df['review'] ]
    df['review'] = [re.sub(cleanr, ' ', review) for review in df['review'] ]
    
    df['uid'] = [str(i) for i in range(len(df))]
    df = df[['uid', 'sentiment', 'review']]
    # train test 
    dfTrain, dfTest = train_test_split(df, shuffle=False, test_size=1-float(transParamDict["train_frac"]),
                                      random_state=SEED)
    
    print('Number of samples in train: ', len(dfTrain))
    print('Number of samples in test: ', len(dfTest))
    
    #writing train file
    dfTrain.to_csv(os.path.join(wrtDir, 'imdb_sentiment_train.tsv'), sep='\t',index=False,header=False)
    print('Train file written at: ', os.path.join(wrtDir, 'imdb_sentiment_train.tsv'))
    
    #writing test file
    dfTest.to_csv(os.path.join(wrtDir, 'imdb_sentiment_test.tsv'), sep='\t',index=False,header=False)
    print('Test file written at: ', os.path.join(wrtDir, 'imdb_sentiment_test.tsv'))
    
def qqp_query_similarity_to_tsv(dataDir, readFile, wrtDir, transParamDict, isTrainFile=False):
    
    """
    This function transforms the QQP (Quora Question Pairs) query similarity data available at `QQP <http://qim.fs.quoracdn.net/quora_duplicate_questions.tsv>`_ 
    
    If the second query is similar to first query in a query-pair, the pair is labeled -> 1 and if not, then 
    labeled -> 0.
    Following transformed files are written at wrtDir

    - Sentence pair transformed train tsv file for query similarity task
    - Sentence pair transformed dev tsv file for query similarity task
    - Sentence pair transformed test tsv file for query similarity task

    For using this transform function, set ``transform_func`` : **snli_entailment_to_tsv** in transform file.

    Args:
        dataDir (:obj:`str`) : Path to the directory where the raw data files to be read are present..
        readFile (:obj:`str`) : This is the file which is currently being read and transformed by the function.
        wrtDir (:obj:`str`) : Path to the directory where to save the transformed tsv files.
        transParamDict (:obj:`dict`, defaults to :obj:`None`): Dictionary of function specific parameters. Not required for this transformation function.

            - ``train_frac`` (defaults to 0.8) : Fraction of data to consider for training. Remaining will be divided into dev and test.
    """
    
    transParamDict.setdefault("train_frac", 0.8)
    
    print("Making data from file {} ...".format(readFile))
    
    readDf = pd.read_csv(os.path.join(dataDir, readFile), sep="\t")
    wrtDf = readDf[['id', 'is_duplicate', 'question1', 'question2']]
    
    # writing train file
    trainDf, testDf = train_test_split(wrtDf, shuffle=True, random_state=SEED,
                                      test_size = 1 - float(transParamDict["train_frac"]))
    trainDf.to_csv(os.path.join(wrtDir, 'qqp_query_similarity_train.tsv'), sep="\t",
                  index=False, header=False)
    print('Train file saved at: {}'.format(os.path.join(wrtDir, 'qqp_query_similarity_train.tsv')))
    
    #writing dev file
    devDf, testDf = train_test_split(testDf, shuffle=True, random_state=SEED,
                                    test_size  = 0.5)
    devDf.to_csv(os.path.join(wrtDir, 'qqp_query_similarity_dev.tsv'), sep="\t",
                  index=False, header=False)
    print('Dev file saved at: {}'.format(os.path.join(wrtDir, 'qqp_query_similarity_dev.tsv')))
    
    #writing test file
    testDf.to_csv(os.path.join(wrtDir, 'qqp_query_similarity_test.tsv'), sep="\t",
                  index=False, header=False)   
    print('Test file saved at: {}'.format(os.path.join(wrtDir, 'qqp_query_similarity_test.tsv')))
    

    
def query_correctness_to_tsv(dataDir, readFile, wrtDir, transParamDict, isTrainFile=False):

    """

    - Query correctness transformed file

    For using this transform function, set ``transform_func`` : **query_correctness_to_tsv** in transform file.

    Args:
        dataDir (:obj:`str`) : Path to the directory where the raw data files to be read are present..
        readFile (:obj:`str`) : This is the file which is currently being read and transformed by the function.
        wrtDir (:obj:`str`) : Path to the directory where to save the transformed tsv files.
        transParamDict (:obj:`dict`, defaults to :obj:`None`): Dictionary of function specific parameters. Not required for this transformation function.

    """
    print('Making data from file {}'.format(readFile))
    df = pd.read_csv(os.path.join(dataDir, readFile), sep='\t', header=None, names = ['query', 'label'])
    
    # we consider anything above 0.6 as structured query (3 or more annotations as structured), and others as non-structured
    
    #df['label'] = [str(lab) for lab in df['label']]
    df['label'] = [int(lab>=0.6)for lab in df['label']]
    
    data = [ [str(i), str(row['label']), row['query'] ] for i, row in df.iterrows()]
    
    wrtDf = pd.DataFrame(data, columns = ['uid', 'label', 'query'])
    
    #writing
    wrtDf.to_csv(os.path.join(wrtDir, 'query_correctness_{}'.format(readFile)), sep="\t", index=False, header=False)
    print('File saved at: ', os.path.join(wrtDir, 'query_correctness_{}'.format(readFile)))
    
def clinc_out_of_scope_to_tsv(dataDir, readFile, wrtDir, transParamDict, isTrainFile=False):
    
    """

    For using this transform function, set ``transform_func`` : **clinc_out_of_scope_to_tsv** in transform file.

    Args:
        dataDir (:obj:`str`) : Path to the directory where the raw data files to be read are present..
        readFile (:obj:`str`) : This is the file which is currently being read and transformed by the function.
        wrtDir (:obj:`str`) : Path to the directory where to save the transformed tsv files.
        transParamDict (:obj:`dict`, defaults to :obj:`None`): Dictionary requiring the following parameters as key-value
            
            - ``samples_per_intent_train`` (defaults to 7) : Number of in-scope samples per intent to consider, as this data has imbalance for inscope and outscope

    """
    transParamDict.setdefault("samples_per_intent_train", 7)
    
    print("Making data from file {} ...".format(readFile))
    raw = json.load(open(os.path.join(dataDir, readFile)))
    
    print('Num of train samples in-scope: ', len(raw['train']))
    inScopeTrain = defaultdict(list)
    for sentence, intent in raw['train']:
        inScopeTrain[intent].append(sentence)

    #sampling 
    inscopeSampledTrain = []
    numSamplesPerInt = 7
    random.seed(SEED)
    for intent in inScopeTrain:
        inscopeSampledTrain += random.sample(inScopeTrain[intent], int(transParamDict["samples_per_intent_train"]))
        
    print('Num of sampled train samples in-scope: ', len(inscopeSampledTrain))
    #out of scope train
    outscopeTrain = [sample[0] for sample in raw['oos_train']]
    print('Num of train out-scope samples: ', len(outscopeTrain))
    
    #train data
    allTrain = inscopeSampledTrain + outscopeTrain
    allTrainLabels = [1]*len(inscopeSampledTrain) + [0]*len(outscopeTrain)
    
    #writing train data file
    trainF = open(os.path.join(wrtDir, 'clinc_outofscope_train.tsv'), 'w')
    for uid, (samp, lab) in enumerate(zip(allTrain, allTrainLabels)):
        trainF.write("{}\t{}\t{}\n".format(uid, lab, samp))
    print('Train file written at: ', os.path.join(wrtDir, 'clinc_outofscope_train.tsv'))
    trainF.close()

    #making dev set
    inscopeDev = [sample[0] for sample in raw['val']]
    outscopeDev = [sample[0] for sample in raw['oos_val']]
    print('Num of val out-scope samples: ', len(outscopeDev))
    print('Num of val in-scope samples: ', len(inscopeDev))

    #allDev = inscopeDev + outscopeDev
    allDev = outscopeDev
    #allDevLabels = [1]*inscopeDev + [0]*outscopeDev
    allDevLabels = [0]*len(outscopeDev)
    
    #writing dev data file
    devF = open(os.path.join(wrtDir, 'clinc_outofscope_dev.tsv'), 'w')
    for uid, (samp, lab) in enumerate(zip(allDev, allDevLabels)):
        devF.write("{}\t{}\t{}\n".format(uid, lab, samp))
    print('Dev file written at: ', os.path.join(wrtDir, 'clinc_outofscope_dev.tsv'))
    devF.close()

    #making test set 
    inscopeTest = [sample[0] for sample in raw['test']]
    outscopeTest = [sample[0] for sample in raw['oos_test']]
    print('Num of test out-scope samples: ', len(outscopeTest))
    print('Num of test in-scope samples: ', len(inscopeTest))
    
    allTest = inscopeTest + outscopeTest
    allTestLabels = [1]*len(inscopeTest) + [0]*len(outscopeTest)
    
    #writing test data file
    testF = open(os.path.join(wrtDir, 'clinc_outofscope_test.tsv'), 'w')
    for uid, (samp, lab) in enumerate(zip(allTest, allTestLabels)):
        testF.write("{}\t{}\t{}\n".format(uid, lab, samp))
    print('Test file written at: ', os.path.join(wrtDir, 'clinc_outofscope_test.tsv'))
    testF.close()
    
