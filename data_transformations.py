'''
For transforming the raw data in different formats to standard tsv format 
to be consumed for multi-task
'''

import joblib
import argparse
import os
import json
from statistics import median

def snips_intent_ner_tsv(readPath, nerWrtPath, isTrainFile = False,
                        hasIntent = True, intWrtPath = None):
    f = open(readPath)
    w = open(nerWrtPath, 'w')
    if hasIntent:
        assert intWrtPath != None, "intent write path wasn't given for hasIntent True"
        intW = open(intWrtPath, 'w')

    labelMapNer = {}
    labelMapInt = {}

    sentence = []
    label = []
    uid = 0
    print("Making data from file {} ...".format(readPath))
    for i, line in enumerate(f):
        if i%5000 == 0:
            print("Processing {} rows...".format(i))

        line = line.strip(' ') #don't use strip empty as it also removes \n
        wordSplit = line.rstrip('\n').split(' ')
        if line == '\n':

            w.write("{}\t{}\t{}\n".format(uid, label, sentence))
            #print("len of sentence :", len(sentence))
            sentence = []
            label = []
            uid += 1

        elif hasIntent == True and len(wordSplit) == 1:
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

    print("NER File Written at {}".format(nerWrtPath))
    if hasIntent:
        print("Intent File Written at {}".format(intWrtPath))
    #writing label map
    if labelMapNer != {} and isTrainFile:
        print("Created NER label map from train file {}".format(readPath))
        print(labelMapNer)
        labelMapNerPath = "{}_label_map.joblib".format(nerWrtPath.split('.')[0])
        joblib.dump(labelMapNer, labelMapNerPath)
        print("label Map NER written at {}".format(labelMapNerPath))

    if labelMapInt != {} and isTrainFile:
        print("Created Intent label map from train file {}".format(readPath))
        print(labelMapInt)
        labelMapIntPath = "{}_label_map.joblib".format(intWrtPath.split('.')[0])
        joblib.dump(labelMapInt, labelMapIntPath)
        print("label Map Intent written at {}".format(labelMapIntPath))

    f.close()
    w.close()

def coNLL_ner_pos_tsv(readPath, nerWrtPath, isTrainFile = False,
                    posWrtPath = None):
    f = open(readPath)
    nerW = open(nerWrtPath, 'w')
    posW = open(posWrtPath, 'w')

    labelMapNer = {}
    labelMapPos = {}

    sentence = []
    senLens = []
    labelNer = []
    labelPos = []
    uid = 0
    print("Making data from file {} ...".format(readPath))
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
        labelPos.append(wordSplit[1])
        labelNer.append(wordSplit[-1])
        if isTrainFile:
            if wordSplit[-1] not in labelMapNer:
                # ONLY TRAIN FILE SHOULD BE USED TO CREATE LABEL MAP FILE.
                labelMapNer[wordSplit[-1]] = len(labelMapNer)
            if wordSplit[1] not in labelMapPos:
                labelMapPos[wordSplit[1]] = len(labelMapPos)
    
    print("NER File Written at {}".format(nerWrtPath))
    print("POS File Written at {}".format(posWrtPath))
    #writing label map
    if labelMapNer != {} and isTrainFile:
        print("Created NER label map from train file {}".format(readPath))
        print(labelMapNer)
        labelMapNerPath = "{}_label_map.joblib".format(nerWrtPath.split('.')[0])
        joblib.dump(labelMapNer, labelMapNerPath)
        print("label Map NER written at {}".format(labelMapNerPath))

    if labelMapPos != {} and isTrainFile:
        print("Created POS label map from train file {}".format(readPath))
        print(labelMapPos)
        labelMapPosPath = "{}_label_map.joblib".format(posWrtPath.split('.')[0])
        joblib.dump(labelMapPos, labelMapPosPath)
        print("label Map POS written at {}".format(labelMapPosPath))

    f.close()
    nerW.close()
    posW.close()

    print('Max len of sentence: ', max(senLens))
    print('Mean len of sentences: ', sum(senLens)/len(senLens))
    print('Median len of sentences: ', median(senLens))

def snli_tsv(readPath, wrtPath):
    mapping = {"contradiction" : 0, "neutral" : 0, "entailment" : 1}
    f = open(readPath)
    w = open(wrtPath, 'w')
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
    
    print('total number of samples: {}'.format(i+1))
    print('number of positive samples: {}'.format(posCnt))
    print("number of negative samples: {}".format(i+1 - posCnt))

    f.close()
    w.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_data_dir', type=str, required=True,
                        help="Directory where data files to be transformed present")
    parser.add_argument('--out_dir', type=str, required=True,
                        help="Directory where transformed tsv file to be written")
    args = parser.parse_args()

    assert os.path.exists(args.raw_data_dir), "raw data dir doesn't exist"
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    
    for file in os.listdir(args.raw_data_dir):
        if not file.startswith('.'):
            readP = os.path.join(args.raw_data_dir, file)
            #nerWrtP = os.path.join(args.out_dir, "ner_{}.tsv".format(file.split('.')[0]))
            #posWrtP = os.path.join(args.out_dir, "pos_{}.tsv".format(file.split('.')[0]))
            wrtP = os.path.join(args.out_dir, "entailment_{}.tsv".format(file.split('.')[0]))
            isTrain = False
            if "train" in file:
                isTrain = True

            #snips_intent_ner_tsv(readP, nerWrtPath=nerWrtP, intWrtPath=intWrtP,
                                #isTrainFile=isTrain, hasIntent = True)
            #coNLL_ner_pos_tsv(readP, nerWrtPath=nerWrtP, posWrtPath=posWrtP,
                                #isTrainFile=isTrain)
            snli_tsv(readP, wrtP)

if __name__ == "__main__":
    main()