Task Formats
============

- To standardize the data input files, all the tasks require ``tsv`` format files as input data files.
- The tsv data files shouldn’t contain any headers. Detailed tsv formats required for specific task types are mentioned in following subsection.

Task types
----------
Input data formats for different NLI tasks can vary from task to task. We support the following three task types.
Majority of the NLI tasks can be modeled using one of these task types.

- ``SingleSenClassification``: This task type is to be used for classification of single sentences. The data files needs to have following columns separated by **"\\t"** 
  in the order as mentioned below.

  1. **Unique id** :- an id to uniquely identify each row/sample.
  2. **Label** :- label for the sentence. Labels can be numeric or strings. In case labels are strings, label mapping needs to be provided.
  3. **Sentence** :- The sentence which needs to be classified.

- ``SentencePairClassification``: This task type is to be used for classification of sentence pairs (two sentences). The data files needs to have following columns separated by **"\\t"** 
  in the order as mentioned below.

  1. **Unique id** :- an id to uniquely identify each row/sample.
  2. **Label** :- label for the sentence. Labels can be numeric or strings. In case labels are strings, label mapping needs to be provided.
  3. **SentenceA** :-  First sentence of the sentence pair.
  4. **SentenceB** :- Second sentence of the sentence pair.

- ``NER`` : This task type is to be used for sequence labelling tasks like Named Entity Recognition , entity mention detection, keyphrase extraction etc. The data files need to have following columns separated by **"\\t"** in the order as mentioned below.

  1. **Unique id** :- an id to uniquely identify each row/sample. 
  2. **Label** :- List of tags for words in sentence.
  3. **Sentence** :- List of words in sentence.



NOTE:- The tsv data files must not have the header names.




