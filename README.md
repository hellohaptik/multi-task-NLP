
<h1 align="center">multi-task-NLP</h1>  
<p align="center">
    <a href='https://multi-task-nlp.readthedocs.io/en/latest/?badge=latest'>
        <img src='https://readthedocs.org/projects/multi-task-nlp/badge/?version=latest' alt='Documentation Status' />
    </a>
    <a href="https://github.com/hellohaptik/multi-task-NLP/blob/master/LICENSE">
        <img src="https://img.shields.io/github/license/hellohaptik/multi-task-NLP">
    </a>
    <a href="https://github.com/hellohaptik/multi-task-NLP/graphs/contributors">
        <img src="https://img.shields.io/badge/contributors-3-yellow">
    </a>
    <a href="https://github.com/hellohaptik/multi-task-NLP/issues">
        <img src="https://img.shields.io/github/issues/hellohaptik/multi-task-NLP?color=orange">
    </a>
</p>

<p align="center">
    <img src="docs/source/multi_task.png" width="500", height="550">
</p> 

multi_task_NLP is a utility toolkit enabling NLP developers to easily train and infer a single model for multiple tasks.
We support various data formats for majority of NLU tasks and multiple transformer-based encoders (eg. BERT, Distil-BERT, ALBERT, RoBERTa, XLNET etc.)

For complete documentation for this library, please refer to [documentation](https://multi-task-nlp.readthedocs.io/en/latest/)

## What is multi_task_NLP about?

Any conversational AI system involves building multiple components to perform various tasks and a pipeline to stitch all components together.
Provided the recent effectiveness of transformer-based models in NLP, it’s very common to build a transformer-based model to solve your use case.
But having multiple such models running together for a conversational AI system can lead to expensive resource consumption, increased latencies for predictions and make the system difficult to manage.
This poses a real challenge for anyone who wants to build a conversational AI system in a simplistic way.

multi_task_NLP gives you the capability to define multiple tasks together and train a single model which simultaneously learns on all defined tasks.
This means one can perform multiple tasks with latency and resource consumption equivalent to a single task.

## Installation

To use multi-task-NLP, you can clone the repository into the desired location on your system
with the following terminal command.

```console
$ cd /desired/location/
$ git clone https://github.com/hellohaptik/multi-task-NLP.git
$ cd multi-task-NLP
$ pip install -r requirements.txt 
```

NOTE:- The library is built and tested using ``Python 3.7.3``. It is recommended to install the requirements in a virtual environment.
 
## Quickstart Guide

A quick guide to show how a model can be trained for single/multiple NLU tasks in just 3 simple steps
and with **no requirement to code!!**

Follow these 3 simple steps to train your multi-task model!

### Step 1 - Define your task file

Task file is a YAML format file where you can add all your tasks for which you want to train a multi-task model.

```yaml

TaskA:
    model_type: BERT
    config_name: bert-base-uncased
    dropout_prob: 0.05
    label_map_or_file:
    - label1
    - label2
    - label3
    metrics:
    - accuracy
    loss_type: CrossEntropyLoss
    task_type: SingleSenClassification
    file_names:
    - taskA_train.tsv
    - taskA_dev.tsv
    - taskA_test.tsv

TaskB:
    model_type: BERT
    config_name: bert-base-uncased
    dropout_prob: 0.3
    label_map_or_file: data/taskB_train_label_map.joblib
    metrics:
    - seq_f1
    - seq_precision
    - seq_recall
    loss_type: NERLoss
    task_type: NER
    file_names:
    - taskB_train.tsv
    - taskB_dev.tsv
    - taskB_test.tsv
```
For knowing about the task file parameters to make your task file, [task file parameters](https://multi-task-nlp.readthedocs.io/en/latest/define_multi_task_model.html#task-file-parameters).

### Step 2 - Run data preparation

After defining the task file, run the following command to prepare the data.

```console
$ python data_preparation.py \ 
    --task_file 'sample_task_file.yml' \
    --data_dir 'data' \
    --max_seq_len 50
```

For knowing about the ``data_preparation.py`` script and its arguments, refer [running data preparation](https://multi-task-nlp.readthedocs.io/en/latest/training.html#running-data-preparation).

### Step 3 - Run train

Finally you can start your training using the following command.

```console
$ python train.py \
    --data_dir 'data/bert-base-uncased_prepared_data' \
    --task_file 'sample_task_file.yml' \
    --out_dir 'sample_out' \
    --epochs 5 \
    --train_batch_size 4 \
    --eval_batch_size 8 \
    --grad_accumulation_steps 2 \
    --log_per_updates 25 \
    --save_per_updates 1000 \
    --eval_while_train True \
    --test_while_train True \
    --max_seq_len 50 \
    --silent True 
```

For knowing about the ``train.py`` script and its arguments, refer [running train](https://multi-task-nlp.readthedocs.io/en/latest/training.html#running-train)


## How to Infer?

Once you have a multi-task model trained on your tasks, we provide a convenient and easy way to use it for getting
predictions on samples through the **inference pipeline**.

For running inference on samples using a trained model for say TaskA, TaskB and TaskC,
you can import ``InferPipeline`` class and load the corresponding multi-task model by making an object of this class.

```python
>>> from infer_pipeline import inferPipeline
>>> pipe = inferPipeline(modelPath = 'sample_out_dir/multi_task_model.pt', maxSeqLen = 50)
```

``infer`` function can be called to get the predictions for input samples
for the mentioned tasks.

```python
>>> samples = [ ['sample_sentence_1'], ['sample_sentence_2'] ]
>>> tasks = ['TaskA', 'TaskB']
>>> pipe.infer(samples, tasks)
```

For knowing about the ``infer_pipeline``, refer [infer](https://multi-task-nlp.readthedocs.io/en/latest/infering.html).

## Examples

Here you can find various conversational AI tasks as examples and can train multi-task models
in simple steps mentioned in the notebooks.

### Example-1 Intent detection, NER, Fragment detection
*(Setup : Multi-task , Task type : Multiple)*

**Intent Detection(Task type : Single sentence classification)**

```
 Query: I need a reservation for a bar in bangladesh on feb the 11th 2032
 
 Intent: BookRestaurant
```

**NER (Task type :sequence labelling)**

```
Query: ['book', 'a', 'spot', 'for', 'ten', 'at', 'a', 'top-rated', 'caucasian', 'restaurant', 'not', 'far', 'from', 'selmer']

NER tags: ['O', 'O', 'O', 'O', 'B-party_size_number', 'O', 'O', 'B-sort', 'B-cuisine', 'B-restaurant_type', 'B-spatial_relation', 'I-spatial_relation', 'O', 'B-city']
```

**Fragment Detection (Task type : single sentence classification)**

```
Query: a reservation for

Label: fragment
```

**Notebook** :- [intent_ner_fragment](https://github.com/hellohaptik/multi-task-NLP/blob/master/examples/intent_ner_fragment/intent_ner_fragment.ipynb)

**Transform file** :- [transform_file_snips](https://github.com/hellohaptik/multi-task-NLP/blob/master/examples/intent_ner_fragment/transform_file_snips.yml)

**Tasks file** :-  [tasks_file_snips](https://github.com/hellohaptik/multi-task-NLP/blob/master/examples/intent_ner_fragment/tasks_file_snips.yml)

### Example-2 Entailment detection
*(Setup : single-task , Task type : sentence pair classification)*

```
Query1: An old man with a package poses in front of an advertisement.

Query2: A man poses in front of an ad.

Label: entailment

Query1: An old man with a package poses in front of an advertisement.

Query2: A man poses in front of an ad for beer.

Label: non-entailment

```

**Notebook** :- [entailment_snli](https://github.com/hellohaptik/multi-task-NLP/tree/master/examples/entailment_detection/entailment_snli.ipynb)

**Transform file** :- [transform_file_snli](https://github.com/hellohaptik/multi-task-NLP/tree/master/examples/entailment_detection/transform_file_snli.yml)

**Tasks file** :- [tasks_file_snli](https://github.com/hellohaptik/multi-task-NLP/tree/master/examples/entailment_detection/tasks_file_snli.yml)



### Example-3 Answerability detection
*(Setup : single-task , Task type : sentence pair classification)*

```
Query: how much money did evander holyfield make

Context: Evander Holyfield Net Worth. How much is Evander Holyfield Worth? Evander Holyfield Net Worth: Evander Holyfield is a retired American professional boxer who has a net worth of $500 thousand. A professional boxer, Evander Holyfield has fought at the Heavyweight, Cruiserweight, and Light-Heavyweight Divisions, and won a Bronze medal a the 1984 Olympic Games.

Label: answerable
```
**Notebook** :- [answerability_detection_msmarco](https://github.com/hellohaptik/multi-task-NLP/tree/master/examples/answerability_detection/answerability_detection_msmarco.ipynb)

**Transform file** :- [transform_file_answerability](https://github.com/hellohaptik/multi-task-NLP/tree/master/examples/answerability_detection/transform_file_answerability.yml)

**Tasks file** :- [tasks_file_answerability](https://github.com/hellohaptik/multi-task-NLP/tree/master/examples/answerability_detection/tasks_file_answerability.yml)

### Example-4 Query type detection
*(Setup : single-task , Task type : single sentence classification)*

```
Query: what's the distance between destin florida and birmingham alabama?

Label: NUMERIC

Query: who is suing scott wolter

Label: PERSON

```

**Notebook** :- [query_type_detection](https://github.com/hellohaptik/multi-task-NLP/blob/master/examples/query_type_detection/query_type_detection.ipynb)

**Transform file** :- [transform_file_querytype](https://github.com/hellohaptik/multi-task-NLP/blob/master/examples/query_type_detection/transform_file_querytype.yml)

**Tasks file** :- [tasks_file_querytype](https://github.com/hellohaptik/multi-task-NLP/blob/master/examples/query_type_detection/tasks_file_querytype.yml)

### Example-5 POS tagging, NER tagging
*(Setup : Multi-task , Task type : sequence labelling)*

```
Query: ['Despite', 'winning', 'the', 'Asian', 'Games', 'title', 'two', 'years', 'ago', ',', 'Uzbekistan', 'are', 'in', 'the', 'finals', 'as', 'outsiders', '.']

NER tags: ['O', 'O', 'O', 'I-MISC', 'I-MISC', 'O', 'O', 'O', 'O', 'O', 'I-LOC', 'O', 'O', 'O', 'O', 'O', 'O', 'O']

POS tags: ['I-PP', 'I-VP', 'I-NP', 'I-NP', 'I-NP', 'I-NP', 'B-NP', 'I-NP', 'I-ADVP', 'O', 'I-NP', 'I-VP', 'I-PP', 'I-NP', 'I-NP', 'I-SBAR', 'I-NP', 'O']

```

**Notebook** :- [ner_pos_tagging_conll](https://github.com/hellohaptik/multi-task-NLP/blob/master/examples/ner_pos_tagging/ner_pos_tagging_conll.ipynb)

**Transform file** :- [transform_file_conll](https://github.com/hellohaptik/multi-task-NLP/blob/master/examples/ner_pos_tagging/transform_file_conll.yml)

**Tasks file** :- [tasks_file_conll](https://github.com/hellohaptik/multi-task-NLP/blob/master/examples/ner_pos_tagging/tasks_file_conll.yml)

## Example-6 Query correctness
*(Setup : single-task , Task type : single sentence classification)*

```

Query: What places have the oligarchy government ?

Label: well-formed

Query: What day of Diwali in 1980 ?

Label: not well-formed

```

**Notebook** :- [query_correctness](https://github.com/hellohaptik/multi-task-NLP/blob/master/examples/query_correctness/query_correctness.ipynb)

**Transform file** :- [transform_file_query_correctness](https://github.com/hellohaptik/multi-task-NLP/blob/master/examples/query_correctness/transform_file_query_correctness.yml)

**Tasks file** :- [tasks_file_query_correctness](https://github.com/hellohaptik/multi-task-NLP/blob/master/examples/query_correctness/tasks_file_query_correctness.yml)


## Example-7 Query similarity
*(Setup : single-task , Task type : single sentence classification)*

```

Query1: What is the most used word in Malayalam?

Query2: What is meaning of the Malayalam word ""thumbatthu""?

Label: not similar

Query1: Which is the best compliment you have ever received?

Query2: What's the best compliment you've got?

Label: similar

```
**Notebook** :- [query_similarity](https://github.com/hellohaptik/multi-task-NLP/blob/master/examples/query_pair_similarity/query_similarity_qqp.ipynb)

**Transform file** :- [transform_file_qqp](https://github.com/hellohaptik/multi-task-NLP/blob/master/examples/query_pair_similarity/transform_file_qqp.yml)

**Tasks file** :- [tasks_file_qqp](https://github.com/hellohaptik/multi-task-NLP/blob/master/examples/query_pair_similarity/tasks_file_query_qqp.yml)

## Example-8 Sentiment Analysis
*(Setup : single-task , Task type : single sentence classification)*

```

Review: What I enjoyed most in this film was the scenery of Corfu, being Greek I adore my country and I liked the flattering director's point of view. Based on a true story during the years when Greece was struggling to stand on her own two feet through war, Nazis and hardship. An Italian soldier and a Greek girl fall in love but the times are hard and they have a lot of sacrifices to make. Nicholas Cage looking great in a uniform gives a passionate account of this unfulfilled (in the beginning) love. I adored Christian Bale playing Mandras the heroine's husband-to-be, he looks very very good as a Greek, his personality matched the one of the Greek patriot! A true fighter in there, or what! One of the movies I would like to buy and keep it in my collection...for ever!

Label: positive

```

**Notebook** :- [IMDb_sentiment_analysis](https://github.com/hellohaptik/multi-task-NLP/blob/master/examples/sentiment_analysis/IMDb_sentiment_analysis.ipynb)

**Transform file** :- [transform_file_imdb](https://github.com/hellohaptik/multi-task-NLP/blob/master/examples/sentiment_analysis/transform_file_imdb.yml)

**Tasks file** :- [tasks_file_imdb](https://github.com/hellohaptik/multi-task-NLP/blob/master/examples/sentiment_analysis/tasks_file_query_imdb.yml)


