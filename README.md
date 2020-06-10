
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
We support various data formats for majority of NLI tasks and multiple transformer-based encoders (eg. BERT, Distil-BERT, ALBERT, RoBERTa, XLNET etc.)

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

A quick guide to show how a single model can be trained for multiple NLI tasks in just 3 simple steps
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

**Tasks Description**

``Intent Detection`` :- This is a single sentence classification task where an `intent` specifies which class the data sample belongs to. 

``NER`` :- This is a Named Entity Recognition/ Sequence Labelling/ Slot filling task where individual words of the sentence are tagged with an entity label it belongs to. The words which don't belong to any entity label are simply labeled as "O". 

``Fragment Detection`` :- This is modeled as a single sentence classification task which detects whether a sentence is incomplete (fragment) or not (non-fragment).

**Conversational Utility** :-  Intent detection is one of the fundamental components for conversational system as it gives a broad understand of the category/domain the sentence/query belongs to.

NER helps in extracting values for required entities (eg. location, date-time) from query.

Fragment detection is a very useful piece in conversational system as knowing if a query/sentence is incomplete can aid in discarding bad queries beforehand.

**Data** :- In this example, we are using the [SNIPS](https://snips-nlu.readthedocs.io/en/latest/dataset.html) data for intent and entity detection. For the sake of simplicity, we provide 
the data in simpler form under ``snips_data`` directory taken from [here](https://github.com/LeePleased/StackPropagation-SLU/tree/master/data/snips>).

**Transform file** :- [transform_file_snips](https://github.com/hellohaptik/multi-task-NLP/blob/master/examples/intent_ner_fragment/transform_file_snips.yml)

**Tasks file** :-  [tasks_file_snips](https://github.com/hellohaptik/multi-task-NLP/blob/master/examples/intent_ner_fragment/tasks_file_snips.yml)

**Notebook** :- [intent_ner_fragment](https://github.com/hellohaptik/multi-task-NLP/blob/master/examples/intent_ner_fragment/intent_ner_fragment.ipynb)

### Example-2 Entailment detection

**Tasks Description**

``Entailment`` :- This is a sentence pair classification task which determines whether the second sentence in a sample can be inferred from the first.

**Conversational Utility** :-  In conversational AI context, this task can be seen as determining whether the second sentence is similar to first or not.
Additionally, the probability score can also be used as a similarity score between the sentences. 

**Data** :- In this example, we are using the [SNLI](https://nlp.stanford.edu/projects/snli) data which is having sentence pairs and labels.

**Transform file** :- [transform_file_snli](https://github.com/hellohaptik/multi-task-NLP/tree/master/examples/entailment_detection/transform_file_snli.yml)

**Tasks file** :- [tasks_file_snli](https://github.com/hellohaptik/multi-task-NLP/tree/master/examples/entailment_detection/tasks_file_snli.yml)

**Notebook** :- [entailment_snli](https://github.com/hellohaptik/multi-task-NLP/tree/master/examples/entailment_detection/entailment_snli.ipynb)

### Example-3 Answerability detection

**Tasks Description**

``answerability`` :- This is modeled as a sentence pair classification task where the first sentence is a query and second sentence is a context passage.
The objective of this task is to determine whether the query can be answered from the context passage or not.

**Conversational Utility** :- This can be a useful component for building a question-answering/ machine comprehension based system.
In such cases, it becomes very important to determine whether the given query can be answered with given context passage or not before extracting/abstracting an answer from it.
Performing question-answering for a query which is not answerable from the context, could lead to incorrect answer extraction.

**Data** :- In this example, we are using the [MSMARCO_triples](https://msmarco.blob.core.windows.net/msmarcoranking/triples.train.small.tar.gz") data which is having sentence pairs and labels.
The data contains triplets where the first entry is the query, second one is the context passage from which the query can be answered (positive passage) , while the third entry is a context
passage from which the query cannot be answered (negative passage).

Data is transformed into sentence pair classification format, with query-positive context pair labeled as 1 (answerable) and query-negative context pair labeled as 0 (non-answerable)

**Transform file** :- [transform_file_answerability](https://github.com/hellohaptik/multi-task-NLP/tree/master/examples/answerability_detection/transform_file_answerability.yml)

**Tasks file** :- [tasks_file_answerability](https://github.com/hellohaptik/multi-task-NLP/tree/master/examples/answerability_detection/tasks_file_answerability.yml)

**Notebook** :- [answerability_detection_msmarco](https://github.com/hellohaptik/multi-task-NLP/tree/master/examples/answerability_detection/answerability_detection_msmarco.ipynb)
