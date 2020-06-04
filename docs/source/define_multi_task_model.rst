How to define your multi-task model?
====================================

Let’s consider you have three tasks - **TaskA**, **TaskB** and **TaskC** to train together. TaskA is single sentence classification type,
TaskB is NER type and TaskC is sentence pair classification type. 
You can define a task file mentioning the required details about the task in following YAML format.
::

  TaskA:
    model_type: BERT
    config_name: bert-base-uncased
    dropout_prob: 0.05
    label_map_or_file:
    -label1
    -label2
    -label3
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

  TaskC:
    model_type: BERT
    config_name: bert-base-uncased
    dropout_prob: 0.05
    metrics:
    - accuracy
    loss_type: CrossEntropyLoss
    class_num: 2
    task_type: SentencePairClassification
    file_names:
    - taskC_train.tsv
    - taskC_dev.tsv
    - taskC_test.tsv

Few points to keep in mind while making the task file,

- You can keep the tasks which you want to train a single model for in this file.
- The file can have either a single task or multiple tasks. In case only a single task is mentioned, the model will act like single-task model.
- The task names (TaskA, TaskB and TaskC) are unique identifiers for the task, hence the task names must always be distinct. 
- The model type for all the tasks mentioned in the file must be the same, as the library uses a single shared encoder model for all these tasks.

Task file parameters
--------------------

Detailed description of the parameters available in the task file.

- ``task_type`` `(required)` :  Format of the task as described in :ref:`Task types<Task types>`

- ``file_names`` `(required)` : List of standard data tsv file names required for task. The first file is considered as **train** file, second file as **dev** file and the third file as **test** file.

- ``model_type`` `(required)` : Type of shared encoder model to use. The model type for all the tasks mentioned in the file must be the same. You can refer :ref:`Model type<Choice of shared encoder>` for selecting model type.

- ``config_name`` `(optional)` : Config of the encoder model. You can refer :ref:`Model type<Choice of shared encoder>` for selecting the model type config. In case this parameter is not present, default config will be used.

- ``class_num``  `(required/optional)` : Number of classes present for classification. This parameter is optional if label_map_or_file is provided, required otherwise.

- ``label_map_or_file``  `(required/optional)` :

  - In case labels are strings, this is the list of unique labels.
  - You can also give a joblib dumped dictionary map file like {‘label1’:0, ‘label2’:1, ..}.
  - If you’re using :ref:`Data Transformations<Data transformations>` to create the data files, path to the label_map file created along with transformed files is to be given here.
  
- ``loss_type`` `(required)` : Type of loss for training as defined in :ref:`Losses<Losses>`.

- ``dropout_prob`` `(optional)`: Dropout probability to use between encoder hidden outputs and task specific headers.

- ``metrics`` `(optional)` : List of metrics to use during evaluation as defined in :ref:`Metrics<Metrics>`.

- ``loss_weight`` `(optional)`: Loss weight value (between 0 to 1) for individual task.



