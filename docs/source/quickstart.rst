Quickstart
===========
Follow these 3 simple steps to train your multi-task model!

Step 1 - Define your task file
------------------------------

Task file is a YAML format file where you can add all your tasks for which you want to train a multi-task model.

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

For knowing about the task file parameters to make your task file, refer :ref:`here<Task file parameters>`.

Step 2 - Run data preparation
-----------------------------

After defining the task file in :ref:`Step 1<Step 1 - Define your task file>`, run the following command to prepare the data.

.. code-block:: console
  
  $ python data_preparation.py \ 
      --task_file 'sample_task_file.yml' \
      --data_dir 'data' \
      --max_seq_len 50 

For knowing about the ``data_preparation.py`` script and its arguments, refer :ref:`here<Running data preparation>`.

Step 3 - Run train
------------------

Finally you can start your training using the following command.

.. code-block:: console
  
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

For knowing about the ``train.py`` script and its arguments, refer :ref:`here<Running train>`.


