How to train?
=============

Once you have made the task file with the tasks you want to train for,
the next step is to run ``data_preparation.py`` and ``train.py``.

Running data preparation
------------------------

- The job of this script is to convert the given tsv data files to model inputs such as **Token Ids**, **Attention Masks** and **Token Type Ids** based on the shared encoder type.

- The script uses **multi-processing** which effectively reduces the data preparation time for large data files.

- It stores the prepared data in json files under the directory name **prepared_data** prefixed with the shared encoder config name.

The script takes the following arguments,

- ``task_file`` `(required)` :- Path to the created task file for which you want to train.

- ``data_dir`` `(required)` :- Path to the directory where the data files mentioned in task file are present.

- ``do_lower_case`` `(optional, default True)` :- Set this to False in case you are using  a `cased` config for model type.

- ``max_seq_len`` `(required, default 128)` :- Maximum sequence length for inputs. Truncating or padding will occur accordingly.

You can use the following terminal command with your own argument values to run.

.. code-block:: console

  $ python data_preparation.py \ 
        --task_file 'sample_task_file.yml' \
        --data_dir 'data' \
        --max_seq_len 50 

Running train
-------------

After ``data_preparation.py`` has finished running, it will store the respective prepared files
under the directory name ‘prepared_data’ prefixed with the shared encoder config name. 
The ``train.py`` can be run from terminal to start the training. Following arguments are
available

- ``data_dir`` `(required)` :- Path to the directory where prepared data is stored. (eg. bert_base_uncased_prepared_data)
- ``task_file`` `(required)` :-  Path to task file for training.
- ``out_dir`` `(required)` :- Path to save the multi-task model checkpoints.
- ``epochs`` `(required)` :- Number of epochs to train.
- ``train_batch_size`` `(optional, default 8)` :- Batch size for training.
- ``eval_batch_size`` `(optional, default 32)` :- Batch size for evaluation.
- ``grad_accumulation_steps`` `(optional, default 1)` :- Number of batches to accumulate before update.
- ``log_per_updates`` `(optional, default 10)` :- Number of updates after which to log loss.
- ``silent`` `(optional, default True)` :- Set to False for logs to be shown on terminal output as well. 
- ``max_seq_len`` `(optional, default 128)` :- Maximum sequence length which was used during data preparation.
- ``save_per_updates`` `(optional, default 0)` :- Number of update steps after which model checkpoint to be saved. Model is always saved at the end of every epoch. 
- ``load_saved_model`` `(optional, default None)` :- Path to the saved model in case of loading.
- ``resume_train`` `(optional, default False)` :- Set to True for resuming training from the saved model. Training will resume from the step at which the loaded model was saved.

You can use the following terminal command with your own argument values to run.

.. code-block:: console

  $ python train.py \
        --data_dir 'data/bert-base-uncased_prepared_data' \
        --task_file 'sample_task_file.yml' \
        --out_dir 'sample_out' \
        --epochs 5 \
        --train_batch_size 4 \
        --eval_batch_size 8 \
        --grad_accumulation_steps 2 \
        --max_seq_len 50 \
        --log_per_updates 25 \
        --save_per_updates 1000 \
        --eval_while_train \
        --test_while_train  \
        --silent

Logs and tensorboard
--------------------

- Logs for the training should be saved in a time-stamp named directory (eg. 05_05-17_30). 
- The tensorboard logs are also present in the same directory and tensorboard can be started with the following command

.. code-block:: console

  $ tensorboard --logdir 05_05-17_30/tb_logs


