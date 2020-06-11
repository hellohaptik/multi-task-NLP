Shared Encoder
==============

What is a shared encoder?
-------------------------

The concept of this library is to provide a single model for multiple tasks.
To achieve this we place a transformer-based encoder at centre. Data for all tasks will go through this centre encoder.
This encoder is called shared as it is responsible for majority of learnings on all the tasks. 
Further, task specific headers are formed over the shared encoder.

Task specific headers
---------------------

The encoder hidden states are consumed by task specific layers defined to output logits in the format required by the task.
Forward pass for a data batch belonging to say taskA occurs through the shared encoder and header for taskA.
The computed loss (which is called as ‘task loss’) is back-propagated through the same path.

Choice of shared encoder
------------------------

We support multiple transformer-based encoder models.
For ease of use, we’ve integrated the encoders from the `transformers <https://github.com/huggingface/transformers>`_ library.
Available encoders with their config names are mentioned below.

+------------------+---------------------------+---------------------------+
|    Model type    |       Config name         | Default config            |
+==================+===========================+===========================+
|                  |  distilbert-base-uncased  |                           |
|   DISTILBERT     +---------------------------+  distilbert-base-uncased  |
|                  |  distilbert-base-cased    |                           |
+------------------+---------------------------+---------------------------+
|                  |    bert-base-uncased      |                           |
|                  +---------------------------+                           |
|                  |     bert-base-cased       |                           |
|      BERT        +---------------------------+     bert-base-uncased     |
|                  |    bert-large-uncased     |                           |
|                  +---------------------------+                           |
|                  |     bert-large-cased      |                           |
+------------------+---------------------------+---------------------------+
|                  |      roberta-base         |                           |
|     ROBERTA      +---------------------------+       roberta-base        |
|                  |      roberta-large        |                           |
+------------------+---------------------------+---------------------------+
|                  |    albert-base-v1         |                           |
|                  +---------------------------+                           |
|                  |     albert-large-v1       |                           |
|                  +---------------------------+                           |
|                  |    albert-xlarge-v1       |                           |
|                  +---------------------------+                           |
|                  |     albert-xxlarge-v1     |                           |
|     ALBERT       +---------------------------+      albert-base-v1       |
|                  |    albert-base-v2         |                           |
|                  +---------------------------+                           |
|                  |     albert-large-v2       |                           |
|                  +---------------------------+                           |
|                  |    albert-xlarge-v2       |                           |
|                  +---------------------------+                           |
|                  |     albert-xxlarge-v2     |                           |
+------------------+---------------------------+---------------------------+
|                  |      xlnet-base-cased     |                           |
|     XLNET        +---------------------------+      xlnet-base-cased     |
|                  |      xlnet-large-cased    |                           |
+------------------+---------------------------+---------------------------+

Losses
------
We support following two types of loss functions.

.. autoclass:: models.loss.CrossEntropyLoss
    :members: forward

.. autoclass:: models.loss.NERLoss
    :members: forward

Metrics
-------
For evaluating the performance on dev and test sets during training, we provide the following standard metrics.

.. automodule:: utils.eval_metrics
    :members: classification_accuracy, classification_f1_score, seqeval_f1_score, 
        seqeval_precision, seqeval_recall, snips_f1_score, snips_precision, snips_recall, classification_recall_score

