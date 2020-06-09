Examples
===========
Here you can find various conversational AI tasks as examples and can train multi-task models
in simple steps mentioned in the notebooks.

Example-1 Intent detection, NER, Fragment detection
---------------------------------------------------

**Tasks Description**

``Intent Detection`` :- This is a single sentence classification task where an `intent` specifies which class the data sample belongs to.
Intent detection is one of the fundamental components for conversational system as it gives a broad understand of the category/domain the sentence/query belongs to. 

``NER`` :- This is a Named Entity Recognition/ Sequence Labelling/ Slot filling task where individual words of the sentence are tagged with an entity label it belongs to.
The words which don't belong to any entity label are simply labeled as "O". NER helps in extracting values for required entities (eg. location, date-time) from query.

``Fragment Detection`` :- This is modeled as a single sentence classification task which detects whether a sentence is incomplete (fragment) or not (non-fragment).
This is a very useful piece in conversational system as knowing if a query/sentence is incomplete can aid in discarding bad queries beforehand.

**Data** :- In this example, we are using the `SNIPS <https://snips-nlu.readthedocs.io/en/latest/dataset.html>`_  data for intent and entity detection. For the sake of simplicity, we provide 
the data in simpler form under ``snips_data`` directory taken from `here <https://github.com/LeePleased/StackPropagation-SLU/tree/master/data/snips>`_.

**Transform file** :- `transform_file_snips <https://github.com/hellohaptik/multi-task-NLP/blob/master/examples/intent_ner_fragment/transform_file_snips.yml>`_

**Tasks file** :-  `tasks_file_snips <https://github.com/hellohaptik/multi-task-NLP/blob/master/examples/intent_ner_fragment/tasks_file_snips.yml>`_

**Notebook** :- `intent_ner_fragment <https://github.com/hellohaptik/multi-task-NLP/blob/master/examples/intent_ner_fragment/intent_ner_fragment.ipynb>`_

Example-2 Entailment detection
------------------------------

**Tasks Description**

``Entailment`` :- This is a sentence pair classification task which determines whether the second sentence
in a sample can be inferred from the first. In conversational AI context, this task can be seen as determining whether the second sentence is similar to first or not.
Additionally, the probability score can also be used as a similarity score between the sentences. 


**Data** :- In this example, we are using the `SNLI <https://nlp.stanford.edu/projects/snli>`_ data which is having sentence pairs and labels.

**Transform file** :- `transform_file_snli <https://github.com/hellohaptik/multi-task-NLP/tree/master/examples/entailment_detection/transform_file_snli.yml>`_

**Tasks file** :- `tasks_file_snli <https://github.com/hellohaptik/multi-task-NLP/tree/master/examples/entailment_detection/tasks_file_snli.yml>`_

**Notebook** :- `entailment_snli <https://github.com/hellohaptik/multi-task-NLP/tree/master/examples/entailment_detection/entailment_snli.ipynb>`_

