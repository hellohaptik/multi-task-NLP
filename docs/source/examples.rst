Examples
===========
Here you can find various NLP (especially conversational AI) tasks as examples and can train them either in multi-task or single-task manner, using some simple steps mentioned in the notebooks.

Example-1 Intent detection, NER, Fragment detection
---------------------------------------------------

**Tasks Description**

``Intent Detection`` :- This can be modeled as a single sentence classification task where an `intent` specifies which class the query belongs to. 

``NER`` :- This is a Named Entity Recognition/ Sequence Labelling/ Slot filling task where individual words of the sentence are tagged with an entity label it belongs to. The words which don't belong to any entity label are simply labeled as "O". 

``Fragment Detection`` :- This is modeled as a single sentence classification task which detects whether a sentence is incomplete (fragment) or not (non-fragment).

**Conversational Utility** :-  Intent detection is one of the fundamental components for conversational system as it gives a broad understand of the category/domain the sentence/query belongs to.

NER helps in extracting values for required entities (eg. location, date-time) from query.

Fragment detection is a very useful piece in conversational system as knowing if a query/sentence is incomplete can aid in recognising ill-formed , incomplete user queries and apprpriate expanded options can be provided to the user .

**Data** :- In this example, we are using the `SNIPS <https://snips-nlu.readthedocs.io/en/latest/dataset.html>`_  data for intent and entity detection. For the sake of simplicity, we provide 
the data in simpler form under ``snips_data`` directory taken from `here <https://github.com/LeePleased/StackPropagation-SLU/tree/master/data/snips>`_.

**Transform file** :- `transform_file_snips <https://github.com/hellohaptik/multi-task-NLP/blob/master/examples/intent_ner_fragment/transform_file_snips.yml>`_

**Tasks file** :-  `tasks_file_snips <https://github.com/hellohaptik/multi-task-NLP/blob/master/examples/intent_ner_fragment/tasks_file_snips.yml>`_

**Notebook** :- `intent_ner_fragment <https://github.com/hellohaptik/multi-task-NLP/blob/master/examples/intent_ner_fragment/intent_ner_fragment.ipynb>`_

Example-2 Recognising Textual Entailment 
------------------------------

**Tasks Description**

``Entailment`` :- This is a sentence pair classification task . Given two text instances ‘premise’ and ‘Hypothesis’, Textual Entailment Recognition is the task of determining whether the hypothesis is entailed (can be inferred) from the premise or not.

**Conversational Utility** :-  it can be used for evaluating pairwise similarity between queries , also to determine if a query is out-of-context or adversarial in conversational AI usecases like FAQ question answering. 

**Data** :- In this example, we are using the `SNLI <https://nlp.stanford.edu/projects/snli>`_ data which is having sentence pairs and labels.

**Transform file** :- `transform_file_snli <https://github.com/hellohaptik/multi-task-NLP/tree/master/examples/entailment_detection/transform_file_snli.yml>`_

**Tasks file** :- `tasks_file_snli <https://github.com/hellohaptik/multi-task-NLP/tree/master/examples/entailment_detection/tasks_file_snli.yml>`_

**Notebook** :- `entailment_snli <https://github.com/hellohaptik/multi-task-NLP/tree/master/examples/entailment_detection/entailment_snli.ipynb>`_

Example-3 Answerability detection
---------------------------------

**Tasks Description**

``answerability`` :- This is modeled as a sentence pair classification task where the first sentence is a query and second sentence is a context passage.
The objective of this task is to determine whether the query can be answered from the context passage or not.

**Conversational Utility** :- This can be a useful component for building a question-answering/ machine comprehension based system.
In such cases, it becomes very important to determine whether the given query can be answered with given context passage or not before extracting/abstracting an answer from it.
Performing question-answering for a query which is not answerable from the context, could lead to incorrect answer extraction.

**Data** :- In this example, we are using the `MSMARCO_triples <https://msmarco.blob.core.windows.net/msmarcoranking/triples.train.small.tar.gz">`_ data which is having sentence pairs and labels.
The data contains triplets where the first entry is the query, second one is the context passage from which the query can be answered (positive passage) , while the third entry is a context
passage from which the query cannot be answered (negative passage).

Data is transformed into sentence pair classification format, with query-positive context pair labeled as 1 (answerable) and query-negative context pair labeled as 0 (non-answerable)

**Transform file** :- `transform_file_answerability <https://github.com/hellohaptik/multi-task-NLP/tree/master/examples/answerability_detection/transform_file_answerability.yml>`_

**Tasks file** :- `tasks_file_answerability <https://github.com/hellohaptik/multi-task-NLP/tree/master/examples/answerability_detection/tasks_file_answerability.yml>`_

**Notebook** :- `answerability_detection_msmarco <https://github.com/hellohaptik/multi-task-NLP/tree/master/examples/answerability_detection/answerability_detection_msmarco.ipynb>`_
