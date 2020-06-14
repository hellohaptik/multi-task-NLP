Examples
===========
Here you can find various NLP (especially conversational AI) tasks as examples and can train them either in multi-task or single-task manner, using some simple steps mentioned in the notebooks.

Example-1 Intent detection, NER, Fragment detection
---------------------------------------------------

**Tasks Description**

``Intent Detection`` :- This is a single sentence classification task where an `intent` specifies which class the data sample belongs to. 

``NER`` :- This is a Named Entity Recognition/ Sequence Labelling/ Slot filling task where individual words of the sentence are tagged with an entity label it belongs to. The words which don't belong to any entity label are simply labeled as "O". 

``Fragment Detection`` :- This is modeled as a single sentence classification task which detects whether a sentence is incomplete (fragment) or not (non-fragment).

**Conversational Utility** :-  Intent detection is one of the fundamental components for conversational system as it gives a broad understand of the category/domain the sentence/query belongs to.

NER helps in extracting values for required entities (eg. location, date-time) from query.

Fragment detection is a very useful piece in conversational system as knowing if a query/sentence is incomplete can aid in discarding bad queries beforehand.

**Intent Detection**

  Query: I need a reservation for a bar in bangladesh on feb the 11th 2032
 
  Intent: BookRestaurant

**NER**

 
  Query: ['book', 'a', 'spot', 'for', 'ten', 'at', 'a', 'top-rated', 'caucasian', 'restaurant', 'not', 'far', 'from', 'selmer']

  NER tags: ['O', 'O', 'O', 'O', 'B-party_size_number', 'O', 'O', 'B-sort', 'B-cuisine', 'B-restaurant_type', 'B-spatial_relation', 'I-spatial_relation', 'O', 'B-city']
 

**Fragment Detection**

 
  Query: a reservation for

  Label: fragment
 

**Notebook** :- `intent_ner_fragment <https://github.com/hellohaptik/multi-task-NLP/blob/master/examples/intent_ner_fragment/intent_ner_fragment.ipynb>`_

**Transform file** :- `transform_file_snips <https://github.com/hellohaptik/multi-task-NLP/blob/master/examples/intent_ner_fragment/transform_file_snips.yml>`_

**Tasks file** :-  `tasks_file_snips <https://github.com/hellohaptik/multi-task-NLP/blob/master/examples/intent_ner_fragment/tasks_file_snips.yml>`_

Example-2 Recognising Textual Entailment 
----------------------------------------

**Tasks Description**

``Entailment`` :- This is a sentence pair classification task which determines whether the second sentence in a sample can be inferred from the first.

**Conversational Utility** :-  In conversational AI context, this task can be seen as determining whether the second sentence is similar to first or not. Additionally, the probability score can also be used as a similarity score between the sentences. 
 
  Query1: An old man with a package poses in front of an advertisement.

  Query2: A man poses in front of an ad.

  Label: entailment

  Query1: An old man with a package poses in front of an advertisement.

  Query2: A man poses in front of an ad for beer.

  Label: non-entailment

 

**Notebook** :- `entailment_snli <https://github.com/hellohaptik/multi-task-NLP/tree/master/examples/entailment_detection/entailment_snli.ipynb>`_

**Transform file** :- `transform_file_snli <https://github.com/hellohaptik/multi-task-NLP/tree/master/examples/entailment_detection/transform_file_snli.yml>`_

**Tasks file** :- `tasks_file_snli <https://github.com/hellohaptik/multi-task-NLP/tree/master/examples/entailment_detection/tasks_file_snli.yml>`_



Example-3 Answerability detection
---------------------------------
**Tasks Description**

``answerability`` :- This is modeled as a sentence pair classification task where the first sentence is a query and second sentence is a context passage. The objective of this task is to determine whether the query can be answered from the context passage or not.

**Conversational Utility** :- This can be a useful component for building a question-answering/ machine comprehension based system. In such cases, it becomes very important to determine whether the given query can be answered with given context passage or not before extracting/abstracting an answer from it. Performing question-answering for a query which is not answerable from the context, could lead to incorrect answer extraction.
 
  Query: how much money did evander holyfield make

  Context: Evander Holyfield Net Worth. How much is Evander Holyfield Worth? Evander Holyfield Net Worth: Evander Holyfield is a retired American professional boxer who has a net worth of $500 thousand. A professional boxer, Evander Holyfield has fought at the Heavyweight, Cruiserweight, and Light-Heavyweight Divisions, and won a Bronze medal a the 1984 Olympic Games.

  Label: answerable
 
**Notebook** :- `answerability_detection_msmarco <https://github.com/hellohaptik/multi-task-NLP/tree/master/examples/answerability_detection/answerability_detection_msmarco.ipynb>`_

**Transform file** :- `transform_file_answerability <https://github.com/hellohaptik/multi-task-NLP/tree/master/examples/answerability_detection/transform_file_answerability.yml>`_

**Tasks file** :- `tasks_file_answerability <https://github.com/hellohaptik/multi-task-NLP/tree/master/examples/answerability_detection/tasks_file_answerability.yml>`_

Example-4 Query type detection
------------------------------
 
**Tasks Description**

``querytype`` :- This is a single sentence classification task to determine what type (category) of answer is expected for the given query. The queries are divided into 5 major classes according to the answer expected for them.

**Conversational Utility** :-  While returning a response for a query, knowing what kind of answer is expected for the query can help in both curating and cross-verfying an answer according to the type.

  Query: what's the distance between destin florida and birmingham alabama?

  Label: NUMERIC

  Query: who is suing scott wolter

  Label: PERSON

 

**Notebook** :- `query_type_detection <https://github.com/hellohaptik/multi-task-NLP/blob/master/examples/query_type_detection/query_type_detection.ipynb>`_

**Transform file** :- `transform_file_querytype <https://github.com/hellohaptik/multi-task-NLP/blob/master/examples/query_type_detection/transform_file_querytype.yml>`_

**Tasks file** :- `tasks_file_querytype <https://github.com/hellohaptik/multi-task-NLP/blob/master/examples/query_type_detection/tasks_file_querytype.yml>`_

Example-5 POS tagging, NER tagging
----------------------------------
 
**Tasks Description**

``NER`` :-This is a Named Entity Recognition task where individual words of the sentence are tagged with an entity label it belongs to. The words which don't belong to any entity label are simply labeled as "O".

``POS`` :- This is a Part of Speech tagging task. A part of speech is a category of words that have similar grammatical properties. Each word of the sentence is tagged with the part of speech label it belongs to. The words which don't belong to any part of speech label are simply labeled as "O".

**Conversational Utility** :-  In conversational AI context, determining the syntactic parts of the sentence can help in extracting noun-phrases or important keyphrases from the sentence.

  Query: ['Despite', 'winning', 'the', 'Asian', 'Games', 'title', 'two', 'years', 'ago', ',', 'Uzbekistan', 'are', 'in', 'the', 'finals', 'as', 'outsiders', '.']

  NER tags: ['O', 'O', 'O', 'I-MISC', 'I-MISC', 'O', 'O', 'O', 'O', 'O', 'I-LOC', 'O', 'O', 'O', 'O', 'O', 'O', 'O']

  POS tags: ['I-PP', 'I-VP', 'I-NP', 'I-NP', 'I-NP', 'I-NP', 'B-NP', 'I-NP', 'I-ADVP', 'O', 'I-NP', 'I-VP', 'I-PP', 'I-NP', 'I-NP', 'I-SBAR', 'I-NP', 'O']

 

**Notebook** :- `ner_pos_tagging_conll <https://github.com/hellohaptik/multi-task-NLP/blob/master/examples/ner_pos_tagging/ner_pos_tagging_conll.ipynb>`_

**Transform file** :- `transform_file_conll <https://github.com/hellohaptik/multi-task-NLP/blob/master/examples/ner_pos_tagging/transform_file_conll.yml>`_

**Tasks file** :- `tasks_file_conll <https://github.com/hellohaptik/multi-task-NLP/blob/master/examples/ner_pos_tagging/tasks_file_conll.yml>`_

Example-6 Query correctness
---------------------------

**Tasks Description**

``querycorrectness`` :- This is modeled as single sentence classification task identifying  whether or not  a query is structurally well formed.  can  enhance  query  un-derstanding.

**Conversational Utility** :- Determining how much the query is structured would help in enhancing query understanding and improve reliability of tasks which depend on query structure to extract information.

  Query: What places have the oligarchy government ?

  Label: well-formed

  Query: What day of Diwali in 1980 ?

  Label: not well-formed

 

**Notebook** :- `query_correctness <https://github.com/hellohaptik/multi-task-NLP/blob/master/examples/query_correctness/query_correctness.ipynb>`_

**Transform file** :- `transform_file_query_correctness <https://github.com/hellohaptik/multi-task-NLP/blob/master/examples/query_correctness/transform_file_query_correctness.yml>`_

**Tasks file** :- `tasks_file_query_correctness <https://github.com/hellohaptik/multi-task-NLP/blob/master/examples/query_correctness/tasks_file_query_correctness.yml>`_


Example-7 Query similarity
--------------------------
 
**Tasks Description**

``Query similarity`` :- This is a sentence pair classification task which determines whether the second sentence in a sample can be inferred from the first.

**Conversational Utility** :-  In conversational AI context, this task can be seen as determining whether the second sentence is similar to first or not. Additionally, the probability score can also be used as a similarity score between the sentences. 


  Query1: What is the most used word in Malayalam?

  Query2: What is meaning of the Malayalam word ""thumbatthu""?

  Label: not similar

  Query1: Which is the best compliment you have ever received?

  Query2: What's the best compliment you've got?

  Label: similar

 
**Notebook** :- `query_similarity <https://github.com/hellohaptik/multi-task-NLP/blob/master/examples/query_pair_similarity/query_similarity_qqp.ipynb>`_

**Transform file** :- `transform_file_qqp <https://github.com/hellohaptik/multi-task-NLP/blob/master/examples/query_pair_similarity/transform_file_qqp.yml>`_

**Tasks file** :- `tasks_file_qqp <https://github.com/hellohaptik/multi-task-NLP/blob/master/examples/query_pair_similarity/tasks_file_query_qqp.yml>`_

Example-8 Sentiment Analysis
----------------------------

**Tasks Description**

``sentiment`` :- This is modeled as single sentence classification task to determine where a piece of text conveys a positive or negative sentiment.

**Conversational Utility** :- To determine whether a review is positive or negative.

  Review: What I enjoyed most in this film was the scenery of Corfu, being Greek I adore my country and I liked the flattering director's point of view. Based on a true story during the years when Greece was struggling to stand on her own two feet through war, Nazis and hardship.
  An Italian soldier and a Greek girl fall in love but the times are hard and they have a lot of sacrifices to make. Nicholas Cage looking great in a uniform gives a passionate account of this unfulfilled (in the beginning) love. I adored Christian Bale playing Mandras
  the heroine's husband-to-be, he looks very very good as a Greek, his personality matched the one of the Greek patriot! A true fighter in there, or what! One of the movies I would like to buy and keep it in my collection...for ever!

  Label: positive

 

**Notebook** :- `IMDb_sentiment_analysis <https://github.com/hellohaptik/multi-task-NLP/blob/master/examples/sentiment_analysis/IMDb_sentiment_analysis.ipynb>`_

**Transform file** :- `transform_file_imdb <https://github.com/hellohaptik/multi-task-NLP/blob/master/examples/sentiment_analysis/transform_file_imdb.yml>`_ 

**Tasks file** :-  `tasks_file_imdb <https://github.com/hellohaptik/multi-task-NLP/blob/master/examples/sentiment_analysis/tasks_file_query_imdb.yml>`_

