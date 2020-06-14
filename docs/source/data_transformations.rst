Data transformations
====================

It is very likely that the data you have is not in the format as required by the library.
Hence, data transformations provide a way to convert data in raw form to standard tsv format required.

Transform functions
-------------------

Transform functions are the functions which can be used for performing transformations.
Each function is defined to take raw data in certain format, perform the defined transformation steps and
create the respective ``tsv`` file.

Sample transform functions
^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: utils.tranform_functions
    :members: snips_intent_ner_to_tsv, snli_entailment_to_tsv, create_fragment_detection_tsv,
        msmarco_answerability_detection_to_tsv, msmarco_query_type_to_tsv, bio_ner_to_tsv, coNLL_ner_pos_to_tsv, qqp_query_similarity_to_tsv,
        query_correctness_to_tsv, imdb_sentiment_data_to_tsv

Your own transform function
^^^^^^^^^^^^^^^^^^^^^^^^^^^
In case, you need to convert some custom format data into the standard tsv format, you can do that
by writing your own transform function. You must keep the following points in mind while writing your function

- The function must take the standard input arguments like :ref:`sample transform functions<Sample transform functions>`
  Any extra function specific parameter can be added to the ``transParamDict`` argument.

- You should add the function in ``utils/tranform_functions.py`` file.

- You should add a name map for the function in ``utils/data_utils.py`` file under ``TRANSFORM_FUNCS`` map. This
  step is required for transform file to recognize your function.

- You should be able to use your function in the :ref:`transform file<Transform File>`.

Transform File
--------------

You can easily use the sample transformation functions or your own transformation function, 
by defining a YAML format ``transform_file``. Say you want to perform these transformations -
**sample_transform1**, **sample_transform2**, ..., **sample_transform5**.
Following is an example for the transform file,
::

  sample_transform1:
    transform_func: snips_intent_ner_to_tsv
    read_file_names:
      - snips_train.txt
      - snips_dev.txt
      - snips_test.txt
    read_dir: snips_data
    save_dir: demo_transform


  sample_transform2:
    transform_func: snli_entailment_to_tsv
    read_file_names:
      - snli_train.jsonl
      - snli_dev.jsonl
      - snli_test.jsonl
    read_dir : snli_data
    save_dir: demo_transform

  sample_transform3:
    transform_func: bio_ner_to_tsv
    transform_params:
      save_prefix : sample
      tag_col : 1
      col_sep : " "
      sen_sep : "\n"
    read_file_names:
      - coNLL_train.txt
      - coNLL_testa.txt
      - coNLL_testb.txt

    read_dir: coNLL_data
    save_dir: demo_transform

  sample_transform4:
    transform_func: fragment_detection_to_tsv
    transform_params:
      data_frac : 0.2
      seq_len_right : 3
      seq_len_left : 2
      sep : "\t"
      query_col : 2
    read_file_names:
      - int_snips_train.tsv
      - int_snips_dev.tsv
      - int_snips_test.tsv

    read_dir: data
    save_dir: demo_transform

  sample_transform5:
    transform_func: msmarco_query_type_to_tsv
    transform_params:
      data_frac : 0.2
    read_file_names:
      - train_v2.1.json
      - dev_v2.1.json
      - eval_v2.1_public.json

    read_dir: msmarco_qna_data
    save_dir: demo_transform


NOTE:- The transform names (sample_transform1, sample_transform2, ...) are unique identifiers for the transform, hence the transform names must always be distinct. 

Transform file parameters
^^^^^^^^^^^^^^^^^^^^^^^^^

Detailed description of the parameters available in the transform file.

- ``transform_func`` `(required)` : Name of the :ref:`transform function<Sample transform functions>` to use.
- ``transform_params`` `(optional)` : Dictionary of function specific parameters which will go in ``transParamDict`` parameter of function.
- ``read_file_names`` `(required)` : List of raw data files for transformations. The first file will be considered as **train file** and will be used to create label
  map file when required.
- ``read_dir`` `(required)` : Directory containing the input files.
- ``save_dir`` `(required)` : Directory to save the transformed tsv/label map files.


Running data transformations
----------------------------

Once you have made the :ref:`transform file<Transform File>` with all the transform operations, 
you can run data transformations with the following terminal command.

.. code-block:: console

  $ python data_transformations.py \
        --transform_file 'transform_file.yml'




