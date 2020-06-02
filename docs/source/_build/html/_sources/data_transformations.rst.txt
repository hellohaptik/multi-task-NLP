Data transformations
====================

It is very likely that the data you have is not in the format as required by the library.
Hence, data transformations provide a way to convert data in raw form to standard tsv format required.

Transform Functions
-------------------

Transform functions are the functions which can be used for performing transformations.
Each function is defined to take raw data in certain format, perform the defined transformation steps and
create the respective ``tsv`` file.

Sample Transform Function
-------------------------
.. automodule:: utils.tranform_functions
    :members: snips_intent_ner_to_tsv, bio_ner_to_tsv, snli_entailment_to_tsv,
        fragment_detection_to_tsv, msmarco_query_type_to_tsv


