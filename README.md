# AutoQGS

AutoQGS: Auto-Prompt for Low-Resource Knowledge-based Question Generation from SPARQL

This paper was published at 31st ACM International Conference on Information and Knowledge Management (CIKM 2022).

## Abstract

This study investigates the task of knowledge-based question generation (KBQG). Conventional KBQG works generated questions from fact triples in the knowledge graph, which could not express complex operations like aggregation and comparison in SPARQL. Moreover, due to the costly annotation of large-scale SPARQLquestion pairs, KBQG from SPARQL under low-resource scenarios urgently needs to be explored. Recently, since the generative pre-trained language models (PLMs) typically trained in natural language (NL)-to-NL paradigm have been proven effective for lowresource generation, e.g., T5 and BART, how to effectively utilize them to generate NL-question from non-NL SPARQL is challenging. To address these challenges, AutoQGS, an auto-prompt approach for low-resource KBQG from SPARQL, is proposed. Firstly, we put forward to generate questions directly from SPARQL for KBQG task to handle complex operations. Secondly, we propose an auto-prompter trained on large-scale unsupervised data to rephrase SPARQL into NL description, smoothing the low-resource transformation from non-NL SPARQL to NL question with PLMs. Experimental results on the WebQuestionsSP, ComlexWebQuestions 1.1, and PathQuestions show that our model achieves state-of-the-art performance, especially in low-resource settings. Furthermore, a corpora of 330k factoid complex question-SPARQL pairs is generated for further KBQG research.

# Datasets

#### WQCWQ1.1

We merge and reprocess [WebQuestionsSP](https://www.microsoft.com/en-us/download/details.aspx?id=52763) and [ComplexWebQuestions V1.1](https://www.tau-nlp.sites.tau.ac.il/compwebq) datasets by ourselves and release a new dataset, `WQCWQ1.1`. We leave the original training set untouched and randomly divide the validation/test set equally.

The dataset is available at: [dataset/WQCWQ1.1_KGQG.tar.gz](dataset/WQCWQ1.1_KGQG.tar.gz)


#### PathQuestion

We also reprocess [PathQuestion](https://github.com/zmtkeke/IRN). The original dataset did not provide entity mid, so we try to do entity linking in Freebase using some string matching strategies. Note that despite missing entity mid, we are able to construct the subgraph based on the entity name and generate questions as well.

The dataset is available at: [dataset/PathQuestion_KGQG.tar.gz](dataset/PathQuestion_KGQG.tar.gz)


## Citation

TBC