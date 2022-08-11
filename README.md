# Open_Domain_Question_Answering_Agent-ODQA-
This repository consists of ODQA implementation used for MSc Artificial Intelligence Masters Project in Heriot-Watt-University 

# Introduction

This is the Pytorch Re-implementation of DrQA model described in the paper: [Reading Wikipedia to Answer Open-Domain Questions](https://arxiv.org/abs/1704.00051) 
and fine-tuning of BERT-base as described in the paper: [End-to-End Open-Domain Question Answering with BERTserini](https://arxiv.org/abs/1902.01718).

The goal of this project is to implement an End-to-end open domain Question Answering system (ODQA) for any factoid question using Wikipedia as the unique knowledge source. 
This task of large scale machine reading is combined with the challenges of document retrieval (finding the relevant articles from Wikipedia )
and machine comprehension of text (locating the answer spans from those articles). For a given question, this project consists of two
components: (1) Information Retrieval System to extract relevant documents among the collection of more than 6 million Wikipedia articles and (2) Machine Reading Comprehension
to scan the retrieved documents to find the answer. Lastly, the performance of both components are assessed individually using the intrinsic evaluation metrics.


# Quick Demo of interactive Session

For interactive Session as illustrated bellow, run the __interactive__.py file. In the example, Top Predicted answer is
the algorithm that selects the best answers among the top listed Answers displayed in the table. 

```bash
__interactive__.py
```
Example 1:
![Interactive Sesion-1](Images/interactive.png)

Example 2:
![Interactive Sesion-2](Images/interactive2.png)


## Installing

Installation of this system is simple. 

```bash
git clone git@github.com:shreyasarunesh/Open_Domain_Question_Answering_Agent-ODQA-.git
cd Open_Domain_Question_Answering_Agent-ODQA-; pip install -r requirements.txt; 
```

```
Open_Domain_Question_Answering_Agent-ODQA-
├── Retriever
    │   ├── WikiIndexing
        │   ├── __main__.py
        │   ├── CreateIndex.py
        │   ├── DataWriter.py
        │   ├── MergeFiles.py
        │   ├── PageProcessor.py
        │   ├── SaxParcer.py
        │   ├── TextPreProcessor.py           
    │   ├── WikiSearching
        │   ├── __main__.py
        │   ├── BM25.py
        │   ├── FileTraverser.py 
        │   ├── QueryResults.py 
        │   ├── RunQuery.py      
    │   ├── Evaluation
        │   ├── Evaluation.py
├── Reader
    │   ├── SQuAD-v1.1-<train/dev>.<txt/json>
    │   ├── WebQuestions-<train/test>.txt
    │   ├── freebase-entities.txt
    │   ├── CuratedTrec-<train/test>.txt
    │   └── WikiMovies-<train/test/entities>.txt
    ├── reader
    │   ├── multitask.mdl
    │   └── single.mdl
    └── wikipedia
        ├── docs.db
        └── docs-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz
```