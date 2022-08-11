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
This project requires Linux/OSX and Python 3.5 or higher. It also requires installing PyTorch version >= 1.0.0. 
Download  SQuAD datafiles, GloVe word vectors, Transformers and other dependencies listed in requirements.txt. 

CUDA is necessary to train the models.

Run the following commands to clone the repository and install DrQA:

```bash
git clone git@github.com:shreyasarunesh/Open_Domain_Question_Answering_Agent-ODQA-.git
cd Open_Domain_Question_Answering_Agent-ODQA-; pip install -r requirements.txt; 
```

The repository looks like the bellow format. 
```
Open_Domain_Question_Answering_Agent-ODQA-
├── Retriever
         ├── WikiIndexing
                ├── __main__.py
                ├── CreateIndex.py
                ├── DataWriter.py
                ├── MergeFiles.py
                ├── PageProcessor.py
                ├── SaxParcer.py
                ├── TextPreProcessor.py           
         ├── WikiSearching
                 ├── __main__.py
                 ├── BM25.py
                 ├── FileTraverser.py 
                 ├── QueryResults.py 
                 ├── RunQuery.py      
        ├── Evaluation
                 ├── Evaluation.py
├── Reader
        ├── BERT.py
        ├── DrQA.py
        ├── Eval_metrics.py
        ├── Preprocessing.py
├── Reader_model_output
        ├── BERT_finetuned_model_on_SQuAD_V 1.0
                    ├── config.json pytorch_model.bin
                    ├── pytorch_model.bin
        ├── BERT_finetuned_model_on_SQuAD_V 2.0
                    ├── config.json pytorch_model.bin
                    ├── pytorch_model.bin
        ├── DrQA_trained_SQuAD-1.pth
        ├── DrQA_trained_SQuAD-2.pthenglish_wiki_index
├── Dataset
        ├── english_wikipedia_Dump
        ├── squad1.0
        ├── squad2.1
        ├── output_data
                ├── english_wiki_index
        
```

