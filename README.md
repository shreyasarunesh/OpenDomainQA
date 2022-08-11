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

>>> Enter Query:- where is nepal Located
> Top Predicted Answer: it is Located in the kathmandu valley, a large valley in the high plateaus in central nepal
|  Top Answers   |   Top Wikipedia Titles  |
|-----|-----|
| it is located in the kathmandu valley,
a large valley in the high plateaus in central nepal    |  kingdom of nepal   |
|  | the windswept and arid Land around lo manthang, located at an altitude between 3000m and 3500m, is not suitable for agriculture at all   |   | nepal  |
|     |     |
|     |     |
|     |     |
|     |     |
|     |     |
|     |     |
|     |     |
