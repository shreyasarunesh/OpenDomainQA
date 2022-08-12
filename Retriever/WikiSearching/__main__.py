

'''
 *
 * Name                   :   F21MP- Open Domain Question Answering (ODQA) Agent.
 *
 * Description            :  This is the main function for Multi stage retriever
                            that extracts top_K_N paragraphs for a given question.

 * Prerequisite           : Wikipedia index output file. (~ 22 GB)
 *
 *
 * Author                 :   Shreyas Arunesh
 *
 *
'''

import time
from Retriever.WikiSearching.RunQuery import *
from Retriever.WikiSearching.FileTraverser import *
from Retriever.WikiSearching.QueryResults import *
from Retriever.WikiSearching.BM25 import *

from Retriever.WikiIndexing.TextPreProcessor import *
import math
import argparse
import linecache
from collections import Counter

'''
 *
 *  Summary : main class for retriever. Queries can be simple or field specific. Search query types:

                type1: Simple- who is johnny depp
                type2: field query- t:who is jonny c:who is depp
                type3: Mixed - what is capital of scotland b:capital of scotland
 *
 *  Args    : Param - input query taken from user. 
 *
 *  Returns : list of top_K_N relevant paragraphs for a given question. 
 *
'''

if __name__ == '__main__':
    start = time.time()

    file_name = '/Retriever/Evaluation/questions'
    num_article = 10  # Value of K- number of articles.
    num_article2 = 25
    num_article3 = 35
    num_para = 20 # Number of para..
    num_para2 = 15
    num_para3 = 10

    print('Loading search engine ')
    html_tags = re.compile('&amp;|&apos;|&gt;|&lt;|&nbsp;|&quot;')
    with open('../Dataset/stemwords.txt',
              'r') as f:
        stop_words = [word.strip() for word in f]

    with open('../Dataset/stemwords.txt',
              'r') as f:
        stemmer = [word.strip() for word in f]

    with open('../Dataset/output_data/english_wiki_index/num_pages.txt','r') as f:
        num_pages = float(f.readline().strip())

    text_pre_processor = TextPreProcessor(html_tags, stemmer, stop_words)
    file_traverser = FileTraverser()
    ranker = BM25(num_pages)
    query_results = QueryResults(file_traverser)
    run_query = RunQuery(text_pre_processor, file_traverser, ranker, query_results)

    temp = linecache.getline('../Dataset/output_data/english_wiki_index/id_title_map.txt', 0)

    print('Loaded in', time.time() - start, 'seconds')

    print('Starting Querying')
    while True:
        question = input('Enter Query:- ')

        start = time.time()

        # run_query.take_input_from_file(file_name, num_article, num_para)

        # run_query.get_squad_predictions(num_article1,num_article2, num_article3, num_para1,num_para2, num_para3)


        top_para, doc_scores = run_query.take_input_from_user(question, num_article, num_para)

        # run_query.get_wikiQA_predictions(num_results)

        print('Done querying in', time.time() - start, 'seconds')
        print(top_para)
        print("###############")

        print(doc_scores)

