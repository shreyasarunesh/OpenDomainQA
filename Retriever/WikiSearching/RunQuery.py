import math
import os
from collections import Counter

from rank_bm25 import BM25Plus

from Retriever.WikiIndexing.TextPreProcessor import *
import time
import wikipediaapi
import json
import bs4
import requests
import unicodedata
import re
import pandas as pd

wiki_wiki = wikipediaapi.Wikipedia(language='en', extract_format=wikipediaapi.ExtractFormat.WIKI)

def get_paragraphs( page_name):
    r = requests.get('https://en.wikipedia.org/api/rest_v1/page/html/{0}'.format(page_name))
    soup = bs4.BeautifulSoup(r.content, 'html.parser')
    html_paragraphs = soup.find_all('p')

    for p in html_paragraphs:
        cleaned_text = re.sub('(\[[0-9]+\])', '', unicodedata.normalize('NFKD', p.text)).strip()
        if cleaned_text:
            yield cleaned_text

'''
This class runs all the classes to implement search and ranking and returns the required results.
'''


class RunQuery():

    def __init__(self, text_pre_processor, file_traverser, ranker, query_results):

        self.file_traverser = file_traverser
        self.text_pre_processor = text_pre_processor
        self.ranker = ranker
        self.query_results = query_results

    def identify_query_type(self, query):

        field_replace_map = {
            ' t:': ';t:',
            ' b:': ';b:',
            ' c:': ';c:',
            ' i:': ';i:',
            ' l:': ';l:',
            ' r:': ';r:',
        }

        if (
                't:' in query or 'b:' in query or 'c:' in query or 'i:' in query or 'l:' in query or 'r:' in query) and query[
                                                                                                                        0:2] not in [
            't:', 'b:', 'i:', 'c:', 'r:', 'l:']:

            for k, v in field_replace_map.items():
                if k in query:
                    query = query.replace(k, v)

            query = query.lstrip(';')

            return query.split(';')[0], query.split(';')[1:]

        elif 't:' in query or 'b:' in query or 'c:' in query or 'i:' in query or 'l:' in query or 'r:' in query:

            for k, v in field_replace_map.items():
                if k in query:
                    query = query.replace(k, v)

            query = query.lstrip(';')

            return query.split(';'), None

        else:
            return query, None

    def return_query_results(self, query, query_type):

        if query_type == 'field':
            preprocessed_query = [
                [qry.split(':')[0], self.text_pre_processor.preprocess_text(qry.split(':')[1])] for qry
                in query]
        else:
            preprocessed_query = self.text_pre_processor.preprocess_text(query)

        if query_type == 'field':

            preprocessed_query_final = []
            for field, words in preprocessed_query:
                for word in words:
                    preprocessed_query_final.append([field, word])

            page_freq, page_postings = self.query_results.field_query(preprocessed_query_final)

        else:

            page_freq, page_postings = self.query_results.simple_query(preprocessed_query)

        ranked_results = self.ranker.do_ranking(page_freq, page_postings)

        return ranked_results

    def get_paragraphs(self, page_name):

        r = requests.get('https://en.wikipedia.org/api/rest_v1/page/html/{0}'.format(page_name))
        soup = bs4.BeautifulSoup(r.content, 'html.parser')
        html_paragraphs = soup.find_all('p')

        for p in html_paragraphs:
            cleaned_text = re.sub('(\[[0-9]+\])', '', unicodedata.normalize('NFKD', p.text)).strip()
            if cleaned_text:
                yield cleaned_text

    def take_input_from_file(self, file_name, num_article, num_para):
        results_file = file_name.split('.txt')[0]

        with open(file_name, 'r') as f:

            for i, query in enumerate(f):
                s = time.time()
                tp = open(
                    f'/Users/shreyasarunesh/Desktop/Open_Domain_Question_Answering_Agent/Retriever/output_data/question_para/question{i + 1}.txt',
                    'w')

                query = query.strip()

                ranked_results = self.return_query_results(query, 'simple')

                results = sorted(ranked_results.items(), key=lambda item: item[1], reverse=True)
                results = results[:num_article]

                tokenized_query = self.text_pre_processor.preprocess_text(query)

                if results:
                    for id, _ in results:
                        title = self.file_traverser.title_search(id)
                        corpus = (list(self.get_paragraphs(title))[:])
                        if corpus != '':
                            for para in corpus:
                                tp.write(para)
                                print(
                                    'Document ID: {0} \n Document title: {1} \n Document Context: {2}'.format(id, title,
                                                                                                              para))
                                tp.write('\n\n')

                else:
                    tp.write('No matching Doc found')

                tp.close()

                predicted_file = open(
                    f'/Users/shreyasarunesh/Desktop/Open_Domain_Question_Answering_Agent/Retriever/output_data/question_para/question{i + 1}.txt',
                    'r')

                corpus = (predicted_file.read().split('\n\n'))
                while '' in corpus:
                    corpus.remove('')

                tokenized_corpus = [self.text_pre_processor.preprocess_text(doc) for doc in corpus]
                bm25 = BM25Plus(tokenized_corpus)

                # doc_scores = sorted(bm25.get_scores(tokenised_query), reverse=True)

                top_para = bm25.get_top_n(tokenized_query, corpus, n=num_para)
                fc = open(
                    f'/Users/shreyasarunesh/Desktop/Open_Domain_Question_Answering_Agent/Retriever/output_data/top_n_para/question{i + 1}.txt',
                    'w')
                for para in top_para:
                    fc.write(para)
                    fc.write('\n\n')

                e = time.time()

                print('Done query: {0} \n Finished in: {1} seconds'.format(i + 1, str(e - s)) + '\n')

            fc.close()

        print('Done writing results from Question file')

    def get_squad_predictions(self, num_article1, num_article2, num_article3, num_para1, num_para2, num_para3):

        true_context_directory = [f for f in os.listdir(
            "/Users/shreyasarunesh/Desktop/Open_Domain_Question_Answering_Agent/Retriever/Evaluation/squad1_output/true_context")
                                  if not f.startswith('.')]
        true_context_directory.sort(key=lambda f: int(re.sub('\D', '', f)))

        start = time.time()

        for iny, file in enumerate(true_context_directory):

            print('Extracting contexts and titles for true_context:' + str({iny + 51}))

            t = open(
                f'/Users/shreyasarunesh/Desktop/Open_Domain_Question_Answering_Agent/Retriever/Evaluation/squad1_output/true_context/context{iny + 51}.txt',
                'r')
            data = t.read().split('\n\n')

            true_context, questions = data[0], data[1].split('\n')

            while '' in questions:
                questions.remove('')

            s = time.time()

            for i, query in enumerate(questions):

                print('Executing Query number: {0} in true_context file: {1}'.format(i + 1, iny + 51))

                fp1 = open(
                    f'/Users/shreyasarunesh/Desktop/Open_Domain_Question_Answering_Agent/Retriever/Evaluation/squad1_output/predicted_contexts/1-predicted_context/{iny + 51}_{i + 1}.txt',
                    'w')
                fp2 = open(
                    f'/Users/shreyasarunesh/Desktop/Open_Domain_Question_Answering_Agent/Retriever/Evaluation/squad1_output/predicted_contexts/2-predicted_context/{iny + 51}_{i + 1}.txt',
                    'w')
                fp3 = open(
                    f'/Users/shreyasarunesh/Desktop/Open_Domain_Question_Answering_Agent/Retriever/Evaluation/squad1_output/predicted_contexts/3-predicted_context/{iny + 51}_{i + 1}.txt',
                    'w')

                query = query.strip()

                ranked_results = self.return_query_results(query, 'simple')

                results = sorted(ranked_results.items(), key=lambda item: item[1], reverse=True)

                results1 = results[:num_article1]
                results2 = results[:num_article2]
                results3 = results[:num_article3]

                tokenized_query = self.text_pre_processor.preprocess_text(query)

                if results1:

                    for id, _ in results1:

                        title = self.file_traverser.title_search(id)
                        title = title.strip()

                        corpus = (list(self.get_paragraphs(title))[:])
                        if corpus != '':
                            for para in corpus:
                                if len(para) >= 300:
                                    fp1.write(para)
                                    fp1.write('\n\n')
                                else:
                                    pass
                        else:
                            fp1.write("No document found")
                else:
                    fp1.write("No document found")
                    fp1.close()

                predicted_file = open(
                    f'/Users/shreyasarunesh/Desktop/Open_Domain_Question_Answering_Agent/Retriever/Evaluation/squad1_output/predicted_contexts/1-predicted_context/{iny + 51}_{i + 1}.txt',
                    'r')

                corpus = (predicted_file.read().split('\n\n'))
                while '' in corpus:
                    corpus.remove('')

                tokenized_corpus = [self.text_pre_processor.preprocess_text(doc) for doc in corpus]
                bm25 = BM25Plus(tokenized_corpus)

                # doc_scores = sorted(bm25.get_scores(tokenised_query), reverse=True)

                top_para = bm25.get_top_n(tokenized_query, corpus, n=num_para1)
                fc1 = open(
                    f'/Users/shreyasarunesh/Desktop/Open_Domain_Question_Answering_Agent/Retriever/Evaluation/squad1_output/predicted_contexts/1-top_n_pc/{iny + 51}_{i + 1}.txt',
                    'w')
                for para in top_para:
                    fc1.write(para)
                    fc1.write('\n\n')
                else:
                    pass

                if results2:

                    for id, _ in results2:

                        title = self.file_traverser.title_search(id)
                        title = title.strip()

                        corpus = (list(self.get_paragraphs(title))[:])
                        if corpus != '':
                            for para in corpus:
                                if len(para) >= 300:
                                    fp2.write(para)
                                    fp2.write('\n\n')
                                else:
                                    pass
                        else:
                            fp2.write("No document found")
                else:
                    fp2.write("No document found")
                    fp2.close()

                predicted_file = open(
                    f'/Users/shreyasarunesh/Desktop/Open_Domain_Question_Answering_Agent/Retriever/Evaluation/squad1_output/predicted_contexts/2-predicted_context/{iny + 51}_{i + 1}.txt',
                    'r')

                corpus = (predicted_file.read().split('\n\n'))
                while '' in corpus:
                    corpus.remove('')

                tokenized_corpus = [self.text_pre_processor.preprocess_text(doc) for doc in corpus]
                bm25 = BM25Plus(tokenized_corpus)

                # doc_scores = sorted(bm25.get_scores(tokenised_query), reverse=True)

                top_para = bm25.get_top_n(tokenized_query, corpus, n=num_para2)
                fc2 = open(
                    f'/Users/shreyasarunesh/Desktop/Open_Domain_Question_Answering_Agent/Retriever/Evaluation/squad1_output/predicted_contexts/2-top_n_pc/{iny + 51}_{i + 1}.txt',
                    'w')
                for para in top_para:
                    fc2.write(para)
                    fc2.write('\n\n')
                else:
                    pass

                if results3:

                    for id, _ in results3:

                        title = self.file_traverser.title_search(id)
                        title = title.strip()

                        corpus = (list(self.get_paragraphs(title))[:])
                        if corpus != '':
                            for para in corpus:
                                if len(para) >= 300:
                                    fp3.write(para)
                                    fp3.write('\n\n')
                                else:
                                    pass
                        else:
                            fp3.write("No document found")
                else:
                    fp3.write("No document found")
                    fp3.close()

                predicted_file = open(
                    f'/Users/shreyasarunesh/Desktop/Open_Domain_Question_Answering_Agent/Retriever/Evaluation/squad1_output/predicted_contexts/3-predicted_context/{iny + 51}_{i + 1}.txt',
                    'r')

                corpus = (predicted_file.read().split('\n\n'))
                while '' in corpus:
                    corpus.remove('')

                tokenized_corpus = [self.text_pre_processor.preprocess_text(doc) for doc in corpus]
                bm25 = BM25Plus(tokenized_corpus)

                # doc_scores = sorted(bm25.get_scores(tokenised_query), reverse=True)

                top_para = bm25.get_top_n(tokenized_query, corpus, n=num_para3)
                fc3 = open(
                    f'/Users/shreyasarunesh/Desktop/Open_Domain_Question_Answering_Agent/Retriever/Evaluation/squad1_output/predicted_contexts/3-top_n_pc/{iny + 51}_{i + 1}.txt',
                    'w')
                for para in top_para:
                    fc3.write(para)
                    fc3.write('\n\n')
                else:
                    pass

        end = time.time()
        temp = end - start
        print(temp)
        hours = temp // 3600
        temp = temp - 3600 * hours
        minutes = temp // 60
        seconds = temp - 60 * minutes
        print(
            'Done writing predicted contexts to files \n Finished in: {0} hrs: {1} mis: {2} sec '.format(hours, minutes,
                                                                                                         seconds))

    def take_input_from_user(self, question, num_article, num_para):
        title_list = []

        query = question

        tokenized_query = self.text_pre_processor.preprocess_text(query)

        s = time.time()

        query = query.strip()
        query1, query2 = self.identify_query_type(query)

        ranked_results = self.return_query_results(query1, 'simple')

        results = sorted(ranked_results.items(), key=lambda item: item[1], reverse=True)
        results = results[:num_article]
        res = results[:num_para]
        for id, _ in res:
            title_list.append(self.file_traverser.title_search(id))

        paragraphs = []

        for id, _ in results:
            title = self.file_traverser.title_search(id)

            corpus = (list(get_paragraphs(title))[:])

            if len(corpus) != 0:

                for para in corpus:
                    if len(para) >= 300:
                        paragraphs.append(para)

            else:
                wiki_corpus = wiki_wiki.page("draft:colonial rule in india - a chronology").summary
                if len(wiki_corpus) != 0:
                    paragraphs = []
                    for para in corpus:
                        paragraphs.append(para)

        tokenized_paras = [self.text_pre_processor.preprocess_text(doc) for doc in paragraphs]

        bm25 = BM25Plus(tokenized_paras)

        doc_scores = sorted(bm25.get_scores(tokenized_query), reverse=True)

        top_para = bm25.get_top_n(tokenized_query, paragraphs, n= num_para)

        return title_list, top_para, doc_scores
