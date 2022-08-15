
'''
 *
 * Name                   :   F21MP- Open Domain Question Answering (ODQA) Agent.
 *
 * Description            :  This file illustrates the interactive session to get answer to Open domain questions from
                                this project system. Just execute this file to start interacting with the system.
 *
 * Author                 :   Shreyas Arunesh
 *
 *
 * Reference              : https://github.com/facebookresearch/DrQA
                          : https://github.com/castorini/bertserini
'''

print("Please wait Loading search engine ...")

import time

s = time.time()
import tqdm
import numpy as np
from Retriever.WikiSearching.RunQuery import *
from Retriever.WikiSearching.FileTraverser import *
from Retriever.WikiSearching.QueryResults import *
from Retriever.WikiSearching.BM25 import *

from Reader.BERT import *
from Reader.BERT import Bert_QA
from Reader.DrQA import DrQA
import torch
from transformers import AutoTokenizer, BertTokenizerFast
from tabulate import tabulate

from Stemmer import Stemmer
from nltk.corpus import stopwords

# Define the bert tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

from Retriever.WikiIndexing.TextPreProcessor import *
import linecache

"""
Search query types:

type1: Simple- who is johnny depp
type2: field query- t:who is jonny c:who is depp
type3: Mixed - what is capital of scotland b:capital of scotland
"""

if __name__ == '__main__':

    '''
     *
     *  Summary : This block loads the Search engine. It takes ~ 45 seconds for loading.
     *
     *
    '''

    file_name = '/Retriever/Evaluation/questions'
    num_article = 10  # Value of K
    num_para = 20  # Number of para extracted in each page.

    html_tags = re.compile('&amp;|&apos;|&gt;|&lt;|&nbsp;|&quot;')
    stop_words = (set(stopwords.words("english")))
    stemmer = Stemmer('english')

    text_pre_processor = TextPreProcessor(html_tags, stemmer, stop_words)
    file_traverser = FileTraverser()
    ranker = BM25(num_pages)
    query_results = QueryResults(file_traverser)
    run_query = RunQuery(text_pre_processor, file_traverser, ranker, query_results)

    temp = linecache.getline('../Dataset/output_data/id_title_map.txt', 0)

    print('Loaded in', np.round(time.time() - s, 2), 'seconds')

    print('Starting Querying')

    # run_query.take_input_from_file(file_name, num_article, num_para)
    # run_query.get_wikiQA_predictions(num_results)
    # run_query.get_squad_predictions(num_article1,num_article2, num_article3, num_para1,num_para2, num_para3)

    '''
     *
     *  Summary : This block takes the user open domain query and displays the top predicted answer and a table to top answers.
     *
     *
    '''

    while True:
        question = input('Enter Query:- ')

        '''
             *
             *  Summary : This function retrieves the top_K_N paragraphs for a given question.
             *
             *  Args    : Param - question, number of articles, number paragraphs.
             *
             *  Returns : lists of top titles, top paragraphs and document scores
             *
        '''
        def get_context(question, num_article, num_para):
            top_titles, top_para, doc_scores = run_query.take_input_from_user(question, num_article, num_para)

            return top_titles, top_para, doc_scores
        '''
             *
             *  Summary : This function extracts the answer for all the retrieved paragraphs and gets the top predicted answer.
                            
             *
             *  Args    : Param - question, top_titles, top_paras_list, doc_scores, weight. 
                             Weight(μ) = 1 is the hyperparameter for fully connected system in the equation:
                                S = (1 − μ) · SRetriever score + μ · SReader score
             *
             *  Returns : lists of top titles, top paragraphs and document scores
             *
        '''

        def get_answers(question, top_titles, top_paras_list, doc_scores, weight):

            # Load the fine-tuned model
            device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
            model_bert = Bert_QA.from_pretrained( '../Reader_model_output/BERT_finetuned_model_on_SQuAD_V 2.0')
            model_bert.to(device)
            model_bert.eval()

            top_total_score = torch.zeros(size=torch.Size([]))
            top_start_score = 0
            top_end_score = 0

            predicted_ans_list = []
            predicted_title_list = []

            with torch.no_grad():
                for top_title, context, ctx_score in zip(top_titles, top_paras_list, doc_scores):
                    predicted_title_list.append(top_title)
                    temp_encoding = tokenizer(context, question, truncation=True, padding=True)
                    input_ids = torch.LongTensor(temp_encoding['input_ids']).unsqueeze(0).to(device)
                    token_type_ids = torch.LongTensor(temp_encoding['token_type_ids']).unsqueeze(0).to(device)
                    attention_mask = torch.LongTensor(temp_encoding['attention_mask']).unsqueeze(0).to(device)

                    start_score, end_score = model_bert.get_scores(input_ids, attention_mask=attention_mask,
                                                                   token_type_ids=token_type_ids)

                    start_score = start_score.squeeze(0).cpu()
                    end_score = end_score.squeeze(0).cpu()

                    answer_start = torch.argmax(start_score).item()
                    answer_end = torch.argmax(end_score).item()

                    predicted_ans_list.append(tokenizer.decode(input_ids[0][answer_start:(answer_end + 1)]))

                    total_ans_score = start_score * end_score
                    if top_total_score.sum() != 0:
                        total_score = (weight * total_ans_score + (1 - weight) * ctx_score)
                        if total_score[0] > top_total_score:
                            top_total_score = total_score[0]
                            top_start_score = start_score
                            top_end_score = end_score

                        else:
                            pass
                    else:
                        total_score = (weight * total_ans_score + (1 - weight) * ctx_score)
                        top_total_score = total_score[0]
                        top_start_score = start_score

                top_answer = (tokenizer.decode(input_ids[0][answer_start:(answer_end + 1)]))

                answer = []
                for i in predicted_ans_list:
                    if i != '[CLS]' and i!= "":
                        answer.append(i)

                # print("Question: {0} \nTop Predicted Answer: {1}".format(question, top_answer))
                print("Question: {0} \nTop Predicted Answer: {1}".format(question, answer[0]))
                print()

                headers = ["Top Answers", "Top Wikipedia Titles"]

                print(tabulate(zip(predicted_ans_list, predicted_title_list), headers, tablefmt="github"))
                print()
                print()


        '''
             *
             *  Summary : This first line of code is Retriever and second line is reader that calls all
                            the above functions to perform Open domain question answering..

             *
        '''
        top_titles, top_paras_list, doc_scores = get_context(question, num_article, num_para)

        get_answers(question, top_titles, top_paras_list, doc_scores, weight=1.0)



'''
     *
     *  Summary : This block of code is to test the reader model.

     *
'''
# """Reader Experiments"""
#
# from Reader.BERT import *
# from Reader.BERT import Bert_QA
# from transformers import AutoTokenizer, BertTokenizerFast
# from tabulate import tabulate
#
# tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
# import torch
#
#
# def get_answers(question, top_paras_list, doc_scores, weight):
#     # Load the fine-tuned model
#     device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
#     model_bert = Bert_QA.from_pretrained(
#         '/Users/shreyasarunesh/Desktop/Open_Domain_Question_Answering_Agent/Reader/model_output/BERT_finetuned_model')
#     model_bert.to(device)
#     model_bert.eval()
#
#     top_total_score = torch.zeros(size=torch.Size([]))
#     top_start_score = 0
#     top_end_score = 0
#
#     predicted_ans_list = []
#
#     lis = []
#
#     with torch.no_grad():
#         for context, ctx_score in zip(top_paras_list, doc_scores):
#             temp_encoding = tokenizer(context, question, truncation=True, padding=True)
#             input_ids = torch.LongTensor(temp_encoding['input_ids']).unsqueeze(0).to(device)
#             token_type_ids = torch.LongTensor(temp_encoding['token_type_ids']).unsqueeze(0).to(device)
#             attention_mask = torch.LongTensor(temp_encoding['attention_mask']).unsqueeze(0).to(device)
#
#             start_score, end_score = model_bert.get_scores(input_ids, attention_mask=attention_mask,
#                                                            token_type_ids=token_type_ids)
#
#             start_score = start_score.squeeze(0).cpu()
#             end_score = end_score.squeeze(0).cpu()
#
#             answer_start = torch.argmax(start_score).item()
#             answer_end = torch.argmax(end_score).item()
#
#             predicted_ans_list.append(tokenizer.decode(input_ids[0][answer_start:(answer_end + 1)]))
#
#             total_ans_score = start_score * end_score
#             if top_total_score.sum() != 0:
#                 total_score = (weight * total_ans_score + (1 - weight) * ctx_score)
#                 lis.append(total_score)
#
#                 if total_score[0] > top_total_score:
#                     top_total_score = total_score[0]
#                     top_start_score = start_score
#                     top_end_score = end_score
#
#                 else:
#                     pass
#             else:
#                 total_score = (weight * total_ans_score + (1 - weight) * ctx_score)
#                 top_total_score = total_score[0]
#                 top_start_score = start_score
#                 lis.append(total_score)
#
#         answer_start = torch.argmax(top_start_score).item()
#         answer_end = torch.argmax(top_end_score).item()
#
#         top_answer = (tokenizer.decode(input_ids[0][answer_start:(answer_end + 1)]))
#
#         answer = 0
#         for i in predicted_ans_list:
#             if i != '[CLS]':
#                 answer = i
#
#
#         print(predicted_ans_list)
#
#         print("Question: {0} \nTop Predicted Answer: {1}".format(question, top_answer))
#         print("Question: {0} \nTop Predicted Answer: {1}".format(question, answer))
#         print()
#
#
# top_sample_contexts = [""" Mount Olympus is the highest mountain in Greece. It is part of the Olympus massif near
#           the Gulf of Thérmai of the Aegean Sea, located in the Olympus Range on the border between
#           Thessaly and Macedonia, between the regional units of Pieria and Larissa, about 80 km (50 mi)
#           southwest from Thessaloniki. Mount Olympus has 52 peaks and deep gorges. The highest peak,
#           Mytikas, meaning "nose", rises to 2917 metres (9,570 ft). It is one of the
#           highest peaks in Europe in terms of topographic prominence. """,
#
#                        """Harry Potter is a series of seven fantasy novels written by British author, J. K. Rowling. The novels chronicle the lives of a young wizard,
#           Harry Potter, and his friends Hermione Granger and Ron Weasley, all of whom are students at Hogwarts School of Witchcraft and Wizardry.
#           The main story arc concerns Harry's struggle against Lord Voldemort, a dark wizard who intends to become immortal, overthrow the wizard
#           governing body known as the Ministry of Magic and subjugate all wizards and Muggles (non-magical people). Since the release of the first novel,
#           Harry Potter and the Philosopher's Stone, on 26 June 1997, the books have found immense popularity, positive reviews, and commercial success worldwide.
#           They have attracted a wide adult audience as well as younger readers and are often considered cornerstones of modern young adult literature.[2]
#           As of February 2018, the books have sold more than 500 million copies worldwide, making them the best-selling book series in history, and have been translated
#           into eighty languages.""",
#
#                        """The COVID-19 pandemic, also known as the coronavirus pandemic, is an ongoing pandemic of coronavirus disease 2019 (COVID-19)
#           caused by severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2). It was first identified in December 2019 in Wuhan, China.
#           The World Health Organization declared the outbreak a Public Health Emergency of International Concern in January 2020 and a pandemic
#           in March 2020. As of 6 February 2021, more than 105 million cases have been confirmed, with more than 2.3 million deaths attributed to COVID-19.
#           Symptoms of COVID-19 are highly variable, ranging from none to severe illness. The virus spreads mainly through the air when people are
#           near each other."""]
#
# questions = [
#            "How many metres is Olympus?",
#            "Where Olympus is near?",
#            "How far away is Olympus from Thessaloniki?"
#           ]
#
# questions=  [
#            "Who wrote Harry Potter's novels?",
#            "Who are Harry Potter's friends?",
#            "Who is the enemy of Harry Potter?",
#            "What are Muggles?",
#            "Which is the name of Harry Poter's first novel?",
#            "When did the first novel release?",
#            "Who was attracted by Harry Potter novels?",
#            "How many languages Harry Potter has been translated into? "
#           ]
#
# questions  = [
#            "What is COVID-19?",
#            "What is caused by COVID-19?",
#            "How many cases have been confirmed from COVID-19?",
#            "How many deaths have been confirmed from COVID-19?",
#            "How is COVID-19 spread?",
#            "How long can an infected person remain infected?",
#            "Can a infected person spread the virus even if they don't have symptoms?",
#            "What do elephants eat?"
#           ]
# top_sample_scores = [1, 3, 4]
# get_answers(question, top_sample_contexts, top_sample_scores, weight=1.0)

