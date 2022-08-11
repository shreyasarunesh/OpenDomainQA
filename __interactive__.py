



import OpenSSL.crypto

print("Loading search engine ...")

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
import torch
from transformers import AutoTokenizer, BertTokenizerFast
from tabulate import tabulate

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

    file_name = '/Retriever/Evaluation/questions'
    num_article = 10  # Value of K

    num_para = 20  # Number of para extracted in each page.

    html_tags = re.compile('&amp;|&apos;|&gt;|&lt;|&nbsp;|&quot;')
    with open('/Users/shreyasarunesh/Desktop/Open_Domain_Question_Answering_Agent/Retriever/wikipedia/stopwords.txt',
              'r') as f:
        stop_words = [word.strip() for word in f]

    with open('/Users/shreyasarunesh/Desktop/Open_Domain_Question_Answering_Agent/Retriever/wikipedia/stemwords.txt',
              'r') as f:
        stemmer = [word.strip() for word in f]

    with open(
            '/Users/shreyasarunesh/Desktop/Open_Domain_Question_Answering_Agent/Retriever/output_data'
            '/english_wiki_index/num_pages.txt',
            'r') as f:
        num_pages = float(f.readline().strip())

    text_pre_processor = TextPreProcessor(html_tags, stemmer, stop_words)
    file_traverser = FileTraverser()
    ranker = BM25(num_pages)
    query_results = QueryResults(file_traverser)
    run_query = RunQuery(text_pre_processor, file_traverser, ranker, query_results)

    temp = linecache.getline(
        '/Users/shreyasarunesh/Desktop/Open_Domain_Question_Answering_Agent/Retriever/output_data/english_wiki_index'
        '/id_title_map.txt',
        0)

    print('Loaded in', np.round(time.time() - s, 2), 'seconds')

    print('Starting Querying')

    # run_query.take_input_from_file(file_name, num_article, num_para)
    # run_query.get_wikiQA_predictions(num_results)
    # run_query.get_squad_predictions(num_article1,num_article2, num_article3, num_para1,num_para2, num_para3)

    while True:
        question = input('Enter Query:- ')


        def get_context(question, num_article, num_para):
            top_titles, top_para, doc_scores = run_query.take_input_from_user(question, num_article, num_para)

            return top_titles, top_para, doc_scores


        def get_answers(question, top_titles, top_paras_list, doc_scores, weight):

            # Load the fine-tuned model
            device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
            model_bert = Bert_QA.from_pretrained( '/Users/shreyasarunesh/Desktop/Open_Domain_Question_Answering_Agent'
                                                  '/Reader/model_output/BERT_finetuned_model')
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

                #     total_ans_score = start_score * end_score
                #     if top_total_score.sum() != 0:
                #         total_score = (weight * total_ans_score + (1 - weight) * ctx_score)
                #         if total_score[0] > top_total_score:
                #             top_total_score = total_score[0]
                #             top_start_score = start_score
                #             top_end_score = end_score
                #
                #         else:
                #             pass
                #     else:
                #         total_score = (weight * total_ans_score + (1 - weight) * ctx_score)
                #         top_total_score = total_score[0]
                #         top_start_score = start_score
                #
                # top_answer = (tokenizer.decode(input_ids[0][answer_start:(answer_end + 1)]))

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


        top_titles, top_paras_list, doc_scores = get_context(question, num_article, num_para)

        get_answers(question, top_titles, top_paras_list, doc_scores, weight=1.0)



#
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
# question = "who is the author of Harry Potter"
# top_sample_scores = [3, 10, 2]
#
# get_answers(question, top_sample_contexts, top_sample_scores, weight=1.0)
