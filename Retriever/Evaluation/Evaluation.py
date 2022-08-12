
import wikipedia
from Retriever.WikiSearching.RunQuery import *
from Retriever.WikiSearching.FileTraverser import *
import torch
import os
import numpy as np
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('bert-base-nli-mean-tokens')
from sklearn.metrics.pairwise import cosine_similarity


def get_ques_score1():
  predicted_titles_dir_list = [f for f in os.listdir('/Users/shreyasarunesh/Desktop/Open_Domain_Question_Answering_Agent/Retriever/Evaluation/squad1_output/predicted_contexts/1-predicted_context') if not f.startswith('.')]

  true_titles_dir_list = [f for f in os.listdir('/Users/shreyasarunesh/Desktop/Open_Domain_Question_Answering_Agent/Retriever/Evaluation/squad1_output/true_context') if not f.startswith('.')]
  predicted_titles_dir_list.sort(), true_titles_dir_list.sort()
  ques_bert_score= []
  for (pfile, tfile) in zip(predicted_titles_dir_list, true_titles_dir_list):
      pred_file = open('/Users/shreyasarunesh/Desktop/Open_Domain_Question_Answering_Agent/Retriever/Evaluation/squad1_output/predicted_contexts/1-predicted_context/'+ pfile, 'r')
      true_file = open('/Users/shreyasarunesh/Desktop/Open_Domain_Question_Answering_Agent/Retriever/Evaluation/squad1_output/true_context/'+ tfile, 'r')
      # print(pred_file, true_file)
      true_pred_cor = []
      true_pred_cor.append(true_file.read().lower())
      pred_contexts = pred_file.read().split('\n\n')
      while '' in pred_contexts:
          pred_contexts.remove('')
      for cont in pred_contexts:
          true_pred_cor.append(cont.lower())
      para_embeddings = model.encode(true_pred_cor)
      if para_embeddings.shape[0] >1:
        score = cosine_similarity([para_embeddings[0]], para_embeddings[1:])
        ques_bert_score.append(np.max(score))
      else:
        pass
  return ques_bert_score


def get_ques_score2():
  predicted_titles_dir_list = [f for f in os.listdir('/Users/shreyasarunesh/Desktop/Open_Domain_Question_Answering_Agent/Retriever/Evaluation/squad1_output/predicted_contexts/2-predicted_context') if not f.startswith('.')]

  true_titles_dir_list = [f for f in os.listdir('/Users/shreyasarunesh/Desktop/Open_Domain_Question_Answering_Agent/Retriever/Evaluation/squad1_output/true_context') if not f.startswith('.')]
  predicted_titles_dir_list.sort(), true_titles_dir_list.sort()
  ques_bert_score= []
  for (pfile, tfile) in zip(predicted_titles_dir_list, true_titles_dir_list):
      pred_file = open('/Users/shreyasarunesh/Desktop/Open_Domain_Question_Answering_Agent/Retriever/Evaluation/squad1_output/predicted_contexts/2-predicted_context/'+ pfile, 'r')
      true_file = open('/Users/shreyasarunesh/Desktop/Open_Domain_Question_Answering_Agent/Retriever/Evaluation/squad1_output/true_context/'+ tfile, 'r')
      # print(pred_file, true_file)
      true_pred_cor = []
      true_pred_cor.append(true_file.read().lower())
      pred_contexts = pred_file.read().split('\n\n')
      while '' in pred_contexts:
          pred_contexts.remove('')
      for cont in pred_contexts:
          true_pred_cor.append(cont.lower())
      para_embeddings = model.encode(true_pred_cor)
      if para_embeddings.shape[0] >1:
        score = cosine_similarity([para_embeddings[0]], para_embeddings[1:])
        ques_bert_score.append(np.max(score))
      else:
        pass
  return ques_bert_score

def get_ques_score3():
  predicted_titles_dir_list = [f for f in os.listdir('/Users/shreyasarunesh/Desktop/Open_Domain_Question_Answering_Agent/Retriever/Evaluation/squad1_output/predicted_contexts/3-predicted_context') if not f.startswith('.')]

  true_titles_dir_list = [f for f in os.listdir('/Users/shreyasarunesh/Desktop/Open_Domain_Question_Answering_Agent/Retriever/Evaluation/squad1_output/true_context') if not f.startswith('.')]
  predicted_titles_dir_list.sort(), true_titles_dir_list.sort()
  ques_bert_score= []
  for (pfile, tfile) in zip(predicted_titles_dir_list, true_titles_dir_list):
      pred_file = open('/Users/shreyasarunesh/Desktop/Open_Domain_Question_Answering_Agent/Retriever/Evaluation/squad1_output/predicted_contexts/3-predicted_context/'+ pfile, 'r')
      true_file = open('/Users/shreyasarunesh/Desktop/Open_Domain_Question_Answering_Agent/Retriever/Evaluation/squad1_output/true_context/'+ tfile, 'r')
      # print(pred_file, true_file)
      true_pred_cor = []
      true_pred_cor.append(true_file.read().lower())
      pred_contexts = pred_file.read().split('\n\n')
      while '' in pred_contexts:
          pred_contexts.remove('')
      for cont in pred_contexts:
          true_pred_cor.append(cont.lower())
      para_embeddings = model.encode(true_pred_cor)
      if para_embeddings.shape[0] >1:
        score = cosine_similarity([para_embeddings[0]], para_embeddings[1:])
        ques_bert_score.append(np.max(score))
      else:
        pass
  return ques_bert_score

def get_squad_avg_precision_at_k1(threshold):
    score = get_ques_score1()
    print('Squad-1 SSR BERT- Similarity score for Each Question(Max score)-{0}'.format(score))
    n_relavant = 0
    for rec in score:

        if rec >= threshold:
            n_relavant += 1

    precision = (n_relavant / (len(score)))*100

    return round(precision, 2)

def get_squad_avg_precision_at_k2(threshold):
    score = get_ques_score2()
    print('Squad-1 SSR BERT- Similarity score for Each Question(Max score)-{0}'.format(score))
    n_relavant = 0
    for rec in score:

        if rec >= threshold:
            n_relavant += 1

    precision = (n_relavant / (len(score)))*100

    return round(precision, 2)

def get_squad_avg_precision_at_k3(threshold):
    score = get_ques_score3()
    print('Squad-1 SSR BERT- Similarity score for Each Question(Max score)-{0}'.format(score))
    n_relavant = 0
    for rec in score:

        if rec >= threshold:
            n_relavant += 1

    precision = (n_relavant / (len(score)))*100

    return round(precision, 2)

def get_ques_score_MSR1():
  predicted_titles_dir_list = [f for f in os.listdir('/Users/shreyasarunesh/Desktop/Open_Domain_Question_Answering_Agent/Retriever/Evaluation/squad1_output/predicted_contexts/1-top_n_pc') if not f.startswith('.')]

  true_titles_dir_list = [f for f in os.listdir('/Users/shreyasarunesh/Desktop/Open_Domain_Question_Answering_Agent/Retriever/Evaluation/squad1_output/true_context') if not f.startswith('.')]
  predicted_titles_dir_list.sort(), true_titles_dir_list.sort()
  ques_bert_score= []
  for (pfile, tfile) in zip(predicted_titles_dir_list, true_titles_dir_list):
      pred_file = open('/Users/shreyasarunesh/Desktop/Open_Domain_Question_Answering_Agent/Retriever/Evaluation/squad1_output/predicted_contexts/1-top_n_pc/'+ pfile, 'r')
      true_file = open('/Users/shreyasarunesh/Desktop/Open_Domain_Question_Answering_Agent/Retriever/Evaluation/squad1_output/true_context/'+ tfile, 'r')
      # print(pred_file, true_file)
      true_pred_cor = []
      true_pred_cor.append(true_file.read().lower())
      pred_contexts = pred_file.read().split('\n\n')
      while '' in pred_contexts:
          pred_contexts.remove('')
      for cont in pred_contexts:
          true_pred_cor.append(cont.lower())
      para_embeddings = model.encode(true_pred_cor)
      if para_embeddings.shape[0] >1:
        score = cosine_similarity([para_embeddings[0]], para_embeddings[1:])
        ques_bert_score.append(np.max(score))
      else:
        pass
  return ques_bert_score

def get_ques_score_MSR2():
  predicted_titles_dir_list = [f for f in os.listdir('/Users/shreyasarunesh/Desktop/Open_Domain_Question_Answering_Agent/Retriever/Evaluation/squad1_output/predicted_contexts/2-top_n_pc') if not f.startswith('.')]

  true_titles_dir_list = [f for f in os.listdir('/Users/shreyasarunesh/Desktop/Open_Domain_Question_Answering_Agent/Retriever/Evaluation/squad1_output/true_context') if not f.startswith('.')]
  predicted_titles_dir_list.sort(), true_titles_dir_list.sort()
  ques_bert_score= []
  for (pfile, tfile) in zip(predicted_titles_dir_list, true_titles_dir_list):
      pred_file = open('/Users/shreyasarunesh/Desktop/Open_Domain_Question_Answering_Agent/Retriever/Evaluation/squad1_output/predicted_contexts/2-top_n_pc/'+ pfile, 'r')
      true_file = open('/Users/shreyasarunesh/Desktop/Open_Domain_Question_Answering_Agent/Retriever/Evaluation/squad1_output/true_context/'+ tfile, 'r')
      # print(pred_file, true_file)
      true_pred_cor = []
      true_pred_cor.append(true_file.read().lower())
      pred_contexts = pred_file.read().split('\n\n')
      while '' in pred_contexts:
          pred_contexts.remove('')
      for cont in pred_contexts:
          true_pred_cor.append(cont.lower())
      para_embeddings = model.encode(true_pred_cor)
      if para_embeddings.shape[0] >1:
        score = cosine_similarity([para_embeddings[0]], para_embeddings[1:])
        ques_bert_score.append(np.max(score))
      else:
        pass
  return ques_bert_score

def get_ques_score_MSR3():
  predicted_titles_dir_list = [f for f in os.listdir('/Users/shreyasarunesh/Desktop/Open_Domain_Question_Answering_Agent/Retriever/Evaluation/squad1_output/predicted_contexts/3-top_n_pc') if not f.startswith('.')]

  true_titles_dir_list = [f for f in os.listdir('/Users/shreyasarunesh/Desktop/Open_Domain_Question_Answering_Agent/Retriever/Evaluation/squad1_output/true_context') if not f.startswith('.')]
  predicted_titles_dir_list.sort(), true_titles_dir_list.sort()
  ques_bert_score= []
  for (pfile, tfile) in zip(predicted_titles_dir_list, true_titles_dir_list):
      pred_file = open('/Users/shreyasarunesh/Desktop/Open_Domain_Question_Answering_Agent/Retriever/Evaluation/squad1_output/predicted_contexts/3-top_n_pc/'+ pfile, 'r')
      true_file = open('/Users/shreyasarunesh/Desktop/Open_Domain_Question_Answering_Agent/Retriever/Evaluation/squad1_output/true_context/'+ tfile, 'r')
      # print(pred_file, true_file)
      true_pred_cor = []
      true_pred_cor.append(true_file.read().lower())
      pred_contexts = pred_file.read().split('\n\n')
      while '' in pred_contexts:
          pred_contexts.remove('')
      for cont in pred_contexts:
          true_pred_cor.append(cont.lower())
      para_embeddings = model.encode(true_pred_cor)
      if para_embeddings.shape[0] >1:
        score = cosine_similarity([para_embeddings[0]], para_embeddings[1:])
        ques_bert_score.append(np.max(score))
      else:
        pass
  return ques_bert_score

def get_squad_avg_precision_at_k_MSR1(threshold):
    score = get_ques_score_MSR1()
    print('Squad-1 MSR BERT- Similarity score for Each Question(Max score)-{0}'.format(score))
    n_relavant = 0
    for rec in score:

        if rec >= threshold:
            n_relavant += 1

    precision = (n_relavant / (len(score)))*100


    return round(precision, 2)

def get_squad_avg_precision_at_k_MSR2(threshold):
    score = get_ques_score_MSR2()
    print('Squad-1 MSR BERT- Similarity score for Each Question(Max score)-{0}'.format(score))
    n_relavant = 0
    for rec in score:

        if rec >= threshold:
            n_relavant += 1

    precision = (n_relavant / (len(score)))*100


    return round(precision, 2)

def get_squad_avg_precision_at_k_MSR3(threshold):
    score = get_ques_score_MSR3()
    print('Squad-1 MSR BERT- Similarity score for Each Question(Max score)-{0}'.format(score))
    n_relavant = 0
    for rec in score:

        if rec >= threshold:
            n_relavant += 1

    precision = (n_relavant / (len(score)))*100


    return round(precision, 2)

import time
start = time.time()
threshold = [0.25, 0.40, 0.50, 0.75, 0.90, 0.95, 1.0]
precision_score1 = []
precision_score_bm251 = []
for thre in threshold:
    score1 = get_squad_avg_precision_at_k1(float(thre))
    precision_score1.append(score1)
    score2 = get_squad_avg_precision_at_k_MSR1(float(thre))
    precision_score_bm251.append(score2)
    print('Precision of SINGLE STAGE RETRIEVER for threshold {0} is: {1}'.format(thre, score1))
    print('Precision OF MULTISTAGE RETRIEVER for threshold {0} is: {1}'.format(thre, score2))
    print('\n')


end = time.time()
temp = end-start
print(temp)
hours = temp//3600
temp = temp - 3600*hours
minutes = temp//60
seconds = temp - 60*minutes
print('%d:%d:%d' %(hours,minutes,seconds))

import time
start = time.time()
threshold = [0.25, 0.40, 0.50, 0.75, 0.90, 0.95, 1.0]
precision_score2 = []
precision_score_bm252 = []
for thre in threshold:
    score1 = get_squad_avg_precision_at_k2(float(thre))
    precision_score2.append(score1)
    score2 = get_squad_avg_precision_at_k_MSR2(float(thre))
    precision_score_bm252.append(score2)
    print('Precision of SINGLE STAGE RETRIEVER for threshold {0} is: {1}'.format(thre, score1))
    print('Precision OF MULTISTAGE RETRIEVER for threshold {0} is: {1}'.format(thre, score2))
    print('\n')

end = time.time()
temp = end-start
print(temp)
hours = temp//3600
temp = temp - 3600*hours
minutes = temp//60
seconds = temp - 60*minutes
print('%d:%d:%d' %(hours,minutes,seconds))

import time
start = time.time()
threshold = [0.25, 0.40, 0.50, 0.75, 0.90, 0.95, 1.0]
precision_score3 = []
precision_score_bm253 = []
for thre in threshold:
    score1 = get_squad_avg_precision_at_k3(float(thre))
    precision_score3.append(score1)
    score2 = get_squad_avg_precision_at_k_MSR3(float(thre))
    precision_score_bm253.append(score2)
    print('Precision of SINGLE STAGE RETRIEVER for threshold {0} is: {1}'.format(thre, score1))
    print('Precision OF MULTISTAGE RETRIEVER for threshold {0} is: {1}'.format(thre, score2))
    print('\n')

end = time.time()
temp = end-start
print(temp)
hours = temp//3600
temp = temp - 3600*hours
minutes = temp//60
seconds = temp - 60*minutes
print('%d:%d:%d' %(hours,minutes,seconds))


import matplotlib.pyplot as plt1
import matplotlib.pyplot as plt2
import matplotlib.pyplot as plt3

plt1.plot(threshold, precision_score1, color='r', label='SSR')
plt1.plot(threshold, precision_score_bm251, color='g', label='MSR')
plt1.title("K_Article=15 N_Para=20")
plt1.xlabel('Threshold')
plt1.ylabel('Precision@k_N.')
plt1.legend()
plt1.show()

plt2.plot(threshold, precision_score2, color='r', label='SSR')
plt2.plot(threshold, precision_score_bm252, color='g', label='MSR')
plt2.title("K_Article=25 N_Para=15")
plt2.xlabel('Threshold')
plt2.ylabel('Precision@k_N.')
plt2.legend()
plt2.show()

plt3.plot(threshold, precision_score3, color='r', label='SSR')
plt3.plot(threshold, precision_score_bm253, color='g', label='MSR')
plt3.title("K_Article=35 N_Para=10")
plt3.xlabel('Threshold')
plt3.ylabel('Precision@k_N.')
plt3.legend()
plt3.show()




























class Evaluation():
    def __int__(self):
        pass

    def extract_squad_questions(self):

        with open('/squad1.0/train-v2.0.json', 'r') as f:

            data = json.load(f)

            with open(
                    '/Users/shreyasarunesh/Desktop/Open_Domain_Question_Answering_Agent/Retriever/Evaluation/squad_output/squad_questions.txt',
                    'w') as p:

                for para in data['data']:

                    for i in para['paragraphs']:

                        for j in i['qas']:
                            p.write(j['question'] + '\n')

    """
    This function writes the true_titles from wikipedia api to the output file
    """

    def get_true_titles(self, Value_of_k):

        true_context_directory = os.listdir(
            '/Users/shreyasarunesh/Desktop/Open_Domain_Question_Answering_Agent/Retriever/Evaluation/squad_output/true_context')
        s = time.time()

        for iny, file in enumerate(true_context_directory):

            print('Extracting contexts and titles for true_context:' + str({iny + 1}))

            t = open(
                f'/Users/shreyasarunesh/Desktop/Open_Domain_Question_Answering_Agent/Retriever/Evaluation/squad_output/true_context/context{iny + 1}.txt',
                'r')
            data = t.read().split('\n\n')

            true_context, questions = data[0], data[1].split('\n')

            while '' in questions:
                questions.remove('')

            for i, query in enumerate(questions):

                print('Executing Query number:' + str(i + 1) + ' in the true_context file ' + str(iny + 1))

                file = open(
                    f'/Users/shreyasarunesh/Desktop/Open_Domain_Question_Answering_Agent/Retriever/Evaluation/squad_output/true_titles/question{iny + 1}_{i + 1}.txt',
                    'w')
                for items in (wikipedia.search(query, results=Value_of_k)):
                    file.write(wiki_wiki.page(items).title + '\n')
            e = time.time()

            print('Done writing true titles and contexts of SQUAD to files \n Finished in:' + str(e - s))

    '''
    this function writes the squad question and associated contexts to a output file
    '''

    def get_squad_context_ques(self):

        with open('/Users/shreyasarunesh/Desktop/Open_Domain_Question_Answering_Agent/squad1.1/train-v1.1.json',
                  'r') as f:
            data = json.load(f)

            for para in data['data']:
                for (i, para) in enumerate(para['paragraphs']):
                    file = open(
                        f'/Users/shreyasarunesh/Desktop/Open_Domain_Question_Answering_Agent/Retriever/Evaluation/squad1_output/true_context/context{i + 1}.txt',
                        'w')

                    true_context = para['context']

                    file.write(true_context.lower().strip())

                    file.write('\n\n')

                    for query in para['qas']:
                        qu = query['question']

                        file.write(qu.lower().strip())

                        file.write('\n')

    '''
    this function writes the true squad contexts to the files for evaluation
    1_1.txt, 1_2.txt, etc...
    '''

    def extract_true_squad_corpus(self):
        true_context_directory = [f for f in os.listdir(
            "/Users/shreyasarunesh/Desktop/Open_Domain_Question_Answering_Agent/Retriever/Evaluation/squad1_output/cont_ques")
                                  if not f.startswith('.')]
        true_context_directory.sort(key=lambda f: int(re.sub('\D', '', f)))

        for iny, file in enumerate(true_context_directory):
            t = open(
                f'/Users/shreyasarunesh/Desktop/Open_Domain_Question_Answering_Agent/Retriever/Evaluation/squad1_output/cont_ques/context{iny + 1}.txt',
                'r')
            data = t.read().split('\n\n')
            true_context, questions = data[0], data[1].split('\n')
            while '' in questions:
                questions.remove('')
            for i, query in enumerate(questions):
                fp = open(
                    f'/Users/shreyasarunesh/Desktop/Open_Domain_Question_Answering_Agent/Retriever/Evaluation/squad1_output/true_context/{iny + 1}_{i + 1}.txt',
                    'w')
                fp.write(true_context)

    """
    This function writes the true_context for a given file of questions used for evaluation of a custom question.
    The true context considered here is wikipedia API generated titles for a given question.
    """

    def get_true_title_for_given_questions(self, value_of_k):
        fp = open(
            '/Users/shreyasarunesh/Desktop/Open_Domain_Question_Answering_Agent/Retriever/Evaluation/questions',
            'r')
        questions = fp.read().split('\n')
        for i, query in enumerate(questions):
            true_content = open(
                f'/Users/shreyasarunesh/Desktop/Open_Domain_Question_Answering_Agent/Retriever/Evaluation/trueoutput/question{i + 1}.txt',
                'w')
            for items in (wikipedia.search(query, results=value_of_k)):
                true_content.write(wiki_wiki.page(items).title)
                true_content.write('\n')

    """
    This function return the list of average cosine similarity score for each question
    comparing it with the top N true_titles.

    """

    def get_questions_similarity_score(self, pre_file, true_file):

        p_list = []

        t_list = []

        for p in pre_file:
            p_list.append(p.strip())

        for t in true_file:
            t_list.append(t.strip())

        titles_score = []

        for p in p_list:
            predict = p.lower().strip()

            cosine = []
            for t in t_list:
                true = t.lower().strip()

                co = self.get_cosine(predict, true)
                cosine.append(co)
            titles_score.append(round(max(cosine), 2))

        question_score = np.average(titles_score)

        return question_score

    def get_score_at_k(self):

        predicted_titles_dir_list = [f for f in os.listdir(os.getcwd() + "/predictedoutput") if not f.startswith('.')]

        true_titles_dir_list = [f for f in os.listdir(os.getcwd() + "/trueoutput") if not f.startswith('.')]

        Questions_score_at_k = []

        for (pfile, tfile) in zip(predicted_titles_dir_list, true_titles_dir_list):
            pred_file = (open(os.getcwd() + "/predictedoutput/" + pfile, 'r'))
            true_file = (open(os.getcwd() + "/trueoutput/" + tfile, 'r'))
            score = self.get_questions_similarity_score(pred_file, true_file)
            Questions_score_at_k.append(round(score, 2))
        return Questions_score_at_k

    """
    This function returns the Overall Precision of all the questions in the dataset
    """

    def get_overall_precision_for_given_questions(self, threshold):

        score = evaluation.get_score_at_k()

        n_relavant = 0
        for rec in score:

            if rec >= threshold:
                n_relavant += 1

        precision = (n_relavant / (len(score))) * 100

        print('Cosine Similarity Score of each question: {0} \n Overall Precision at k: {1}'.format(score, precision))

        return round(precision, 2)

    """
    this function gives the average squad dataset precision for K=20 and for given threshold value. 
    """

    def get_squad_ques_bert_score(self):
        predicted_titles_dir_list = [f for f in os.listdir('/content/drive/MyDrive/Colab Notebooks/predicted_context')
                                     if not f.startswith('.')]

        true_titles_dir_list = [f for f in os.listdir('/content/drive/MyDrive/Colab Notebooks/true_context_test') if
                                not f.startswith('.')]
        ques_bert_score = []
        for (pfile, tfile) in zip(predicted_titles_dir_list, true_titles_dir_list):
            pred_file = open('/content/drive/MyDrive/Colab Notebooks/predicted_context/' + pfile, 'r')
            true_file = open('/content/drive/MyDrive/Colab Notebooks/true_context_test/' + tfile, 'r')
            true_pred_cor = []
            true_pred_cor.append(true_file.read().lower())
            pred_contexts = pred_file.read().split('\n\n')
            while '' in pred_contexts:
                pred_contexts.remove('')
            for cont in pred_contexts:
                true_pred_cor.append(cont.lower())
            para_embeddings = model.encode(true_pred_cor)
            if para_embeddings.shape[0] > 1:
                score = cosine_similarity([para_embeddings[0]], para_embeddings[1:])
                ques_bert_score.append(np.max(score))
            else:
                pass
        return ques_bert_score

    def get_squad_avg_precision_at_k20(self, threshold):
        score = self.get_squad_ques_bert_score()
        n_relavant = 0
        for rec in score:

            if rec >= threshold:
                n_relavant += 1

        precision = (n_relavant / (len(score))) * 100

        print('precision of squad dataset for k=20:{0}'.format(precision))

        return round(precision, 2)


if __name__ == '__main__':
    evaluation = Evaluation()
    evaluation.extract_true_squad_corpus()


    # tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/bert-base-nli-mean-tokens')
    # model = AutoModel.from_pretrained('sentence-transformers/bert-base-nli-mean-tokens')
    # value_of_k = 10  # Make sure the value of K is same for predicted_titles in WikiSearch main file.
    #
    # threshold = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
    # evaluation = Evaluation()
    #
    # # evaluation.get_overall_precision_for_given_questions(threshold=0.7)
    #
    # precision_score = []
    # for thre in threshold:
    #     score = evaluation.get_squad_avg_precision_at_k20(float(thre))
    #     precision_score.append(score)
    #     print('Precision of retriever model for SQUAD dataset with threshold {0} is: {1}'.format(thre, score))
    # plt.plot(threshold, precision_score)
    # plt.xlabel('Threshold')
    # plt.ylabel('Precision for SQUAD dataset')
    # plt.show()


# import bs4
# import requests
# import unicodedata
# import re
#
# def get_paragraphs(page_name):
#
#     r = requests.get('https://en.wikipedia.org/api/rest_v1/page/html/{0}'.format(page_name))
#     soup = bs4.BeautifulSoup(r.content, 'html.parser')
#     html_paragraphs = soup.find_all('p')
#
#     for p in html_paragraphs:
#         cleaned_text = re.sub('(\[[0-9]+\])', '', unicodedata.normalize('NFKD', p.text)).strip()
#         if cleaned_text:
#             yield cleaned_text
#
# # print(list(get_paragraphs('capital punishment in the united kingdom'))[3])




