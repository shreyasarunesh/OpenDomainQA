
'''
 *
 * Name                   :   F21MP- Open Domain Question Answering (ODQA) Agent.
 *
 * Description            :  This file illustrates the data preprocessing for both DrQA and BERT models.
 *
 * Author                 :   Shreyas Arunesh
 *
 *
 * Reference              : https://github.com/facebookresearch/DrQA
                          : https://github.com/castorini/bertserini
'''


import torch
import json
import numpy as np
from tqdm.auto import tqdm


# import BERT tokeniser for tokenisation
from transformers import BertTokenizerFast
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

# import DrQA tokeniser for DrQA
import gensim
import gensim.downloader as api

# Load the Glove word embedding in to the tokeniser.
word2vec = api.load("glove-wiki-gigaword-100")
embed_dim = 100
context_length = 360
question_length = 40

'''
 *
 *  Summary : This function convert a given sentences to tensors of a given length 
 *
 *  Args    : Param - sentence, length of the sentence, unk: is the initial and end token tensors.
 *
 *  Returns : returns tensor and mask.
 *
'''
def sentence_to_tensor(sentence, length, unk):
    array = np.zeros((length, embed_dim))
    mask = np.zeros(length)

    for i in range(min(len(sentence), length)):
        w = sentence[i]
        mask[i] = 1
        if len(w) > 2 and w[:2] == '##':
            w = w[2:]
        if w in word2vec.vocab:
            array[i] = word2vec.get_vector(w)
        else:
            array[i] = unk

    tensor = torch.from_numpy(array.astype(np.float32))
    mask = torch.from_numpy(mask.astype(np.float32))
    return tensor, mask

'''
 *
 *  Summary : This function converts the encodings to dataset while training DrQA
 *
 *  Args    : Param - This inherits torch.utils.data.Dataset 
 *
 *  Returns : returns training dataset.
 *
'''
class SquadDataset_train_DrQA(torch.utils.data.Dataset):

    def __init__(self, context_list, question_list, start_position, end_position):
        self.context_list = context_list
        self.question_list = question_list
        self.start_position = start_position
        self.end_position = end_position

        self.context_length = context_length
        self.question_length = question_length

        np.random.seed(42)
        self.unk = np.random.randn(embed_dim).astype(np.float32)

    def __getitem__(self, idx):
        context_tensor, context_mask = sentence_to_tensor(self.context_list[idx], self.context_length, self.unk)
        question_tensor, question_mask = sentence_to_tensor(self.question_list[idx], self.question_length, self.unk)
        start = min(self.start_position[idx], self.context_length - 1)
        end = min(self.end_position[idx], self.context_length - 1)
        return context_tensor, context_mask, question_tensor, question_mask, start, end

    def __len__(self):
        return len(self.context_list)


'''
 *
 *  Summary : This function converts the encodings to dataset while validating DrQA
 *
 *  Args    : Param - This inherits torch.utils.data.Dataset 
 *
 *  Returns : returns validation dataset.
 *
'''
class SquadDataset_val_DrQA(torch.utils.data.Dataset):

    def __init__(self, context_list, question_list, answers):
        self.context_list = context_list
        self.question_list = question_list

        self.context_length = context_length
        self.question_length = question_length
        self.answers = answers

        np.random.seed(42)
        self.unk = np.random.randn(embed_dim).astype(np.float32)

    def __getitem__(self, idx):
        context_tensor, context_mask = sentence_to_tensor(self.context_list[idx], self.context_length, self.unk)
        question_tensor, question_mask = sentence_to_tensor(self.question_list[idx], self.question_length, self.unk)

        return context_tensor, context_mask, question_tensor, question_mask

    def __len__(self):
        return len(self.context_list)


'''
 *
 *  Summary : This function converts the encodings to dataset while training BERT-base
 *
 *  Args    : Param - This inherits torch.utils.data.Dataset 
 *
 *  Returns : returns training dataset.
 *
'''
class SquadDataset_train_BERT(torch.utils.data.Dataset):

    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)


'''
 *
 *  Summary : This function converts the encodings to dataset while validating BERT-base.
 *
 *  Args    : Param - This inherits torch.utils.data.Dataset 
 *
 *  Returns : returns validation dataset.
 *
'''
class SquadDataset_val_BERT(torch.utils.data.Dataset):

    def __init__(self, encodings, answers):
        self.encodings = encodings
        self.answers = answers

    def __getitem__(self, idx):
        return_dict = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        return_dict['gold_text'] = self.answers[idx]['text']

        return return_dict

    def __len__(self):
        return len(self.encodings.input_ids)

"""
   *
   * Summary :    This is the class for preprocessing the dataset for training and validation and evaluation. 
   *
"""

class Preprocessing:
    def __int__(self):
        pass

    '''
     *
     *  Summary : This function helps to flatten the dataset and store them in lists.
     *
     *  Args    : Param - training dataset path, validation dataset path
     *
     *  Returns : returns list of context, question and answers.
     *
    '''
    def flatten(self, path, mode = 'train'):
        f = open(path, 'r')
        if mode == "train":
            data = json.load(f)
            num_article = len(data['data'])
            temp_data = data['data'][:(9 * num_article // 10)]
        elif mode == 'val':
            data = json.load(f)
            num_article = len(data['data'])
            temp_data = data['data'][(9 * num_article // 10):]
        elif mode == 'eval':
            data = json.load(f)
            num_article = len(data['data'])
            temp_data = data['data'][:]

        contexts = []
        questions = []
        answers = []

        for article in temp_data:
            for p in article['paragraphs']:
                context = p['context']
                for qa in p['qas']:
                    question = qa['question']

                    contexts.append(context)
                    questions.append(question)

                    if qa['is_impossible']:
                        answers.append({'answer_start': 0, 'text': ''})
                    else:
                        answers.append(qa['answers'][0])

        return contexts, questions, answers

    '''
     *
     *  Summary : This function helps to add the end char index for answers in the context.
                    
                Sometimes squad answers are off by a character or two, so this needs to be fixed. 
     *
     *  Args    : Param - answers and contexts
     *
     *  Returns : returns nothing but adds the start and end index to the dataset. 
     *
    '''
    def add_end_idx(self, answers, contexts):

        for answer, context in zip(answers, contexts):

            gold_text = answer['text']
            start_idx = answer['answer_start']
            end_idx = start_idx + len(gold_text)


            if context[start_idx:end_idx] == gold_text:
                answer['answer_end'] = end_idx
            elif context[start_idx - 1:end_idx - 1] == gold_text:
                answer['answer_start'] = start_idx - 1
                answer['answer_end'] = end_idx - 1  # When the gold label is off by one character
            elif context[start_idx - 2:end_idx - 2] == gold_text:
                answer['answer_start'] = start_idx - 2
                answer['answer_end'] = end_idx - 2  # When the gold label is off by two characters


    '''
     *
     *  Summary : This function performs the above preprocessing by adding the necessary pre and post tokens 
                to denote the sentence start and end positions and returns the list of tokens.
     *
     *  Args    : Param- contexts, questions and answers.
     *
     *  Returns : returns preprocessed context list and question list for validation set. 
                    Additionally returns start and end position list for training set
     *
    '''

    def preprocess(self, contexts, questions, answers, train=True):
        context_list = []
        question_list = []

        start_position = []
        end_position = []

        for i in tqdm(range(len(contexts))):
            encoding = tokenizer(contexts[i])
            token_list = tokenizer.convert_ids_to_tokens(encoding['input_ids'], skip_special_tokens=True)
            context_list.append(['[UNK]'] + token_list)

            if train:
                if answers[i]['text'] == '':
                    start_position.append(0)
                    end_position.append(0)
                else:
                    start = encoding.char_to_token(answers[i]['answer_start'])
                    end = encoding.char_to_token(answers[i]['answer_end'] - 1)

                    start_position.append(start)
                    end_position.append(end)

        for i in tqdm(range(len(questions))):
            encoding = tokenizer(questions[i])
            token_list = tokenizer.convert_ids_to_tokens(encoding['input_ids'], skip_special_tokens=True)

            question_list.append(token_list)

        if train:
            return context_list, question_list, start_position, end_position
        else:
            return context_list, question_list


    '''
     *
     *  Summary : This function is only for BERT-tokeniser to denote start and end  special tokens for a given encodings.
                    Ex: [CLS] Sentence A [SEP] Sentence B [SEP]
     *
     *  Args    : Param- encodings, answers, max_length=512
     *
     *  Returns : updates the token encoding with start and end position to differentiate the classes. 
     *
    '''

    def add_token_positions(self, encodings, answers, max_length=512):
        start_positions = []
        end_positions = []
        for i in range(len(answers)):
            if answers[i]['answer_start'] == 0 and answers[i]['answer_end'] == 0:
                start_positions.append(0)
                end_positions.append(0)
            else:
                start_positions.append(encodings.char_to_token(i, answers[i]['answer_start']))
                end_positions.append(encodings.char_to_token(i, answers[i]['answer_end'] - 1))
                # if None, the answer passage has been truncated
                if start_positions[-1] is None:
                    start_positions[-1] = max_length - 1
                if end_positions[-1] is None:
                    end_positions[-1] = max_length - 1
        encodings.update({'start_positions': start_positions, 'end_positions': end_positions})

    '''
     *
     *  Summary : This is the main function that calls above functions for preprocessing. 
                    Two settings: BERT and DrQA

     *
     *  Args    : Param- BERT= True or false
     *
     *  Returns : preprocessed training and validation dataset. 
     *
    '''

    def main(self, BERT=True):

        train_path = '/Users/shreyasarunesh/Desktop/Open_Domain_Question_Answering_Agent/squad2.0/train-v2.0.json'
        eval_path = '/Users/shreyasarunesh/Desktop/Open_Domain_Question_Answering_Agent/squad2.0/dev-v2.0.json'
        prepro = Preprocessing()

        # get training data
        print('#########Train data Preprocessing##########')
        train_contexts, train_questions, train_answers = prepro.flatten(train_path, mode = 'train')
        prepro.add_end_idx(train_answers, train_contexts)
        train_context_list, train_question_list, train_start_position, train_end_position = prepro.preprocess(
            train_contexts, train_questions, train_answers)

        # get validation data
        print('#########Validation data Preprocessing##########')
        val_contexts, val_questions, val_answers = prepro.flatten(train_path, mode = 'val')
        val_context_list, val_question_list = prepro.preprocess(val_contexts, val_questions, val_answers, train=False)


        # get evaluation data

        print('#########evaluation data Preprocessing##########')
        eval_contexts, eval_questions, eval_answers = prepro.flatten(eval_path, mode = 'eval')
        eval_context_list, eval_question_list = prepro.preprocess(eval_contexts, eval_questions, eval_answers, train=False)


        if BERT:
            max_length = 512
            train_encodings = tokenizer(train_contexts, train_questions, truncation=True, padding='max_length',
                                        max_length=max_length)
            prepro.add_token_positions(train_encodings, train_answers)

            val_encodings = tokenizer(val_contexts, val_questions, truncation=True, padding='max_length',
                                      max_length=max_length)
            eval_encodings = tokenizer(eval_contexts, eval_questions, truncation=True, padding='max_length',
                                      max_length=max_length)


            train_dataset = SquadDataset_train_BERT(train_encodings)
            val_dataset = SquadDataset_val_BERT(val_encodings, val_answers)
            eval_dataset = SquadDataset_val_BERT(eval_encodings, eval_answers)

            return train_dataset, val_dataset, eval_dataset

        else:

            train_dataset = SquadDataset_train_DrQA(train_context_list, train_question_list, train_start_position,
                                                    train_end_position)

            val_dataset = SquadDataset_val_DrQA(val_context_list, val_question_list, val_answers)
            eval_dataset = SquadDataset_val_DrQA(eval_context_list, eval_question_list, eval_answers)


            return train_dataset, val_dataset, eval_dataset

