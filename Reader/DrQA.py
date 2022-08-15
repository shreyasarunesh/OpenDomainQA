
'''
 *
 * Name                   :   F21MP- Open Domain Question Answering (ODQA) Agent.
 *
 * Description            :  This file illustrates the implementation of DrQA model
                            for training, validation and evaluation and evaluation.
 *
 * Author                 :   Shreyas Arunesh
 *
 *
 * Reference              : https://github.com/facebookresearch/DrQA

'''


import os

import torch
import torch.nn as nn
import torch.optim as optim
import gensim
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt1
import logging
import time
import gensim.downloader as api

word2vec = api.load("glove-wiki-gigaword-100")

from Reader.Preprocessing import *
from Reader.Eval_metrics  import *

prepro = Preprocessing()
eval = Evaluation()
from torch.utils.data import DataLoader, BatchSampler, SequentialSampler


class DrQA(nn.Module):

    def __init__(self, num_layers, input_size, hidden_size, dropout=0.5):
        '''
         *
         *  Summary : This is DrQA model constructor class to create DrQA model consisting of 2 BiLSTM models using
                        Pytorch.
         *
         *  Args    : Param - number of layers input size, hidden size, dropout
         *
         *
        '''
        super().__init__()
        self.hidden_size = hidden_size
        self.context_rnn = nn.LSTM(input_size=2 * input_size, hidden_size=hidden_size, num_layers=num_layers, bias=True,
                                   batch_first=True, dropout=dropout, bidirectional=True)
        self.question_rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, bias=True,
                                    batch_first=True, dropout=dropout, bidirectional=True)
        self.att_vec = nn.Parameter(torch.randn(2 * hidden_size), requires_grad=True)
        self.criterion = nn.CrossEntropyLoss()
        self.W_s = nn.Parameter(torch.randn(2 * hidden_size, 2 * hidden_size), requires_grad=True)
        self.W_e = nn.Parameter(torch.randn(2 * hidden_size, 2 * hidden_size), requires_grad=True)
        self.fc = nn.Linear(input_size, input_size)

    '''
     *
     *  Summary : This is DrQA model simple attention layer.
     *
     *  Args    : Param - input tensors, input mask and attention vector.
     *
     *  Returns : returns attention output.
     *
    '''
    def get_att_output(self, input_tensor, input_mask, att_vec):
        # batch,length
        temp = torch.matmul(input_tensor, att_vec)
        # batch,length
        temp = temp + (input_mask + 1e-45).log()
        # batch,length
        weight = nn.Softmax(dim=1)(temp)
        # batch,length,1
        weight = weight.unsqueeze(-1)

        # batch,dim
        output = torch.sum(input_tensor * weight, dim=1)
        return output


    '''
     *
     *  Summary : This is DrQA model implementation for forward propagation. 
                    1. sets the two BiLSTM model for question and context.
                    2. calculates the gradient loss for both start and end logits.
     *
     *  Args    : Param - context tensor, context mask, 
                        question tensor, question mask, 
                        start_tensor, end_tensor
     *
     *  Returns : loss.
     *
    '''
    def forward(self, context_tensor, context_mask, question_tensor, question_mask, start_tensor, end_tensor):
        q = nn.ReLU()(self.fc(question_tensor))
        c = nn.ReLU()(self.fc(context_tensor))

        # batch,seq_c,seq_q
        temp = torch.matmul(c, torch.transpose(q, 1, 2))
        temp_logits = temp + (question_mask.unsqueeze(1) + 1e-45).log()

        # batch,seq_c,seq_q
        weight = nn.Softmax(2)(temp_logits)

        # batch,seq_c,dim
        c_align = torch.matmul(weight, question_tensor)

        context_tensor = torch.cat((context_tensor, c_align), 2)

        # batch,length,dim
        question_output = self.question_rnn(question_tensor)[0]
        # batch,length,dim
        context_output = self.context_rnn(context_tensor)[0]

        question_vec = self.get_att_output(question_output, question_mask, self.att_vec)

        start_vec = torch.matmul(question_vec, self.W_s)
        end_vec = torch.matmul(question_vec, self.W_e)

        # batch,length
        start_logits = torch.matmul(context_output, start_vec.unsqueeze(2)).squeeze(-1)
        start_logits = start_logits + (context_mask + 1e-45).log()

        end_logits = torch.matmul(context_output, end_vec.unsqueeze(2)).squeeze(-1)
        end_logits = end_logits + (context_mask + 1e-45).log()

        start_loss = self.criterion(start_logits, start_tensor)
        end_loss = self.criterion(end_logits, end_tensor)
        loss = start_loss + end_loss

        return loss

    '''
      *
      *  Summary : This function predicts the start and end score for each question.
      *
      *  Args    : Param - context tensor, context mask, question tensor, question mask
      *
      *  Returns : start score and end score. 
      *
     '''

    def get_scores(self, context_tensor, context_mask, question_tensor, question_mask):
        q = nn.ReLU()(self.fc(question_tensor))
        c = nn.ReLU()(self.fc(context_tensor))

        # batch,seq_c,seq_q
        temp = torch.matmul(c, torch.transpose(q, 1, 2))
        temp_logits = temp + (question_mask.unsqueeze(1) + 1e-45).log()

        # batch,seq_c,seq_q
        weight = nn.Softmax(2)(temp_logits)
        c_align = torch.matmul(weight, question_tensor)

        context_tensor = torch.cat((context_tensor, c_align), 2)

        # batch,length,dim
        question_output = self.question_rnn(question_tensor)[0]
        context_output = self.context_rnn(context_tensor)[0]

        question_vec = self.get_att_output(question_output, question_mask, self.att_vec)

        # batch,dim
        start_vec = torch.matmul(question_vec, self.W_s)
        end_vec = torch.matmul(question_vec, self.W_e)

        start_logits = torch.matmul(context_output, start_vec.unsqueeze(2)).squeeze(
            -1)  # [batch,length,dim]*[batch,dim,1] -> [batch,length,1]
        start_logits = start_logits + (context_mask + 1e-45).log()

        end_logits = torch.matmul(context_output, end_vec.unsqueeze(2)).squeeze(-1)
        end_logits = end_logits + (context_mask + 1e-45).log()

        start_score = nn.Softmax(dim=1)(start_logits)
        end_score = nn.Softmax(dim=1)(end_logits)

        return start_score, end_score


class Train_Validate:
    def __int__(self):
        pass

    '''
      *
      *  Summary : This is the main function for training and validation using Adam optimiser in back propagation.
      *
      *  Args    : Param - epochs = 50 
                    optimizer = Adam 
                     train_loader, validation_loader, 
                     model = DrQA model 
                     device = GUP- CUDA enabled 
      *
      *  Returns : lists of (train_loss , val_loss , em_score , f1-score) for each epochs
      *
     '''

    def train(self, epochs, optimizer, train_loader, validation_loader, model, device):

        iter_counter = 0
        best_f1 = 0
        avg_loss = 0
        train_loss = []
        val_loss = []
        em_lis = []
        f1_lis = []

        s = time.time()
        for epoch_ind, epoch in enumerate(range(epochs)):

            # training
            loss_of_epoch = []
            print('######## Training {0} Epochs ########'.format(epoch_ind + 1))
            log.info('######## Training {0} Epopchs ########'.format(epoch_ind + 1))

            for context_tensor, context_mask, question_tensor, question_mask, start_tensor, end_tensor in train_loader:
                model.train()
                optimizer.zero_grad()
                loss = model(context_tensor.to(device), context_mask.to(device), question_tensor.to(device),
                             question_mask.to(device), start_tensor.to(device), end_tensor.to(device))
                loss.backward()
                optimizer.step()

                iter_counter += 1

                avg_loss += loss.item()
                loss_of_epoch.append(loss.item())

                if iter_counter % 250 == 0:
                    avg_loss /= 250
                    print('iter %d' % iter_counter)
                    print('loss %.5f' % avg_loss)
                    avg_loss = 0
                    print()
                if iter_counter % 1000 == 0:
                    em, f1 = eval.evaluate(model, val_dataset, BERT=False)
                    print('EM: %.5f' % em)
                    print('F1: %.5f' % f1)

                    if f1 > best_f1:
                        best_f1 = f1
                        best_em = em
                        torch.save(model.state_dict(), "/content/drive/MyDrive/ODQA/outputs/DrQA_trained.pth")
                        print("Best model")

                        log.info("Best model of batch_index: {0} EM-Score:{1} & F1-score:{2}".format(len(train_loader),
                                                                                                     best_em, best_f1))

            print(loss_of_epoch)

            train_loss.append(np.average(loss_of_epoch))

            print(train_loss[-1])

            loss_of_epoch = []

            # validating
            model.eval()
            print("########## Validating {0} Epochs ##########".format(epoch_ind + 1))
            log.info("########## Validating {0} Epochs ##########".format(epoch_ind + 1))
            for context_tensor, context_mask, question_tensor, question_mask, start_tensor, end_tensor in validation_loader:
                optimizer.zero_grad()
                loss = model(context_tensor.to(device), context_mask.to(device), question_tensor.to(device),
                             question_mask.to(device), start_tensor.to(device), end_tensor.to(device))

                iter_counter += 1

                avg_loss += loss.item()
                loss_of_epoch.append(loss.item())
                avg_loss = 0

            print(loss_of_epoch)
            val_loss.append(np.average(loss_of_epoch))
            loss_of_epoch = []
            print(val_loss[-1])

            em_lis.append(best_em)
            f1_lis.append(best_f1)

            print(
                "Epoch: {0}/{1} Train loss: {2} Validation_loss: {3} Em-Score: {4} F1-Score: {5}".format(epoch_ind + 1,
                                                                                                         epochs,
                                                                                                         train_loss,
                                                                                                         val_loss,
                                                                                                         em_lis[-1],
                                                                                                         f1_lis[-1]))
            log.info(
                "Epoch: {0}/{1} Train loss: {2} Validation_loss: {3} Em-Score: {4} F1-Score: {5}".format(epoch_ind + 1,
                                                                                                         epochs,
                                                                                                         train_loss,
                                                                                                         val_loss,
                                                                                                         em_lis[-1],
                                                                                                         f1_lis[-1]))
            print()

        e = time.time()
        temp = e - s
        h = temp // 3600
        m = temp % 3600 // 60
        s = temp % 3600 % 60

        em, f1 = eval.evaluate(model_DrQA, val_dataset, BERT=False)
        print('EM: %.5f' % em)
        print('F1: %.5f' % f1)
        if f1 > best_f1:
            best_f1 = f1
            best_em = em
            # Save model
            torch.save(model.state_dict(), "../Reader_model_output/DrQA_trained_SQuAD-1.pth")

            print('best model!')
            print("Best models of batch_index: {0} EM-Score:{1} & F1-score:{2}".format(len(train_loader), best_em,
                                                                                       best_f1))

            log.info("Best models of batch_index: {0} EM-Score:{1} & F1-score:{2}".format(len(train_loader), best_em,
                                                                                          best_f1))

        print("Final model EM-Score: {0} F1-Score: {1} Total time to train & val: {2}hours; {3}mins; {4}seconds".format(
            best_em, best_f1, h, m, s))
        log.info("Final model EM-Score: {0}, F1-Score: {1}".format(best_em, best_f1))
        log.info("Total time to train & val: {0}hours; {1}mins; {2}seconds".format(h, m, s))

        return  train_loss , val_loss , em_lis , f1_lis



class Evaluation():
    def __int__(self):
        pass

    '''
      *
      *  Summary : This is the main function for evaluation. 
      *
      *  Args    : Param - eval_dataset - SQuAD dev_set
      *
      *  Returns : EM-Score and F1-Score
      *
     '''
    def test(self, eval_dataset):

        model_1 = DrQA(num_layers=3, input_size=embed_dim, hidden_size=embed_dim, dropout=0.5)
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        model_1.to(device)
        model_DrQA.load_state_dict(torch.load('../Reader_model_output/DrQA_trained_SQuAD-1.pth'))
        model_1.eval()

        em, f1 = eval.evaluate(model_1, eval_dataset, BERT=False)
        print("Best model Em: {0} & F1-Score: {1}".format(em, f1))
        log.info("Best model Em: {0} & F1-Score: {1}".format(em, f1))


'''
 *
 *  Summary : This is the main class for training, validating and evaluating.
 *
'''

if __name__ == '__main__':

    '''
      *
      *  Summary : This function sets the logger file and format. 
      *
      *
     '''
    # Create a logger.
    def setup_logger(name, log_file, level=logging.INFO):
        handler = logging.FileHandler(log_file)
        handler.setFormatter(formatter)

        logger = logging.getLogger(name)
        logger.setLevel(level)
        logger.addHandler(handler)

        return logger

    # set up the formate of the logger
    formatter = logging.Formatter('%(asctime)s %(message)s')

    # setup train_Eval logger
    log = setup_logger('logger', '../DrQA-TrainVal.log')

    '''
      *
      *  Summary : This block of code performs the preprocessing by calling data_preprocessing class. 
      *
      *
     '''
    train_dataset, val_dataset, eval_dataset = prepro.main(BERT=False)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=True, num_workers=3, pin_memory=True)
    validation_loader = DataLoader(val_dataset, batch_size=64, shuffle=True, drop_last=True, num_workers=3, pin_memory=True)

    torch.cuda.empty_cache()

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    model_DrQA = DrQA(num_layers=3, input_size=100, hidden_size=100, dropout=0.5)
    model_DrQA.to(device)

    optimizer = optim.Adam(model_DrQA.parameters(), lr=1e-3)

    epochs = 50

    train = Train_Validate()
    train_loss , val_loss , em_list , f1_list = train.train(epochs, optimizer, train_loader,
                                                            validation_loader, model_DrQA, device)

    test_set_eval = Evaluation()
    test_set_eval.test(eval_dataset)

    '''
      *
      *  Summary : This block of code plots the train and validation loss against epochs . 
      *
      *
     '''
    assert (len(train_loss) == len(val_loss))
    plt.grid()
    plt.plot(list(range(len(train_loss))), train_loss, "o-", label="Train")
    plt.plot(list(range(len(val_loss))), val_loss, "o-", label="Validation")
    plt.title('Train & val loss vs Epochs for DrQA- SQUAD-2')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    '''
      *
      *  Summary : This block of code plots the Em and F1 score against epochs . 
      *
      *
     '''
    assert (len(em_list) == len(f1_list))
    plt1.grid()
    plt1.plot(list(range(len(em_list))), em_list, "o-", label="Train")
    plt1.plot(list(range(len(f1_list))), f1_list, "o-", label="Validation")
    plt1.title('EM & F1 Scores vs Epochs for DrQA-SQUAD-2')
    plt1.xlabel('Epochs')
    plt1.ylabel('Metrics')
    plt1.legend()
    plt1.show()



