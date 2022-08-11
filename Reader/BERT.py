

'''
 *
 * Name                   :   F21MP- Open Domain Question Answering (ODQA) Agent.
 *
 * Description            :  This file illustrates the implementation of BERT-QA model
                            for training, validation and evaluation and evaluation.

                            The model used in BERT-base uncased from Huggingface. A simple linear classification layer is
                            added for classification of start and end tokens.
 *
 * Author                 :   Shreyas Arunesh
 *
 *
 * Reference              : https://github.com/castorini/bertserini

                           : https://huggingface.co/bert-base-uncased
 *
 *
 *

'''


import time
import os
import logging
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt1

from Reader.Preprocessing import *
from Reader.Eval_metrics import *

prepro = Preprocessing()
eval = Evaluation()

from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AdamW, BertForQuestionAnswering

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

import torch.nn as nn
from transformers import BertModel, BertPreTrainedModel


class Bert_QA(BertPreTrainedModel):
    '''
     *
     *  Summary : This class loads and performs forward propagation.
     *
     *  Args    : Param - Pretrained BERT-base template from huggingface.
     *
     *
    '''

    def __init__(self, config):
        '''
         *
         *  Summary : This is the constructor class to initialise the model and additional of classification
                        layer at the end of last layer.

         *
         *  Args    : Param- BERT-config.
         *
         *  Returns : BERT-base initialised for finetuning.
         *
        '''
        super().__init__(config)

        self.bert = BertModel(config, add_pooling_layer=False)
        self.fc = nn.Linear(config.hidden_size, 2)
        self.criterion = nn.CrossEntropyLoss()

    '''
        *
        *  Summary : This function performs forward propagation to predicts the start and end position of the answer
                        and calculates the loss.

        *
        *  Args    : Param-     input_ids,
                                attention_mask,
                                token_type_ids,
                                start_positions of answer
                                end_positions of answer
        *
        *  Returns : BERT-base initialised for finetuning.
        *
       '''
    def forward(
            self,
            input_ids,
            attention_mask,
            token_type_ids,
            start_positions,
            end_positions,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        sequence_output = outputs.last_hidden_state  # [batch,length,dim]

        logits = self.fc(sequence_output)  # [batch,length,2]

        context_mask = (attention_mask - token_type_ids).unsqueeze(-1)  # [batch,length,1]
        logits = logits + (context_mask + 1e-45).log()

        start_logits, end_logits = logits.split(1, dim=-1)  # 2*[batch,length,1]
        start_logits = start_logits.squeeze(-1)  # [batch,length]
        end_logits = end_logits.squeeze(-1)  # [batch,length]

        start_loss = self.criterion(start_logits, start_positions)
        end_loss = self.criterion(end_logits, end_positions)
        loss = start_loss + end_loss

        return loss

    '''
        *
        *  Summary : This function get the start and end score of tokens classified.
                       where end score position greater than or equal to start score.

        *
        *  Args    : Param-       input_ids,
                                   attention_mask,
                                   token_type_ids
        *
        *  Returns : start and end score of tokens.
        *
       '''

    def get_scores(self,
                   input_ids,
                   attention_mask,
                   token_type_ids
                   ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True
        )

        sequence_output = outputs.last_hidden_state

        logits = self.fc(sequence_output)  # [batch,length,2]

        context_mask = (attention_mask - token_type_ids).unsqueeze(-1)  # [batch,length,1]
        logits = logits + (context_mask + 1e-45).log()

        start_logits, end_logits = logits.split(1, dim=-1)  # 2*[batch,length,1]
        start_logits = start_logits.squeeze(-1)  # [batch,length]
        end_logits = end_logits.squeeze(-1)  # [batch,length]

        start_score = nn.Softmax(dim=1)(start_logits)
        end_score = nn.Softmax(dim=1)(end_logits)

        return start_score, end_score

'''
    *
    *  Summary : This is the class for training and validation of BERT-base model through back propagation.
    *
    *
   '''

class Train_Validate:
    def __int__(self):
        pass

    '''
        *
        *  Summary : This is function to train and validate BERT-base model.
                    The weights are updated in back propagation by calculating train and val loss

        *
        *  Args    : Param-       epochs, train_loader, val_loader, 
                                    optim, model, device                      
        *
        *  Returns : train_loss, val_loss, em_list, f1_list
        *
       '''
    def train_validate(self, epochs, optim, train_loader, val_loader, model, device):

        s = time.time()
        iter_counter = 0
        best_f1 = 0
        best_em = 0
        avg_loss = 0
        em_list = []
        f1_list = []

        train_loss = []
        val_loss = []

        model.train()

        for epoch_ind, epoch in enumerate(range(epochs)):
            print('#### Training for Epoch {0} ####'.format(epoch_ind + 1))
            logger.info('#### Training for Epoch {0} ####'.format(epoch_ind + 1))

            loss_of_epoch = 0

            for batch_ind, batch in enumerate(train_loader):
                optim.zero_grad()
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                start_positions = batch['start_positions'].to(device)
                end_positions = batch['end_positions'].to(device)
                token_type_ids = batch['token_type_ids'].to(device)
                loss = model(input_ids, attention_mask=attention_mask, start_positions=start_positions,
                             end_positions=end_positions, token_type_ids=token_type_ids)
                loss.backward()
                optim.step()

                iter_counter += 1

                avg_loss += loss.item()
                loss_of_epoch += loss.item()

                if iter_counter % 250 == 0:
                    avg_loss /= 250
                    print('iter {0}'.format(iter_counter))
                    print('loss {0}'.format(avg_loss))
                    avg_loss = 0
                if iter_counter % 2000 == 0:
                    em, f1 = eval.evaluate(model, val_dataset)
                    model.train()
                    if f1 > best_f1:
                        best_f1 = f1
                        best_em = em
                        model.save_pretrained('/content/drive/MyDrive/ODQA/outputs/bert_model_final')
                        print('best model!')
                        print(
                            "Best model of batch: {0} / {1} EM-Score:{2} & F1-score:{3}".format(batch_ind + 1,
                                                                                                len(train_loader),
                                                                                                best_em,
                                                                                                best_f1))
                        logger.info("Best models of batch: {0} / {1} EM-Score:{2} & F1-score:{3}".format(batch_ind + 1,
                                                                                                         len(train_loader),
                                                                                                         best_em,
                                                                                                         best_f1))
                        print('\n')

            loss_of_epoch /= len(train_loader)
            train_loss.append(avg_loss)

            em_list.append(best_em)
            f1_list.append(best_f1)

            """Validation on dev set """

            print("############# Validation on Epoch {0} ########".format(epoch_ind + 1))
            logger.info("############# Validation on Epoch {0} ########".format(epoch_ind + 1))

            model.eval()

            loss_of_epoch = 0

            for batch_ind, batch in enumerate(val_loader):
                optim.zero_grad()

                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                start_positions = batch['start_positions'].to(device)
                end_positions = batch['end_positions'].to(device)
                token_type_ids = batch['token_type_ids'].to(device)

                loss = model(input_ids, attention_mask=attention_mask, start_positions=start_positions,
                             end_positions=end_positions, token_type_ids=token_type_ids)

                avg_loss += loss.item()
                loss_of_epoch += loss.item()

            loss_of_epoch /= len(val_loader)
            val_loss.append(avg_loss)

            print('--------epoch %d finish!-----------' % (epoch + 1))
            logger.info('epoch %d finish!' % (epoch + 1))

            print("Train Loss of this epoch:{2} \n Validation Loss:{3} \nEm-Score: {0} \n F1-Score: {1}\n".format(
                em_list[-1], f1_list[-1],
                train_loss[-1], val_loss[-1]))
            print("--------------------------------")
            print('\n')

            logger.info("Train Loss of this epoch:{2} \n Validation Loss:{3} \nEm-Score: {0} \n F1-Score: {1}".format(
                em_list[-1], f1_list[-1],
                train_loss[-1], val_loss[-1]))

        e = time.time()
        temp = e - s
        hours = temp // 3600
        temp = temp - 3600 * hours
        minutes = temp // 60
        seconds = temp - 60 * minutes

        print("Time taken for training: {0} hours; {1} min; {2} seconds".format(hours, minutes, seconds))
        logger.info("Time taken for training: {0} hours; {1} min; {2} seconds".format(hours, minutes, seconds))

        return train_loss, val_loss, em_list, f1_list

    '''
        *
        *  Summary : This function is to plot metrics_score vs Epochs and train_loss, vali_loss vs epochs

        *
        *  Args    : Param-       train loss, vali loss, em_list, f1 list and epochs                     
        *
        *  Returns : plot Metrics_score vs Epochs and train_loss, vali_loss vs epochs
        *
    '''
    def plot(self, train_loss, val_loss, em_list, f1_list):
        assert (len(train_loss) == len(val_loss))
        assert (len(em_list) == len(f1_list))

        plt.grid()
        plt.plot(list(range(len(train_loss))), train_loss, "o-", label="Train_loss")
        plt.plot(list(range(len(train_loss))), val_loss, "o-", label="Val_loss")
        plt.title('Train & val loss vs Epochs for BERT-base')
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()

        plt1.grid()
        plt1.plot(list(range(len(em_list))), em_list, "o-", label="EM-Score")
        plt1.plot(list(range(len(f1_list))), f1_list, "o-", label="F1-Score")
        plt1.title('EM & F1 Scores vs Epochs for BERT-base')
        plt1.xlabel('Epochs')
        plt1.ylabel('Metric Score')
        plt1.legend()
        plt1.show()

'''
    *
    *  Summary : This function evaluates the best saved model after training on eval dataset

    *
    *  Args    : Param-   evaluation dataset: SQUAD development set                    
    *
    *  Returns : EM-Score and F1-Score
    *
'''
class Evaluate:
    def __int__(self):
        pass

    def test(self):
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        model_bert = Bert_QA.from_pretrained(
            '/Users/shreyasarunesh/Desktop/Open_Domain_Question_Answering_Agent/Reader/BERTbase/Reader_model_output/bert_model_final')
        model_bert.to(device)
        model_bert.eval()
        em, f1 = eval.evaluate(model_bert, val_dataset)
        print("##########Evaluating on best model #########")
        print('EM: %.5f' % em)
        print('F1: %.5f' % f1)
        logger.info("##########Evaluating on best model #########")
        logger.info('EM: %.5f' % em)
        logger.info('F1: %.5f' % f1)

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
    def setup_logger(name, log_file, level=logging.INFO):
        handler = logging.FileHandler(log_file)
        handler.setFormatter(formatter)

        logger = logging.getLogger(name)
        logger.setLevel(level)
        logger.addHandler(handler)

        return logger

    # setup the formate of the logger
    formatter = logging.Formatter('%(asctime)s %(message)s')
    # setup train_Eval logger
    logger = setup_logger('logger',
                          '/Users/shreyasarunesh/Desktop/Open_Domain_Question_Answering_Agent/Reader/model_output/BERT-TrainVal.log')

    train_vali = Train_Validate()

    '''
      *
      *  Summary : This block of code performs the preprocessing by calling data_preprocessing class. 
      *
      *
     '''
    train_dataset, val_dataset = prepro.main(BERT=True)

    torch.cuda.empty_cache()

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    model = Bert_QA.from_pretrained('bert-base-uncased')
    model.to(device)

    optim = AdamW(model.parameters(), lr=3e-5)
    epochs = 5
    train_batch_size = 16
    val_batch_size = 16

    train_loader = DataLoader(train_dataset, train_batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, val_batch_size, shuffle=True, drop_last=True)

    #
    '''
      *
      *  Summary : This block of code performs the training of BERT model 
                    on SQUAD-train and validating on SQUAD-dev
      *
      *
     '''

    train_loss, val_loss, em_list, f1_list = train_vali.train_validate(epochs, optim, train_loader, val_loader, model,
                                                                       device)

    '''
      *
      *  Summary : This block of code calls the plot function.
      *
      *
     '''
    train_vali.plot(train_loss, val_loss, em_list, f1_list)


    '''
       *
       *  Summary : Evaluating saved best BERT-model on SQUAD-dev dataset.
       *
       *
    '''

    best_eval = Evaluate()
    best_eval.test()
