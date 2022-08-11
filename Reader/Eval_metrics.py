

'''
 *
 * Name                   :   F21MP- Open Domain Question Answering (ODQA) Agent.
 *
 * Description            :  This file illustrates the evaluation metrics used for both DrQA and BERT models.

 *
 *
  Metrics:                  EM- Exact Match and F1 Score
 *
 * Author                 :   Shreyas Arunesh
 *
 *
 * Reference              : https://github.com/facebookresearch/DrQA
                          : https://github.com/castorini/bertserini
'''


from transformers import BertTokenizerFast
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
import collections
from collections import Counter
from tqdm.auto import tqdm
from torch.utils.data import DataLoader, BatchSampler, SequentialSampler

import torch

class Evaluation_metrics():

    def __int__(self):
        pass
    '''
     *
     *  Summary : This function performs all typical text processing for a given sentence 
                    before calculating the evaluation metrics. 
                    Steps:        Removing articles and punctuation, and standardizing whitespace.
     *
     *
     *  Args    : Param - input sentence
     *
     *  Returns : returns normalised text.
     *
    '''
    def normalize_text(self, s):
        """Removing articles and punctuation, and standardizing whitespace are all typical text processing steps."""
        import string, re

        def remove_articles(text):
            regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
            return re.sub(regex, " ", text)

        def white_space_fix(text):
            return " ".join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return "".join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(s))))

    def get_tokens(self, s):
        if not s:
            return []
        return self.normalize_text(s).split()

    '''
     *
     *  Summary : This function calculates a Exact Match score for gold_sentence and predicted sentence
     *
     *  Args    : Param - gold answer and predicted answer.
     *
     *  Returns : returns Exact match score.
     *
    '''
    # metric implementation here
    def compute_exact(self, a_gold, a_pred):

        if self.normalize_text(a_gold) == self.normalize_text(a_pred):
            return 1
        else:
            return 0

    '''
      *
      *  Summary : This function calculates a F1-Score for gold_sentence and predicted sentence
      *
      *  Args    : Param - gold answer and predicted answer.
      *
      *  Returns : returns F1-Score.
      *
     '''
    def compute_f1(self, a_gold, a_pred):

        gold_tokens = self.get_tokens(a_gold)
        pred_tokens = self.get_tokens(a_pred)

        overlap = collections.Counter(gold_tokens) & collections.Counter(pred_tokens)
        num_overlap = sum(overlap.values())

        # if one of ground-truth and prediction is un-answered,
        # then F1 would be 1 if they are the same, otherwise be 0.
        if len(gold_tokens) == 0 or len(pred_tokens) == 0:
            if gold_tokens == pred_tokens:
                return 1
            else:
                return 0

        # if both are answered, just calculate F1 using the formula given above

        if num_overlap == 0:
            return 0

        precision = 1.0 * num_overlap / len(pred_tokens)
        recall = 1.0 * num_overlap / len(gold_tokens)
        f1 = (2.0 * precision * recall) / (precision + recall)
        return f1


class Evaluation():
    '''
      *
      *  Summary : This is the main class to calculate the evaluation metrics for a given model.
                    Two settings: BERT-base and DrQA.
      *
      *
      *  Args    : Param - model and preprocessed dataset
      *
      *  Returns :EM and f1 score for the given model and dataset.
      *
     '''
    def __int__(self):
        pass

    def evaluate(self, model, dataset, BERT= True):
        metrics = Evaluation_metrics()
        with torch.no_grad():
            model.eval()
            device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
            model.to(device)

            em = 0.0
            f1 = 0.0

            if BERT:

                for i in tqdm(range(len(dataset))):
                    temp_data = dataset[i]
                    input_ids = temp_data['input_ids'].to(device).unsqueeze(0)
                    attention_mask = temp_data['attention_mask'].to(device).unsqueeze(0)
                    token_type_ids = temp_data['token_type_ids'].to(device).unsqueeze(0)
                    start_score, end_score = model.get_scores(input_ids, attention_mask=attention_mask,
                                                              token_type_ids=token_type_ids)

                    start_score = start_score.squeeze(0).cpu()
                    end_score = end_score.squeeze(0).cpu()

                    answer_start = torch.argmax(start_score).item()
                    answer_end = torch.argmax(end_score).item()

                    pred = ''
                    length = start_score.size(0)

                    if answer_start == 0 or answer_end == 0 or answer_start == (length - 1) or answer_end == (length - 1):
                        pred = ''

                    elif answer_end < answer_start:
                        pred = ''

                    elif answer_end - answer_start > 20:
                        pred = ''

                    else:
                        input_ids.cpu()
                        pred = tokenizer.decode(input_ids[0][answer_start:(answer_end + 1)])

                    truth = dataset[i]['gold_text']

            else:

                loader = DataLoader(dataset=dataset, sampler=SequentialSampler(dataset), batch_size=512,
                                    drop_last=False)

                start_list = []
                end_list = []
                for context_tensor, context_mask, question_tensor, question_mask in tqdm(loader):
                    start_score, end_score = model.get_scores(context_tensor.to(device),
                                                              context_mask.to(device),
                                                              question_tensor.to(device),
                                                              question_mask.to(device))
                    answer_start = torch.argmax(start_score, dim=1).cpu().tolist()
                    answer_end = torch.argmax(end_score, dim=1).cpu().tolist()

                    start_list.extend(answer_start)
                    end_list.extend(answer_end)

                for i in range(len(dataset)):

                    prediction = ''

                    answer_start = start_list[i]
                    answer_end = end_list[i]
                    temp_cont_list = dataset.context_list[i]
                    seq_length = len(temp_cont_list)

                    if answer_start == 0 or answer_end == 0 or answer_start >= (
                            seq_length - 1) or answer_end >= (seq_length - 1):
                        pred = ''

                    elif answer_end < answer_start:
                        pred = ''

                    elif answer_end - answer_start > 20:
                        pred = ''

                    else:
                        prediction = temp_cont_list[answer_start]
                        for j in range(answer_start + 1, min(answer_end + 1, len(temp_cont_list))):
                            if len(temp_cont_list[j]) > 2 and temp_cont_list[j][0:2] == '##':
                                prediction += temp_cont_list[j][2:]
                            else:
                                prediction += (' ' + temp_cont_list[j])

                    truth = dataset.answers[i]['text']

        em += metrics.compute_exact(truth, prediction)
        f1 += metrics.compute_f1(truth, prediction)
        em /= len(dataset)
        f1 /= len(dataset)

        return em, f1
