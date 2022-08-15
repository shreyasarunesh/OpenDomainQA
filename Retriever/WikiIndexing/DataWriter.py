
'''
 *
 * Name                   :   F21MP- Open Domain Question Answering (ODQA) Agent.
 *
 * Description            :  This class writes indexed data into different types of field files.
                            This may be initial or intermediate indexing files.
                            Ex. - input text ----> intermediate files or final index files
 *
 * Author                 :   Shreyas Arunesh
 *
 *
 * Reference              : https://github.com/neufang/wiki-Dump-Indexing/blob/master/buildInvertedIndexFromWikiDump.py
'''

import re
from tqdm import tqdm
from collections import defaultdict

index_map = defaultdict(str)
num_files = 0
num_pages = 0
id_title_map = {}

class WriteData():

    def __init__(self):

        pass

    '''
      *
      *  Summary : This function write maps all the titles index in a single file named title_id_map. 
                     This is the inital contact point for file traverser. 
                     Tags are diffrent fields (title, body, reference etc...)


      *
      *  Args    : Param - idex_id, index_map
      *
      *  Returns : Dosent returns anything but creates and writes to files of diffrent fields. 
      *
     '''

    def write_id_title_map(self):

        global id_title_map
        temp_id_title = []

        temp_id_title_map = sorted(id_title_map.items(), key=lambda item: int(item[0]))

        for id, title in tqdm(temp_id_title_map):
            t = str(id) + '-' + title.strip()
            temp_id_title.append(t)

        with open('../Dataset/output_data/english_wiki_index/id_title_map.txt', 'a', encoding= 'UTF-8') as f:
            f.write('\n'.join(temp_id_title))
            f.write('\n')

    '''
       *
       *  Summary : This function writes to initial index files and assigns a tag to the respective file.
       *
       *  Args    : Param - index_file, index_map.
       *
       *  Returns : Dosent returns anything but creates and writes to intermediate index files. 
       *
      '''
    def write_intermed_index(self):

        global num_files
        global index_map
        temp_index_map = sorted(index_map.items(), key=lambda item: item[0])

        temp_index = []
        for word, posting in tqdm(temp_index_map):
            temp_index.append(word + '-' + posting)

        with open(f'../Dataset/output_data/english_wiki_index/index_{num_files}.txt','w', encoding= 'UTF-8') as f:
            f.write('\n'.join(temp_index))

        num_files += 1

    '''
     *
     *  Summary : This function writes to iniermediate index files and assigns a tag to the respective file.  
     *
     *  Args    : Param - title_dict, body_dict, category_dict, infobox_dict, link_dict, reference_dict.
     *
     *  Returns : Dosent returns anything but creates and writes to intermediate index files. 
     *
    '''
    def write_final_files(self, data_to_merge, num_files_final):

        title_dict, body_dict, category_dict, infobox_dict, link_dict, reference_dict = defaultdict(dict), defaultdict(
            dict), defaultdict(dict), defaultdict(dict), defaultdict(dict), defaultdict(dict)

        unique_tokens_info = {}

        sorted_data = sorted(data_to_merge.items(), key=lambda item: item[0])

        for i, (token, postings) in tqdm(enumerate(sorted_data)):
            for posting in postings.split(';')[:-1]:

                id = posting.split(':')[0]

                fields = posting.split(':')[1]

                if 't' in fields:
                    title_dict[token][id] = re.search(r'.*t([0-9]*).*', fields).group(1)

                if 'b' in fields:
                    body_dict[token][id] = re.search(r'.*b([0-9]*).*', fields).group(1)

                if 'c' in fields:
                    category_dict[token][id] = re.search(r'.*c([0-9]*).*', fields).group(1)

                if 'i' in fields:
                    infobox_dict[token][id] = re.search(r'.*i([0-9]*).*', fields).group(1)

                if 'l' in fields:
                    link_dict[token][id] = re.search(r'.*l([0-9]*).*', fields).group(1)

                if 'r' in fields:
                    reference_dict[token][id] = re.search(r'.*r([0-9]*).*', fields).group(1)

            token_info = '-'.join([token, str(num_files_final), str(len(postings.split(';')[:-1]))])
            unique_tokens_info[token] = token_info + '-'

        final_titles, final_body_texts, final_categories, final_infoboxes, final_links, final_references = [], [], [], [], [], []

        for i, (token, _) in tqdm(enumerate(sorted_data)):

            if token in title_dict.keys():
                posting = title_dict[token]
                final_titles = self.get_diff_postings(token, posting, final_titles)
                t = len(final_titles)
                unique_tokens_info[token] += str(t) + '-'
            else:
                unique_tokens_info[token] += '-'

            if token in body_dict.keys():
                posting = body_dict[token]
                final_body_texts = self.get_diff_postings(token, posting, final_body_texts)
                t = len(final_body_texts)
                unique_tokens_info[token] += str(t) + '-'
            else:
                unique_tokens_info[token] += '-'

            if token in category_dict.keys():
                posting = category_dict[token]
                final_categories = self.get_diff_postings(token, posting, final_categories)
                t = len(final_categories)
                unique_tokens_info[token] += str(t) + '-'
            else:
                unique_tokens_info[token] += '-'

            if token in infobox_dict.keys():
                posting = infobox_dict[token]
                final_infoboxes = self.get_diff_postings(token, posting, final_infoboxes)
                t = len(final_infoboxes)
                unique_tokens_info[token] += str(t) + '-'
            else:
                unique_tokens_info[token] += '-'

            if token in link_dict.keys():
                posting = link_dict[token]
                final_links = self.get_diff_postings(token, posting, final_links)
                t = len(final_links)
                unique_tokens_info[token] += str(t) + '-'
            else:
                unique_tokens_info[token] += '-'

            if token in reference_dict.keys():
                posting = reference_dict[token]
                final_references = self.get_diff_postings(token, posting, final_references)
                t = len(final_references)
                unique_tokens_info[token] += str(t) + '-'
            else:
                unique_tokens_info[token] += '-'

        with open('../Dataset/output_data/english_wiki_index/tokens_info.txt', 'a', encoding= 'UTF-8') as f:
            f.write('\n'.join(unique_tokens_info.values()))
            f.write('\n')

        self.write_diff_postings('title', final_titles, num_files_final)

        self.write_diff_postings('body', final_body_texts, num_files_final)

        self.write_diff_postings('category', final_categories, num_files_final)

        self.write_diff_postings('infobox', final_infoboxes, num_files_final)

        self.write_diff_postings('link', final_links, num_files_final)

        self.write_diff_postings('reference', final_references, num_files_final)

        num_files_final += 1

        return num_files_final


    '''
         *
         *  Summary : This function writes to diffrent postings possition.  
         *
         *  Args    : Param - tag_type, final_tag, num_files_final.
         *
         *  Returns : th.
         *
        '''

    def write_diff_postings(self, tag_type, final_tag, num_files_final):

        with open(f'../Dataset/output_data/english_wiki_index/{tag_type}_data_{str(num_files_final)}.txt', 'w', encoding='UTF-8') as f:
            f.write('\n'.join(final_tag))

    '''
     *
     *  Summary : This function gets diffrent postings possition.  
     *
     *  Args    : Param - token, postings, final_tag.
     *
     *  Returns : returns the postion tag of the given title_id  in initial or intermediate index.
     *
    '''

    def get_diff_postings(self, token, postings, final_tag):

        postings = sorted(postings.items(), key=lambda item: int(item[0]))

        final_posting = token + '-'
        for id, freq in postings:
            final_posting += str(id) + ':' + freq + ';'

        final_tag.append(final_posting.rstrip(';'))

        return final_tag