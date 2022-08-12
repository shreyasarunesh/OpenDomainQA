
'''
 *
 * Name                   :   F21MP- Open Domain Question Answering (ODQA) Agent.
 *
 * Description            :  This class traverses various index files and return the required data
 *
 * Author                 :   Shreyas Arunesh
 *
 *
 * Reference              : https://github.com/dorianbrown/rank_bm25
'''

import linecache

class FileTraverser():

        def __init__(self):

                pass

        '''
         *
         *  Summary : This function performs binary search to get the token information. 
         *
         *  Args    : Param - filename, inp_token
         *
         *  Returns : returns token information.
         *
        '''
        def binary_search_token_info(self, high, filename, inp_token):

                low = 0
                while low < high:

                        mid = (low + high) // 2
                        line = linecache.getline(filename, mid)
                        token = line.split('-')[0]

                        if inp_token == token:
                                token_info = line.split('-')[1:-1]
                                return token_info

                        elif inp_token > token:
                                low = mid + 1

                        else:
                                high = mid

                return None

        '''
         *
         *  Summary : This function performes the title search given the title id. 
         *
         *  Args    : Param - page_id
         *
         *  Returns : returns titles information.
         *
        '''
        def title_search(self, page_id):

                title = linecache.getline('../Dataset/output_data/english_wiki_index/id_title_map.txt', int(page_id) + 1).strip()
                title = title.split('-', 1)[1]

                return title

        '''
         *
         *  Summary : This function gets the field of the given token where wrt to . 
         *
         *  Args    : Param - field, file_num, line_num
         *
         *  Returns : returns titles information.
         *
        '''

        def search_field_file(self, field, file_num, line_num):

                if line_num != '':
                        line = linecache.getline(f'../Dataset/output_data/english_wiki_index/{field}_data_{str(file_num)}.txt', int(line_num)).strip()
                        postings = line.split('-')[1]

                        return postings

                return ''

        '''
         *
         *  Summary : This function performs the title search given the title id. 
         *
         *  Args    : Param - page_id
         *
         *  Returns : returns titles information.
         *
        '''

        def get_token_info(self, token):

                char_list = [chr(i) for i in range(97, 123)]
                num_list = [str(i) for i in range(0, 10)]

                if token[0] in char_list:
                        with open(f'../Dataset/output_data/english_wiki_index/tokens_info_{token[0]}_count.txt', 'r') as f:
                                num_tokens = int(f.readline().strip())

                        tokens_info_pointer = f'../Dataset/output_data/english_wiki_index/tokens_info_{token[0]}.txt'
                        token_info = self.binary_search_token_info(num_tokens, tokens_info_pointer, token)

                elif token[0] in num_list:
                        with open(
                                f'../Dataset/output_data/english_wiki_index/tokens_info_{token[0]}_count.txt', 'r') as f:
                                num_tokens = int(f.readline().strip())

                        tokens_info_pointer = f'../Dataset/output_data/english_wiki_index/tokens_info_{token[0]}.txt'
                        token_info = self.binary_search_token_info(num_tokens, tokens_info_pointer, token)

                else:
                        with open( f'../Dataset/output_data/english_wiki_index/tokens_info_others_count.txt', 'r') as f:
                                num_tokens = int(f.readline().strip())

                        tokens_info_pointer = f'../Dataset/output_data/english_wiki_index/tokens_info_others.txt'
                        token_info = self.binary_search_token_info(num_tokens, tokens_info_pointer, token)

                return token_info