
'''
 *
 * Name                   :   F21MP- Open Domain Question Answering (ODQA) Agent.
 *
 * Description            :  This class illustrates the process of extracting the freequency of the tokens along with
                                the posting in the form of dictionary.
                                1. Simple query
                                2. Field specific query
 *
 * Author                 :   Shreyas Arunesh
 *
 *
 * Reference              : https://github.com/dorianbrown/rank_bm25
'''

from collections import defaultdict

index_map = defaultdict(str)
num_files = 0
num_pages = 0
id_title_map = {}

'''
This class takes query as input and returns the corresponding postings along with theis fields
'''

class QueryResults():

    def __init__(self, file_traverser):

        self.file_traverser = file_traverser
    '''
     *
     *  Summary : This function takes preprocessed Simple query as an input (Single Query) and outputs the 
                    dictionary with page freequency and postings of that page. 
     *
     *  Args    : Param - preprocessed query
     *
     *  Returns : returns page_freequency and page_posting. 
     *
    '''

    def simple_query(self, preprocessed_query):

        page_freq, page_postings = {}, defaultdict(dict)

        for token in preprocessed_query:
            token_info = self.file_traverser.get_token_info(token)

            if token_info:
                file_num, freq, title_line, body_line, category_line, infobox_line, link_line, reference_line = token_info
                line_map = {
                    'title': title_line, 'body': body_line, 'category': category_line,
                    'infobox': infobox_line, 'link': link_line, 'reference': reference_line
                }

                for field_name, line_num in line_map.items():

                    if line_num != '':
                        posting = self.file_traverser.search_field_file(field_name, file_num,
                                                                        line_num)

                        page_freq[token] = len(posting.split(';'))
                        page_postings[token][field_name] = posting

        return page_freq, page_postings

    '''
     *
     *  Summary : This function takes preprocessed field query as an input and outputs the 
                    dictionary with page freequency and postings of that page. (Multiple query- Max: 2 Query)
                    
                    The function looks for a freequency of the token in that posting specified by user.
     *
     *  Args    : Param - preprocessed query
     *
     *  Returns : returns page_freequency and page_posting. 
     *
    '''
    def field_query(self, preprocessed_query):

        page_freq, page_postings = {}, defaultdict(dict)

        for field, token in preprocessed_query:
            token_info = self.file_traverser.get_token_info(token)

            if token_info:
                file_num, freq, title_line, body_line, category_line, infobox_line, link_line, reference_line = token_info
                line_map = {
                    'title': title_line, 'body': body_line, 'category': category_line,
                    'infobox': infobox_line, 'link': link_line, 'reference': reference_line
                }
                field_map = {
                    't': 'title', 'b': 'body', 'c': 'category', 'i': 'infobox', 'l': 'link',
                    'r': 'reference'
                }

                field_name = field_map[field]
                line_num = line_map[field_name]

                posting = self.file_traverser.search_field_file(field_name, file_num, line_num)
                page_freq[token] = len(posting)
                page_postings[token][field_name] = posting

        return page_freq, page_postings