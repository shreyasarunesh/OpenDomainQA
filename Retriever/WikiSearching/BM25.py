
'''
 *
 * Name                   :   F21MP- Open Domain Question Answering (ODQA) Agent.
 *
 * Description            :  This class illustrates the Implementation of BM25 ranking function.
 *
 * Author                 :   Shreyas Arunesh
 *
 *
 * Reference              : https://github.com/dorianbrown/rank_bm25
'''
from collections import defaultdict
import math
index_map = defaultdict(str)
num_files = 0
num_pages = 0
id_title_map = {}


class BM25():
        '''
         *
         *  Summary : This class ranks the articles by considering the freequency of the words appearing in the posting.
                        based on the fields.
         *
         *  Args    : Param - page freequency and page posting.
         *
         * Hyper-parameters:   weight dictionary: 'titles' = 0.9, 'body': 1.0, 'category': 0.75, 'infobox': 0.20,
                                'link': 0.20, 'reference': 0.25
         *
         *  Returns :  returns ranked articles
         *
        '''

        def __init__(self, num_pages):

                self.num_pages = num_pages

        def do_ranking(self, page_freq, page_postings):

                result = defaultdict(float)
                weightage_dict = {'title': 0.9, 'body': 1.0, 'category': 0.75, 'infobox': 0.20, 'link': 0.20,
                                  'reference': 0.25}

                for token, field_post_dict in page_postings.items():

                        for field, postings in field_post_dict.items():

                                weightage = weightage_dict[field]

                                if len(postings) > 0:
                                        for post in postings.split(';'):
                                                id, post = post.split(':')
                                                result[id] += weightage * (1 + math.log(int(post))) * math.log(
                                                        (self.num_pages - int(page_freq[token])) / int(
                                                                page_freq[token]))

                return result