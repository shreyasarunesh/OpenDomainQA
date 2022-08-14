
'''
 *
 * Name                   :   F21MP- Open Domain Question Answering (ODQA) Agent.
 *
 * Description            :  This class illustrates the process of creating an index.
                            This includes initial and intermediate intermediate indexing
                            This class uses SPIMI- Single-pass in-memory indexing algorithm
 *
 *
 Indexing formate:      jonny-2314:t3b6i2r1;6432:t5c8b3i1;
                        Ex.:- t3b6i2r1 represents token appears 3 times in title, 6 times in the body,
                        2 times in infobox and 1time in reference.
 *
 * Author                 :   Shreyas Arunesh
 *
 *
 * Reference              : https://github.com/neufang/wiki-Dump-Indexing/blob/master/buildInvertedIndexFromWikiDump.py
'''
import re
from collections import defaultdict

index_map = defaultdict(str)
num_files = 0
num_pages = 0
id_title_map = {}


class CreateIndex():

    """
       *
       * Summary :    This class create the respective index field files of wikipedia dump.
                       Ex - input text ----> index files
       *
       *
       Args:            title, body, category, infobox, liks, reference.
       *
    """

    def __init__(self, write_data):

        self.write_data = write_data

    '''
      *
      *  Summary : This creates the index wrt to the fields. 
      *
      *  Args    : Param -  title, body, category, infobox, link, reference
      *
      *  Returns : Creates the index.
      *
     '''
    def index(self, title, body, category, infobox, link, reference):

        global num_pages
        global index_map
        global id_title_map

        words_set, title_dict, body_dict, category_dict, infobox_dict, link_dict, reference_dict = set(), defaultdict(
            int), defaultdict(int), defaultdict(int), defaultdict(int), defaultdict(int), defaultdict(int)

        words_set.update(title)
        for word in title:
            title_dict[word] += 1

        words_set.update(body)
        for word in body:
            body_dict[word] += 1

        words_set.update(category)
        for word in category:
            category_dict[word] += 1

        words_set.update(infobox)
        for word in infobox:
            infobox_dict[word] += 1

        words_set.update(link)
        for word in link:
            link_dict[word] += 1

        words_set.update(reference)
        for word in reference:
            reference_dict[word] += 1

        for word in words_set:
            temp = re.sub(r'^((.)(?!\2\2\2))+$', r'\1', word)
            is_rep = len(temp) == len(word)

            if not is_rep:
                posting = str(num_pages) + ':'

                if title_dict[word]:
                    posting += 't' + str(title_dict[word])

                if body_dict[word]:
                    posting += 'b' + str(body_dict[word])

                if category_dict[word]:
                    posting += 'c' + str(category_dict[word])

                if infobox_dict[word]:
                    posting += 'i' + str(infobox_dict[word])

                if link_dict[word]:
                    posting += 'l' + str(link_dict[word])

                if reference_dict[word]:
                    posting += 'r' + str(reference_dict[word])

                posting += ';'

                index_map[word] += posting

        num_pages += 1

        if not num_pages % 40000:
            self.write_data.write_intermed_index()
            self.write_data.write_id_title_map()

            index_map = defaultdict(str)
            id_title_map = {}
