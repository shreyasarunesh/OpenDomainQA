
'''
 *
 * Name                   :   F21MP- Open Domain Question Answering (ODQA) Agent.
 *
 * Description            :  This is the main function for indexing Wikipedia dump  where all the classes are called inside this
                            class for creating wikipedia indexing.

                            Download the latest wikipedia dump from: https://dumps.wikimedia.org/enwiki/
 *
 *
 * Warning:                 "THIS PROCESS TAKES NEARLY 6 DAYS "
 *
 * Author                 :   Shreyas Arunesh
 *
 *
 * Reference              : https://github.com/neufang/wiki-Dump-Indexing/blob/master/buildInvertedIndexFromWikiDump.py
'''

import sys
import time
from collections import defaultdict
import tqdm

index_map = defaultdict(str)
num_files = 0
num_pages = 0
id_title_map = {}

from  Retriever.WikiIndexing.TextPreProcessor import *
from  Retriever.WikiIndexing.CreateIndex import *
from  Retriever.WikiIndexing.PageProcessor import *
from  Retriever.WikiIndexing.DataWriter import *
from  Retriever.WikiIndexing.MergeFiles import *
from  Retriever.WikiIndexing.SaxParcer import *


if __name__ == '__main__':

    start = time.time()

    '''
     *
     *  Summary : This creates the list of stopwords and stemwords. 
     *
     *  Args    : Param - input_file
     *
     *  Returns : list of stop words, stem words.
     *
    '''


    html_tags = re.compile('&amp;|&apos;|&gt;|&lt;|&nbsp;|&quot;')

    with open('stemwords.txt', 'r', encoding= 'UTF-8') as f:
        stemmer = [word.strip() for word in f]
    with open('stopwords.txt', 'r', encoding= 'UTF-8') as f:
        stop_words = [word.strip() for word in f]

    text_pre_processor = TextPreProcessor(html_tags, stemmer, stop_words)
    page_processor = PageProcessor(text_pre_processor)
    write_data = WriteData()
    create_index = CreateIndex(write_data)

    parser = xml.sax.make_parser()
    parser.setFeature(xml.sax.handler.feature_namespaces, False)
    xml_parser = XMLParser(page_processor, create_index)
    parser.setContentHandler(xml_parser)
    output = parser.parse(sys.argv[1])
    '''
     *
     *  Summary : This creates the intermediate indexing files.
     *
     *  Args    : Param:
     *
     *  Returns : Intermediate index.
     *
    '''

    write_data.write_intermed_index()
    write_data.write_id_title_map()

    '''
     *
     *  Summary : This creates the final indexing files.
     *
     *  Args    : Param:
     *
     *  Returns : Final index.
     *
    '''
    merge_files = MergeFiles(num_files, write_data)
    num_files_final = merge_files.merge_files()

    with open('../output_data/english_wiki_index/num_pages.txt', 'w', encoding= 'UTF-8') as f:
        f.write(str(num_pages))

    num_tokens_final = 0
    with open('../output_data/english_wiki_index/tokens_info.txt', 'r', encoding= 'UTF-8') as f:
        for line in f:
            num_tokens_final += 1

    with open('../output_data/english_wiki_index/num_tokens.txt', 'w', encoding= 'UTF-8') as f:
        f.write(str(num_tokens_final))

    char_list = [chr(i) for i in range(97, 123)]
    num_list = [str(i) for i in range(0, 10)]

    with open(f'../output_data/english_wiki_index/tokens_info.txt', 'r', encoding= 'UTF-8') as f:
        for line in tqdm(f):
            if line[0] in char_list:
                with open(f'../output_data/english_wiki_index/tokens_info_{line[0]}.txt', 'a', encoding= 'UTF-8') as t:
                    t.write(line.strip())
                    t.write('\n')

            elif line[0] in num_list:
                with open(f'../output_data/english_wiki_index/tokens_info_{line[0]}.txt', 'a', encoding= 'UTF-8') as t:
                    t.write(line.strip())
                    t.write('\n')

            else:
                with open(f'../output_data/english_wiki_index/tokens_info_others.txt', 'a', encoding= 'UTF-8') as t:
                    t.write(line.strip())
                    t.write('\n')

    for ch in tqdm(char_list):
        tok_count = 0
        with open(f'../output_data/english_wiki_index/tokens_info_{ch}.txt', 'r', encoding= 'UTF-8') as f:
            for line in f:
                tok_count += 1

        with open(f'../output_data/english_wiki_index/tokens_info_{ch}_count.txt', 'w', encoding= 'UTF-8') as f:
            f.write(str(tok_count))

    for num in tqdm(num_list):
        tok_count = 0
        with open(f'../output_data/english_wiki_index/tokens_info_{num}.txt', 'r', encoding= 'UTF-8') as f:
            for line in f:
                tok_count += 1

        with open(f'../output_data/english_wiki_index/tokens_info_{num}_count.txt', 'w', encoding= 'UTF-8') as f:
            f.write(str(tok_count))

    try:
        tok_count = 0
        with open('../output_data/english_wiki_index/tokens_info_others.txt','r', encoding= 'UTF-8') as f:
            tok_count += 1

        with open(f'../output_data/english_wiki_index/tokens_info_others_count.txt','w', encoding= 'UTF-8') as f:
            f.write(str(tok_count))
    except:
        pass

    os.remove('../output_data/english_wiki_index/tokens_info.txt')
    print('Total tokens', num_tokens_final)
    print('Final files', num_files_final)

    end = time.time()

    print('Finished in -', end - start)