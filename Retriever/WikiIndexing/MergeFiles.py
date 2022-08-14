
'''
 *
 * Name                   :   F21MP- Open Domain Question Answering (ODQA) Agent.
 *
 * Description            :  This class illustrates the process of creating a final index.
 *
 *
 Indexing formate:      jonny-2314:t3b6i2r1;6432:t5c8b3i1;
                        Ex.:- johnny-2141:5;1232:1;5432:78;
                    'Johnny-563-1-3-4--2-4-6-'.split('-') ---> ['apple', '563', '1', '3', '4', '', '2', '4', '6', '']
 *
 * Author                 :   Shreyas Arunesh
 *
 *
 * Reference              : https://github.com/neufang/wiki-Dump-Indexing/blob/master/buildInvertedIndexFromWikiDump.py
'''
import os
from collections import defaultdict

index_map = defaultdict(str)
num_files = 0
num_pages = 0
id_title_map = {}


class MergeFiles():

    """
       *
       * Summary :    This class takes the intermediate files as input, merges the data and write the index to final files.
                        This class uses CreateIndex() under the hood.
                       Ex - intermediate index ----> final index
       *
       *
       Args:            num_itermed_files, write_data
       *
    """

    def __init__(self, num_itermed_files, write_data):

        self.num_itermed_files = num_itermed_files
        self.write_data = write_data

    '''
       *
       *  Summary : This function merges the intermediate index to create the final index. 
                      The intermediate and initial files are deleted at the end of this process. 
       *
       *  Args    : Param - intermediate files and write data class 
       *
       *  Returns : Creates the final index.
       *
      '''
    def merge_files(self):

        files_data = {}
        line = {}
        postings = {}
        is_file_empty = {i: 1 for i in range(self.num_itermed_files)}
        tokens = []

        i = 0
        while i < self.num_itermed_files:

            files_data[i] = open(f'C:/Users/shrey/Desktop/Wiki-Search-Engine-main/output_data/english_wiki_index/index_{i}.txt', 'r', encoding= 'UTF-8')
            line[i] = files_data[i].readline().strip('\n')
            postings[i] = line[i].split('-')
            is_file_empty[i] = 0
            new_token = postings[i][0]
            if new_token not in tokens:
                tokens.append(new_token)
            i += 1

        tokens.sort(reverse=True)
        num_processed_postings = 0
        data_to_merge = defaultdict(str)
        num_files_final = 0

        while sum(is_file_empty.values()) != self.num_itermed_files:

            token = tokens.pop()
            num_processed_postings += 1

            if num_processed_postings % 30000 == 0:
                num_files_final = self.write_data.write_final_files(data_to_merge, num_files_final)

                data_to_merge = defaultdict(str)

            # # Uncomment this if there is memory issue
            # if num_processed_postings%150000==0:
            # 	password = 'your_password'
            # 	command = 'sh -c "sync; echo 3 > /proc/sys/vm/drop_caches"'
            # 	shell_script = f"echo {password} | sudo -S {command}"
            # 	os.system(shell_script)
            # 	print('Cache cleared')

            i = 0
            while i < self.num_itermed_files:

                if is_file_empty[i] == 0:

                    if token == postings[i][0]:

                        line[i] = files_data[i].readline().strip('\n')
                        data_to_merge[token] += postings[i][1]

                        if len(line[i]):
                            postings[i] = line[i].split('-')
                            new_token = postings[i][0]

                            if new_token not in tokens:
                                tokens.append(new_token)
                                tokens.sort(reverse=True)

                        else:
                            is_file_empty[i] = 1
                            files_data[i].close()
                            print(f'Removing file {str(i)}')
                            os.remove(f'C:/Users/shrey/Desktop/Wiki-Search-Engine-main/output_data/english_wiki_index/index_{str(i)}.txt')
                i += 1

        num_files_final = self.write_data.write_final_files(data_to_merge, num_files_final)

        return num_files_final
