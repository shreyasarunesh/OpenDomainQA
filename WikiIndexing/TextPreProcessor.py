'''
 *
 * Name                   :   F21MP- Open Domain Question Answering (ODQA) Agent.
 *
 * Description            :  This class illustrates the text preprocessing while indexing Wikipedia dump
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

'''



removing stopwords, tokenise the text, remove HTML tags, remove non-ASCII characters, etc. It is shown in the below code.
'''


class TextPreProcessor:
    """
       *
       * Summary :    This class takes the raw text as input and returns the preprocessed text the list of words.
       *                Ex- raw text ----> list of words
       *
       """

    def __init__(self, html_tags, stem_words, stop_words):


        self.html_tags = html_tags
        self.stem_words = stem_words
        self.stop_words = stop_words
    '''
     *
     *  Summary : This function reomoves all the stopwords from the given text data by 
                    iterating through the stopwords file
     *
     *  Args    : Param - text_data
     *
     *  Returns : returns cleaned data eliminating the  stop-words.
     *
    '''

    def remove_stopwords(self, text_data):

        cleaned_text = [word for word in text_data if word not in self.stop_words]

        return cleaned_text
    '''
     *
     *  Summary : This function iterates through the stem words file and returns the removes the sufix and prefix.
     *
     *  Args    : Param - single word 
     *
     *  Returns : returns cleaned word removing the stem words.
     *
    '''

    def stem_word(self, word):

        for wrd in self.stem_words:
            if word.endswith(wrd):
                word = word[:-len(wrd)]
                return word

        return word
    '''
     *
     *  Summary : This function removes all the stopwords from the given text data by 
                    iterating through the stem words in the given text. 
     *
     *  Args    : Param - text_data
     *
     *  Returns : returns cleaned word removing the stem words.
     *
    '''

    def stem_text(self, text_data):

        cleaned_text = [self.stem_word(word) for word in text_data]

        return cleaned_text

    '''
       *
       *  Summary : This function removes all the non ASCII characters using regular expression module
       *
       *  Args    : Param - text_data
       *
       *  Returns : returns cleaned data removing the ASCII characters. Ex:( " ",: ^ /) etc...
       *
      '''

    def remove_non_ascii(self, text_data):

        cleaned_text = ''.join([i if ord(i) < 128 else ' ' for i in text_data])

        return cleaned_text

    '''
       *
       *  Summary : This function removes all the HTML tags using regular expression module
       *
       *  Args    : Param - text_data
       *
       *  Returns : returns cleaned data removing the html tags. Ex:(<p>, <!--...-->, <hr>, <img>) etc ...
       *
      '''

    def remove_html_tags(self, text_data):

        cleaned_text = re.sub(self.html_tags, ' ', text_data)

        return cleaned_text

    '''
       *
       *  Summary : This function removes all the unnecessary empty space in the text data
       *
       *  Args    : Param - text_data
       *
       *  Returns : returns cleaned data removing the empty space. Ex:(' ')
       *
      '''

    def remove_special_chars(self, text_data):

        cleaned_text = ''.join(ch if ch.isalnum() else ' ' for ch in text_data)

        return cleaned_text
    '''
       *
       *  Summary : This function removes other special characters that links to external pages in the input text data. 
       *
       *  Args    : Param - text_data
       *
       *  Returns : returns cleaned data without links. Ex:(http, https)
       *
      '''
    def remove_select_keywords(self, text_data):

        text_data = text_data.replace('\n', ' ').replace('File:', ' ')
        text_data = re.sub('(http://[^ ]+)', ' ', text_data)
        text_data = re.sub('(https://[^ ]+)', ' ', text_data)

        return text_data

    '''
       *
       *  Summary : This function calls the above functions to remove non ascii character, 
       *            remove html tags, remove special character.
       *
       *  Args    : Param - text_data
       *
       *  Returns : returns cleaned data.
       *
      '''

    def tokenize_sentence(self, text_data, flag=False):

        if flag:
            text_data = self.remove_select_keywords(text_data)
            text_data = re.sub('\{.*?\}|\[.*?\]|\=\=.*?\=\=', ' ', text_data)
        cleaned_text = self.remove_non_ascii(text_data)
        cleaned_text = self.remove_html_tags(cleaned_text)
        cleaned_text = self.remove_special_chars(cleaned_text)

        return cleaned_text.split()

    '''
       *
       *  Summary : This is the main function to tokenize the input text data. 
       *            This functions calls all the above functions 
       *
       *  Args    : Param - text_data
       *
       *  Returns : returns cleaned data. 
       *
      '''

    def preprocess_text(self, text_data, flag=False):

        cleaned_data = self.tokenize_sentence(text_data.lower(), flag)
        cleaned_data = self.remove_stopwords(cleaned_data)
        cleaned_data = self.stem_text(cleaned_data)

        return cleaned_data