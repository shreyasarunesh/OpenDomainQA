'''
 *
 * Name                   :   F21MP- Open Domain Question Answering (ODQA) Agent.
 *
 * Description            :  This class illustrates the Page preprocessing while indexing Wikipedia dump.
                                using six fields: title, body, category, infobox, links, and references.
                                this is for generic queries or field-specific queries using these fields.
 *
 * Author                 :   Shreyas Arunesh
 *
 *
 * Reference              : https://github.com/neufang/wiki-Dump-Indexing/blob/master/buildInvertedIndexFromWikiDump.py
'''


from collections import defaultdict
from Retriever.WikiIndexing.TextPreProcessor import *

index_map = defaultdict(str)
num_files = 0
num_pages = 0
id_title_map = {}
text_pre_processor = TextPreProcessor()


class PageProcessor():

    """
       *
       * Summary :    This class takes the raw wikipedia page as input and seggregates the text in difference sections.
                        This class uses TextPreProcessor() class in this class.
                        Ex - input_text ----> title, body, cateogry, infobox, link, reference
       *
       *
       Args:            text pre processor class
       *
    """

    def __init__(self, text_pre_processor):

        self.text_pre_processor = text_pre_processor

    '''
     *
     *  Summary : This function cleanes the title text data. 
     *
     *  Args    : Param - Wikipedia page title
     *
     *  Returns : returns cleaned title.
     *
    '''
    def process_title(self, title):

        cleaned_title = self.text_pre_processor.preprocess_text(title)

        return cleaned_title

    '''
     *
     *  Summary : This function cleanes the information box text data. 
     *
     *  Args    : Param - Wikipedia info box text
     *
     *  Returns : returns cleaned infor box text.
     *
    '''
    def process_infobox(self, text):

        cleaned_infobox = []
        try:
            text = text.split('\n')
            i = 0
            while '{{Infobox' not in text[i]:
                i += 1

            data = []
            data.append(text[i].replace('{{Infobox', ' '))
            i += 1

            while text[i] != '}}':
                if '{{Infobox' in text[i]:
                    dt = text[i].replace('{{Infobox', ' ')
                    data.append(dt)
                else:
                    data.append(text[i])
                i += 1

            infobox_data = ' '.join(data)

            cleaned_infobox = self.text_pre_processor.preprocess_text(infobox_data)
        except:
            pass

        return cleaned_infobox

    '''
     *
     *  Summary : This function cleanes the raw text body of the Wikipedia page. 
     *
     *  Args    : Param - Wikipedia info box text
     *
     *  Returns : returns cleaned info box text.
     *
    '''
    def process_text_body(self, text):

        cleaned_text_body = []
        cleaned_text_body = text_pre_processor.preprocess_text(text, True)

        return cleaned_text_body

    '''
     *
     *  Summary : This function cleanes the categories Wikipedia page. 
     *
     *  Args    : Param - Wikipedia category_text
     *
     *  Returns : returns cleaned cleaned text.
     *
    '''

    def process_category(self, text):

        cleaned_category = []
        try:
            text = text.split('\n')
            i = 0
            while not text[i].startswith('[[Category:'):
                i += 1

            data = []
            data.append(text[i].replace('[[Category:', ' ').replace(']]', ' '))
            i += 1

            while text[i].endswith(']]'):
                dt = text[i].replace('[[Category:', ' ').replace(']]', ' ')
                data.append(dt)
                i += 1

            category_data = ' '.join(data)
            cleaned_category = self.text_pre_processor.preprocess_text(category_data)
        except:
            pass

        return cleaned_category

    '''
      *
      *  Summary : This function process the links of Wikipedia page. 
      *
      *  Args    : Param - Wikipedia iexternal links in the  body.
      *
      *  Returns : returns cleaned text removing the external links.
      *
     '''

    def process_links(self, text):

        cleaned_links = []
        try:
            links = ''
            text = text.split("==External links==")

            if len(text) > 1:
                text = text[1].split("\n")[1:]
                for txt in text:
                    if txt == '':
                        break
                    if txt[0] == '*':
                        text_split = txt.split(' ')
                        link = [wd for wd in text_split if 'http' not in wd]
                        link = ' '.join(link)
                        links += ' ' + link

            cleaned_links = self.text_pre_processor.preprocess_text(links)
        except:
            pass

        return cleaned_links

    '''
      *
      *  Summary : This function process the references Wikipedia page. 
      *
      *  Args    : Param - input text
      *
      *  Returns : returns cleaned references.
      *
     '''

    def process_references(self, text):

        cleaned_references = []
        try:
            references = ''
            text = text.split('==References==')

            if len(text) > 1:
                text = text[1].split("\n")[1:]
                for txt in text:
                    if txt == '':
                        break
                    if txt[0] == '*':
                        text_split = txt.split(' ')
                        reference = [wd for wd in text_split if 'http' not in wd]
                        reference = ' '.join(reference)
                        references += ' ' + reference

            cleaned_references = self.text_pre_processor.preprocess_text(references)
        except:
            pass

        return cleaned_references

    '''
      *
      *  Summary : This is the main function that calls all the above function.  
      *
      *  Args    : Param - wikipedia title and text.
      *
      *  Returns : returns cleaned and processed title, body, category, infobox, link and reference of the wikipedia page.
      *
     '''

    def process_page(self, title, text):

        title = self.process_title(title)
        body = self.process_text_body(text)
        category = self.process_category(text)
        infobox = self.process_infobox(text)
        link = self.process_links(text)
        reference = self.process_references(text)

        return title, body, category, infobox, link, reference