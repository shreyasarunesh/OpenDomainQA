

'''
 *
 * Name                   :   F21MP- Open Domain Question Answering (ODQA) Agent.
 *
 * Description            :  This class illustrates the SAX parcer to process the wikipedia pages in the dump.
                            This class uses only two fields, i.e. title and text, from XML in the above code.
 *
 * Author                 :   Shreyas Arunesh
 *
 *
 * Reference              : https://github.com/neufang/wiki-Dump-Indexing/blob/master/buildInvertedIndexFromWikiDump.py
'''
import xml.sax
from collections import defaultdict

index_map = defaultdict(str)
num_files = 0
num_pages = 0
id_title_map = {}

class XMLParser(xml.sax.ContentHandler):
    """
         *
         * Summary :    This class parses the xml, takes input text, preprocesses it and writes in initial indexing files.
                          This class uses PageProcessor() and CreateIndex() under the hood.
                          Ex - input xml ----> inidial index files
         *
         Args:        Wikipedia Preprocessed page and index creator class.
         *
      """

    def __init__(self, page_processor, create_index):

        self.tag = ''
        self.title = ''
        self.text = ''
        self.page_processor = page_processor
        self.create_index = create_index

    '''
     *
     *  Summary : This function creates the start element for the initial index.
     *
     *  Args    : Param - name of the page. 
     *
     *  Returns : Dosent retunrs anything but creates the initial index. 
     *
    '''
    def startElement(self, name, attrs):

        self.tag = name

    '''
       *
       *  Summary : This function creates the end element for initial index with the fields specified earlier.
       *
       *  Args    : Param - name of the page. 
       *
       *  Returns : Dosent retunrs anything but creates the initial index. 
       *
      '''
    def endElement(self, name):

        if name == 'page':
            print('Number of Pages: {0}'.format(num_pages))

            id_title_map[num_pages] = self.title.lower()
            title, body, category, infobox, link, reference = self.page_processor.process_page(self.title, self.text)

            self.create_index.index(title, body, category, infobox, link, reference)

            self.tag = ""
            self.title = ""
            self.text = ""

    '''
      *
      *  Summary : This function formates the initial index as per their "titles" and "text".
      *
      *  Args    : Param - content on the page. 
      *
      *  Returns : Dosent retunrs anything but creates a format for initial index. 
      *
     '''
    def characters(self, content):

        if self.tag == 'title':
            self.title += content

        if self.tag == 'text':
            self.text += content
