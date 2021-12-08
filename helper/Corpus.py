from gensim import utils

class MyCorpus(object):
    def __init__(self, corpus_name):
        self.corpus_name = corpus_name
    
    """An iterator that yields sentences (lists of str)."""

    def __iter__(self):

        for line in open(self.corpus_name):
            
            yield utils.simple_preprocess(line)












