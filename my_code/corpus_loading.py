import sys
import re
from os.path import join, dirname, realpath
import networkx as nx
from itertools import groupby, count
from math import log10
import copy
import unittest

if sys.hexversion < 0x03000000:
    range = xrange

def parse_file(filename):
    '''
    Global function that parses file and form a dictionary.
    '''
    with open(filename) as fptr:
        lines = (line.split('\t') for line in fptr)
        return dict((word, float(number)) for word, number in lines)
#UNIGRAM_COUNTS = parse_file(join(dirname(realpath(__file__)), 'corpus', 'unigrams.txt'))
UNIGRAM_COUNTS = parse_file(join(dirname(realpath(__file__)), 'corpus', 'unigrams.txt.original'))
BIGRAM_COUNTS = parse_file(join(dirname(realpath(__file__)), 'corpus', 'bigrams.txt'))

def as_range(group):
    '''
    Global function returns range
    '''
    tmp_lst = list(group)
    return tmp_lst[0], tmp_lst[-1]

class Corpus(object):
    '''
    Read corpus from path, and provide the following functionalities,
    1. data as "property", it is a dictionary where key is word,
       while the value is the frequency count of this word.
    2. generator that yield word and its frequency
    '''
    def __init__(self, use_google_corpus):
        self._unigram_counts = dict()
        self._use_google_corpus = use_google_corpus
        self._2gram_normality_dic = {} 
        self._3gram_normality_dic = {}
        self._build_2_gram_normality_dic()
        self._build_3_gram_normality_dic()

        if self._use_google_corpus:
            #use pure google corpus
            self._unigram_counts = parse_file(
                join(dirname(realpath(__file__)), 'corpus', 'filtered_1_2_letter_only.txt')
            )
        else:
            #use dictionary-filtered google corpus
            self._unigram_counts = parse_file(
                join(dirname(realpath(__file__)), 'corpus', 'unigrams.txt')
            )

    def _build_2_gram_normality_dic(self):
        dic_2gram = {}
        with open('corpus/count_2l.txt') as f:
            for line in f:
                pattern = re.search(r'(\w+).(.*)', line)
                gram = pattern.group(1)
                count = pattern.group(2)
                dic_2gram[gram] = int(count)
        #print '[debug] {}'.format(dic_2gram)
        self._2gram_normality_dic = dic_2gram

    def _build_3_gram_normality_dic(self):
        dic_3gram = {}
        with open('corpus/count_3l.txt') as f:
            for line in f:
                pattern = re.search(r'(\w+).(.*)', line)
                gram = pattern.group(1)
                count = pattern.group(2)
                dic_3gram[gram] = int(count)

        self._3gram_normality_dic = dic_3gram

    @property
    def bigram_normality_dic(self):
        '''
        return the whole dictionary out to user as a property.
        ngram_distribution = dict()
        instance_d = Corpus(self._use_google_corpus)
        corpus = instance_d.corpus
        '''
        return self._2gram_normality_dic

    @property
    def trigram_normality_dic(self):
        '''
        return the whole dictionary out to user as a property.
        ngram_distribution = dict()
        instance_d = Corpus(self._use_google_corpus)
        corpus = instance_d.corpus
        '''
        return self._3gram_normality_dic
        

    @property
    def corpus(self):
        '''
        return the whole dictionary out to user as a property.
        ngram_distribution = dict()
        instance_d = Corpus(self._use_google_corpus)
        corpus = instance_d.corpus
        '''
        return self._unigram_counts

    

    def __iter__(self):
        for each in self._unigram_counts.keys():
            yield each

class TestStringMethods(unittest.TestCase):

  def test_corpus_size(self):
      instance_d = Corpus(True)
      corpus = instance_d.corpus
      size = len(corpus)
      self.assertEqual(size, 332699)

  def test_load_2gram_normality(self):
      instance_d = Corpus(True)
      bigram_nor = instance_d.bigram_normality_dic
      #print bigram_nor
      self.assertIsNotNone(bigram_nor)


  def test_load_3gram_normality(self):
      instance_d = Corpus(True)
      trigram_nor = instance_d.trigram_normality_dic
      #print trigram_nor
      self.assertIsNotNone(trigram_nor)



if __name__ == '__main__':
    unittest.main()