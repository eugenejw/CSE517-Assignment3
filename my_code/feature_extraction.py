from corpus_loading import Corpus
from nltk.corpus import cmudict
from pyngram import calc_ngram
import unittest
import nltk
import string
import re
import math

class FeatureExtraction(object):
    '''
    extract feature from data files

    '''
    def __init__(self, train_data_file):
        '''
        init
        '''
        self.unigram = {}
        self.cmudict = cmudict.dict()
        self.bigram_nor_dic, self.trigram_nor_dic = self._build_23_gram_normality_dic()
        self.unigram = self._load_unigram_corpus()
        self.train_dic = {}
        self._load_csv_file(train_data_file)
        self.stopwords = nltk.corpus.stopwords.words('english')
        #create tf_idf dic
        self.tf_idf_raw_dic = {}

        self._create_init_tfidf_set()
        #init tf table for each doc, {doc1:tf_dic, ...}
        self.DOC_LIST = self.tf_idf_raw_dic.keys() #[x.strip() for x in self.tf_idf_raw_dic.keys()]
        self.HEADER = ['ave_word_freq', '#_of_numbers', 'ave_syllables', 'ave_2gram_nor', 'ave_3gram_nor'] + ['tf_idf'+x for x in self.DOC_LIST]
        #print "HEADER is {}".format(self.HEADER)
        self.tf_tables = {}
        #init idf cross all docs
        self.idf_cross_docs = {}
        #fill tf_dif
        self.gen_tf_idf_data()

    def _create_init_tfidf_set(self):
        for author in self.train_dic:
            self.tf_idf_raw_dic[author] = []
            for file in self.train_dic[author]:
                #auto adding ../
                file = '../'+file
                with open(file) as f:
                    for line in f:
                        exclude = set(string.punctuation)
                        #remove all the ounctuations
                        line = ''.join(ch for ch in line if ch not in exclude)
                        #tokenize
                        tokenized_lst = self.tokenize(line)
                        self.tf_idf_raw_dic[author] += tokenized_lst
                        

    def _build_23_gram_normality_dic(self):
        instance_d = Corpus(True)
        return (instance_d.bigram_normality_dic, instance_d.trigram_normality_dic)

    def _load_csv_file(self, filename):
        #unified author name
        with open(filename) as f:
            for line in f:
                file, author = line.split('\t') 
                author = author.strip()
                if author not in self.train_dic:
                    self.train_dic[author] = []
                    self.train_dic[author].append(file)
                else:
                    self.train_dic[author].append(file)

    def _load_unigram_corpus(self):
        instance_d = Corpus(True)
        corpus = instance_d.corpus
        return corpus

    #interface
    def get_tf_idf_raw_dic(self):
        '''
        self.tf_idf_dic's key is author, value is raw words
        '''
        return self.tf_idf_raw_dic

    def gen_tf_idf_data(self):
        self.idf_cross_docs = {}
        #create the idf table
        def inverseDocumentFrequency(term):
            allDocuments = self.tf_idf_raw_dic.values()
            numDocumentsWithThisTerm = 0
            for doc in allDocuments:
                if term in doc:
                    numDocumentsWithThisTerm = numDocumentsWithThisTerm + 1
            if numDocumentsWithThisTerm > 0:
                return 1.0 + math.log(float(len(allDocuments)) / numDocumentsWithThisTerm)
            else:
                return 1.0
        cat_doc = []
        for val in self.tf_idf_raw_dic.itervalues():
            cat_doc += val
            
        for word in cat_doc:
            if word not in self.idf_cross_docs:
                self.idf_cross_docs[word] = inverseDocumentFrequency(word)

        #create tf table for each doc
        def termFrequency(term, document):
            return document.count(term) / float(len(document))
        for doc in self.tf_idf_raw_dic.iterkeys():
            if doc not in self.tf_tables:
                self.tf_tables[doc] = {}
            for w in self.tf_idf_raw_dic[doc]:
                if w not in self.tf_tables[doc]:
                    self.tf_tables[doc][w] = termFrequency(w, self.tf_idf_raw_dic[doc])

    def tf_idf_query(self, file):
        '''
        [doc1, doc2, ...]
        return a list of tf*idf for each doc
        [0.1, 0.2, ... ]
        '''
        ret = [0]*len(self.DOC_LIST)
        with open(file) as f:
            for line in f:
                exclude = set(string.punctuation)
                #remove all the ounctuations
                line = ''.join(ch for ch in line if ch not in exclude)
                tokenized_lst = self.tokenize(line)
                L = len(self.DOC_LIST)
                total_tf_dif = [0]*L
                for w in tokenized_lst:
                    for i in xrange(L):
                        doc = self.DOC_LIST[i]
                        #tf
                        if w in self.tf_tables[doc]:
                            tf = self.tf_tables[doc][w]
                        else:
                            tf = 0
                        #idf
                        if w in self.idf_cross_docs:
                            idf = self.idf_cross_docs[w]
                        else:
                            idf = 1.0
                        ret[i] += tf*idf 
        return ret


    #feature #1
    def ave_word_freq(self, file):
        '''
        for each sentence, after removing the punctuations and stopwords,
        get the average word freqence appeared in Google Corpus
        '''
        ave_word_freq = 0
        total = 0
        with open(file) as f:
            for line in f:
                #print '[debug] line is {}'.format(line)
                exclude = set(string.punctuation)
                #remove all the ounctuations
                line = ''.join(ch for ch in line if ch not in exclude)
                tokenized_lst = self.tokenize(line)
                if tokenized_lst:
                    for w in tokenized_lst:
                        if w in self.unigram:
                            total += self.unigram[w]
                            #print 'word: {0} has count {1} in unigram corpus.'.format(w, self.unigram[w])
                        else:
                            #print 'word: {0} has count not found in unigram corpus.'.format(w)
                            total += 1
                else:
                    return 1
                #print 'result is {0} / {1}'.format(total, float(len(tokenized_lst)))
                ave_word_freq = total/float(len(tokenized_lst))

        return ave_word_freq
    
    #feature #2
    def punctuation_count(self, file):
        '''
        return (#of?, #of-, #of!, #of:)
        '''
        p_lst = [0]*4
        with open(file) as f:
            for line in f:
                for w in line:
                    if w == '?':
                        p_lst[0] += 1
                    if w == '-':
                        p_lst[1] += 1
                    if w == '!':
                        p_lst[2] += 1
                    if w == ':':
                        p_lst[3] += 1
        return tuple(p_lst)

    #feature #3
    def number_count(self, file):
        '''
        count how many times number is mentioned
        return int
        '''
        ret = 0
        with open(file) as f:
            for line in f:
                ret = [w.isdigit() for w in line].count(True)
        return ret

    #feature #4
    def ave_syllable_count(self, file):
        '''
        count how many times number is mentioned
        return int
        self.cmudict
        '''
        ret = 0
        total = 0
        length = 0
        def nsyl(word):
            if word in self.cmudict:
                return [len(list(y for y in x if y[-1].isdigit())) for x in self.cmudict[word.lower()]]
            else:
                return [1]

        with open(file) as f:
            for line in f:
                exclude = set(string.punctuation)
                #remove all the ounctuations
                line = ''.join(ch for ch in line if ch not in exclude)
                #tokenize
                tokenized_lst = self.tokenize(line)
                if tokenized_lst:
                #print 'tokenized_lst is {}'.format(tokenized_lst)
                    for w in tokenized_lst:
                        #print 'for word {0}: nsyl->{1}'.format(w, nsyl(w)[0])
                        total += nsyl(w)[0]
                        length += 1                    
                else:
                    length = 1
        #print 'deviding {0} / {1}'.format(total, float(length))
        return total/float(length)


    #feature #5
    def ave_bigram_nor_score(self, file):
        '''
        count how many times number is mentioned
        return int
        self.cmudict
        '''
        ret = 0
        total = 0
        L = 0
        def nsyl(word):
            return [len(list(y for y in x if y[-1].isdigit())) for x in self.cmudict[word.lower()]]

        with open(file) as f:
            for line in f:
                exclude = set(string.punctuation)
                #remove all the ounctuations
                line = ''.join(ch for ch in line if ch not in exclude)
                #tokenize
                tokenized_lst = self.tokenize(line)
                #print 'tokenized_lst is {}'.format(tokenized_lst)
                for w in tokenized_lst:
                    
                    #>>> calc_ngram('gooogle', 2)
                    #[('oo', 2), ('go', 1), ('gl', 1), ('le', 1), ('og', 1)]
                    ngram_lst = calc_ngram(w, 2)
                    local_total = 0
                    local_L = 0
                    if ngram_lst:
                        for each in ngram_lst:
                            if each[0] in self.bigram_nor_dic:
                                #print 'self.bigram_nor_dic[each[0]] is {0} of type {1}'.format(self.bigram_nor_dic[each[0]], type(self.bigram_nor_dic[each[0]]))
                                local_total += (self.bigram_nor_dic[each[0]]*each[1])
                                #print 'for 2-gram {0}: count->{1}'.format(each[0], self.bigram_nor_dic[each[0]])
                            else:
                                #print '[Warning]{} not found in bigram_nor_dic'.format(each[0])
                                local_total += (1*each[1])
                            local_L += each[1]
                    else:
                        local_L = 1
                    #print 'for word {0}: local_total->{1}, local_L-> {2}'.format(w, local_total, local_L)
                    total += (local_total/float(local_L))
                    L += 1
        if L==0:
            L =1
        return total/float(L)

    #feature #6
    def ave_trigram_nor_score(self, file):
        '''
        count how many times number is mentioned
        return int
        self.cmudict
        '''
        ret = 0
        total = 0
        L = 0
        def nsyl(word):
            return [len(list(y for y in x if y[-1].isdigit())) for x in self.cmudict[word.lower()]]

        with open(file) as f:
            for line in f:
                exclude = set(string.punctuation)
                #remove all the ounctuations
                line = ''.join(ch for ch in line if ch not in exclude)
                #tokenize
                tokenized_lst = self.tokenize(line)
                #print 'tokenized_lst is {}'.format(tokenized_lst)
                for w in tokenized_lst:
                    
                    #>>> calc_ngram('gooogle', 2)
                    #[('oo', 2), ('go', 1), ('gl', 1), ('le', 1), ('og', 1)]
                    ngram_lst = calc_ngram(w, 3)
                    local_total = 0
                    local_L = 0
                    if ngram_lst:
                        for each in ngram_lst:
                            if each[0] in self.trigram_nor_dic:
                            #print 'self.trigram_nor_dic[each[0]] is {0} of type {1}'.format(self.trigram_nor_dic[each[0]], type(self.trigram_nor_dic[each[0]]))
                                local_total += (self.trigram_nor_dic[each[0]]*each[1])
                            #print 'for 2-gram {0}: count->{1}'.format(each[0], self.trigram_nor_dic[each[0]])
                            else:
                            #print '[Warning]{} not found in trigram_nor_dic'.format(each[0])
                                local_total += (1*each[1])
                            local_L += each[1]
                    #print 'for word {0}: local_total->{1}, local_L-> {2}'.format(w, local_total, local_L)
                    else:
                        local_L = 1
                    total += (local_total/float(local_L))
                    L += 1
        if L == 0:
            L = 1
        return total/float(L)
                    
    def tokenize(self, sentence):
        '''
        tokenize, and then remove stopwords
        input: string
        return list of strings
        '''
        sentence = sentence.lower()
        tokens = []
        tokens = nltk.word_tokenize(sentence)
        tokens = [x for x in tokens if x not in self.stopwords]
        
        return tokens


    def _unittest_return_csv_dic(self):
        return self.train_dic                
    def _unittest_return_header(self):
        return self.HEADER                

    def gen_feature_per_file(self, file):
        row = []
        obj = FeatureExtraction('train_iter1.tsv')
        row.append(obj.ave_word_freq(file))
        row.append(obj.number_count(file))
        row.append(obj.ave_syllable_count(file))
        row.append(obj.ave_bigram_nor_score(file))
        row.append(obj.ave_trigram_nor_score(file))
        row += obj.tf_idf_query(file)
        return row
        
        
        

        
        
        

class TestStringMethods(unittest.TestCase):
    '''
    Unittest cases
    '''
    '''
    def test_training_set_duplicated_values(self):
        obj = FeatureExtraction()
        dic = obj._unittest_return_csv_dic()
        flag = True
        for key in dic.keys():
            flag = True if len(dic[key])==len(set(dic[key])) else False
            if flag != True:
                break
        self.assertEqual(flag, True)

    def test_tokenize(self):
        obj = FeatureExtraction()
        res = []
        res = obj.tokenize('the cat')
        self.assertEqual(res, ['cat'])

    def test_feature1(self):
        flag = True
        obj = FeatureExtraction()
        try:
            ret = obj.ave_word_freq('../labeled/00013.txt')
        except Exception:
            flag = False
        self.assertEqual(flag, True)
        print '[Unittest_Feature#1]PASSED. For labeled/00013.txt, the ave_word_freq is {}'.format(ret)

    def test_feature2(self):
        flag = True
        obj = FeatureExtraction()
        try:
            ret = obj.punctuation_count('../labeled/00013.txt')
        except Exception:
            flag = False
        self.assertEqual(flag, True)
        print '[Unittest_Feature#2]PASSED. For labeled/00013.txt, the p_lst is {}'.format(ret)

    def test_feature3(self):
        flag = True
        obj = FeatureExtraction()
        ret = obj.number_count('../labeled/00013.txt')
        print '[Unittest_Feature#2]PASSED. For labeled/00013.txt, the # of Number is {}'.format(ret)

    def test_feature4(self):
        obj = FeatureExtraction()
        ret = obj.ave_syllable_count('../labeled/00013.txt')
        print '[Unittest_Feature#2]PASSED. For labeled/00013.txt, the ave # of syllables is {}'.format(ret)

    def test_feature5(self):
        obj = FeatureExtraction()
        ret = obj.ave_bigram_nor_score('../labeled/00013.txt')
        print '[Unittest_Feature#5]PASSED. For labeled/00013.txt, the ave bigram normality is {}'.format(ret)

    def test_feature6(self):
        obj = FeatureExtraction()
        ret = obj.ave_trigram_nor_score('../labeled/00013.txt')
        print '[Unittest_Feature#6]PASSED. For labeled/00013.txt, the ave trigram normality is {}'.format(ret)

    def test_feature0(self):
        obj = FeatureExtraction()
        ret = obj.get_tf_idf_raw_dic()
        print '[Unittest_Feature#0]PASSED. For labeled/00013.txt, the tf_idf_dic has following keys {}'.format(ret.keys())

    def test_load_corpus_check_size(self):
        obj = FeatureExtraction()
        corpus = obj.unigram
        self.assertEqual(len(corpus), 332699)

    def test_tf_idf_query(self):
        obj = FeatureExtraction()
        ret = obj.tf_idf_query('../labeled/00013.txt')
        print '[Unittest_Feature#7]PASSED. For labeled/00013.txt, the sum of tf_idf table for this doc is {}'.format(ret)

    '''
    def test_gen_feature_per_file(self):
        obj = FeatureExtraction('train_iter1.tsv')
        ret = obj.gen_feature_per_file('../labeled/00013.txt')
        print '[Unittest_Feature#8]PASSED. For labeled/00013.txt, the head for this doc is {}'.format([x.strip() for x in obj.HEADER])
        print '[Unittest_Feature#8]PASSED. For labeled/00013.txt, the feature set for this doc is {}'.format(ret)
        self.assertEqual(len(obj.HEADER), len(ret))
    

    def test_init_training_set_size(self):
        obj = FeatureExtraction('train_iter1.tsv')
        dic = obj._unittest_return_csv_dic()
        size = 0
        #for key in dic.keys():
        size += len(dic)
        print '[Unittest]Author in keys() is {}'.format(dic.keys())
        self.assertEqual(size, 19)


if __name__ == '__main__':
    unittest.main()
