from feature_extraction import FeatureExtraction
from os import listdir
from os.path import isfile, join
import csv
import shutil

class CreateCSV(object):
    def __init__(self, index_file, out_put_name, train_file, unlabel_file, init_flag):
        self.fe = FeatureExtraction(train_file)
        self.init_flag = init_flag
        self.unlabel_file = unlabel_file
        self.OUTPUT = out_put_name
        self.index_file = index_file
        self.HEADER = self.fe.HEADER
        self.files = []
        self.lables = []
        self.numeric_lables = {}
        self._load_labelled_file()
        self.unlabeled_files = []
        self._load_unlabelled_file()


    def _load_labelled_file(self):
        filename = self.index_file
        with open(filename) as f:
            for line in f:
                file, author = line.split('\t') 
                self.files.append(file)
                self.lables.append(author)
        tmp_dic = {}
        count = 0
        for l in self.lables:
            if l not in self.numeric_lables:
                self.numeric_lables[l] = count
                tmp_dic[l] = count
                count += 1
        print 'debug] tmp_dic {}'.format(tmp_dic)

    def _load_unlabelled_file(self):
        if self.init_flag:
            mypath = '../unlabeled'
            self.unlabeled_files = [f for f in listdir(mypath) if isfile(join(mypath, f))]
            with open('unlabeled.tsv', 'w') as f:
                for each in self.unlabeled_files:
                    f.write('../unlabeled/' + each + '\n')
        else:
            with open(self.unlabel_file, 'r') as f:
                for line in f:
                    self.unlabeled_files.append(line.strip())
        
        
    def gen_unlabeled_csv(self):
        header = self.HEADER + ['filename']
        count = 0
        row = []
        for file in self.unlabeled_files:
                file_org = file
                if self.init_flag:
                    file = '../unlabeled/'+file
                local_row = self.gen_feature_per_file(file)
                local_row += ['unlabeled/'+file_org]
                count += 1
                row.append(local_row)

        #print 'HEADER - > {}'.format(header)
        count = 1
        for each in row:
            #print 'Row#{0} - > {1}'.format(count, each)
            count += 1

        with open('unlabeled_set_iter6.csv', 'w') as csvoutput:
            writer = csv.writer(csvoutput, lineterminator='\n')
            header = [x.strip() for x in header]
            all = []
            all.append(header)
            for each in row:
                if len(each) == 0:
                    continue
                else:
                    all.append(each)
            writer.writerows(all)


    def gen_csv(self):
        header = self.HEADER + ['author'] + ['numeric_author']
        count = 0
        row = []
        for file in self.files:
                file = '../'+file
                local_row = self.gen_feature_per_file(file)
                local_row += [self.lables[count].strip()]
                local_row += [self.numeric_lables[self.lables[count]]]
                count += 1
                row.append(local_row)

        #print 'HEADER - > {}'.format(header)
        count = 1
        for each in row:
            #print 'Row#{0} - > {1}'.format(count, each)
            count += 1

        with open(self.OUTPUT, 'w') as csvoutput:
            writer = csv.writer(csvoutput, lineterminator='\n')
            header = [x.strip() for x in header]
            all = []
            all.append(header)
            for each in row:
                if len(each) == 0:
                    continue
                else:
                    all.append(each)
            writer.writerows(all)

    def gen_feature_per_file(self, file):
        print '[info]working on file --> {}'.format(file)
        row = []
        obj = self.fe
        row.append(obj.ave_word_freq(file))
        row.append(obj.number_count(file))
        row.append(obj.ave_syllable_count(file))
        row.append(obj.ave_bigram_nor_score(file))
        row.append(obj.ave_trigram_nor_score(file))
        row += obj.tf_idf_query(file)
        return row


if __name__ == '__main__':
    shutil.copy2('train_iter6.tsv', 'mini_train_iter6.tsv')
    obj = CreateCSV('mini_train_iter6.tsv', 'train_set_iter6.csv', 'train_iter6.tsv', 'unlabeled_iter6.tsv', False)
    obj.gen_csv()
    obj = CreateCSV('dev.tsv', 'dev_set_iter6.csv', 'train_iter6.tsv', 'unlabeled_iter6.tsv', False)
    obj.gen_csv()
    obj.gen_unlabeled_csv()
    '''
    obj = CreateCSV('mini_train.tsv', 'train_set.csv')
    obj.gen_csv()
    obj = CreateCSV('dev.tsv', 'dev_set.csv')
    obj.gen_unlabeled_csv()
    '''
