from os import listdir
from os.path import isfile, join

def _load_unlabelled_file(self):
        mypath = '../unlabeled'
        self.unlabeled_files = [f for f in listdir(mypath) if isfile(join(mypath, f))]
        with opne('unlabeled.tsv', 'w') as f:
            for each in self.unlabeled_files:
                f.write('../unlabeled' + each)
