#!/usr/bin/python2.7
# coding: utf-8
__author__ = 'eirikstavelin'
__version__ = '0.0.1a'

from sklearn.externals import joblib  # for pickleing
from sklearn.svm import LinearSVC
from time import time
from nltk.metrics.agreement import AnnotationTask
from nltk.metrics import ConfusionMatrix

import sys
start = time()

# settings
DATA_NAME = 'NAK10'  # "small_test_set"  # used in saved model
DATA_PATH = 'models_tained_NAK11_nelson'


class L1LinearSVC(LinearSVC):
    # this is needed as it is not in the piclked data
    ''' This is how it is done in the algo-tester:
    http://scikit-learn.org/stable/auto_examples/document_classification_20newsgroups.html'''  # noqa
    def fit(self, X, y):
        # The smaller C, the stronger the regularization.
        # The more regularization, the more sparsity.
        self.transformer_ = LinearSVC(penalty="l1",
                                      dual=False, tol=1e-3)
        X = self.transformer_.fit_transform(X, y)
        return LinearSVC.fit(self, X, y)

    def predict(self, X):
        X = self.transformer_.transform(X)
        return LinearSVC.predict(self, X)


class SorteKat:
    '''Class to classify news text into categories
    This class gives you three models based on three
    different algorithms MultinomialNB, l2LinearSVC & SVC,
    all from the sklearn library.
    '''

    def __init__(self, flavour='MultinomialNB'):
        self.labels = joblib.load('%s/%s_labels.pkl' % (DATA_PATH, DATA_NAME))
        self.clf = joblib.load('%s/%s_%s.pkl' % (DATA_PATH, flavour, DATA_NAME))  # noqa
        self.flavour = flavour

    def predict_one(self, text):
        return self.labels[self.clf.predict([text])]

    def predict_many(self, texts):
        '''Excpect a list of texts'''
        return [self.labels[x] for x in self.clf.predict(texts)]
        # print(self.clf.predict(texts))

    def predict_prob_test(self, text):
        return self.clf.predict_proba([text])

    def get_proba(self, text):
        probs = self.clf.predict_proba([text])[0]
        prob_table = list(zip(self.labels, probs))
        sorted_prob_table = sorted(prob_table, key=lambda tup: -tup[1])
        return sorted_prob_table



if __name__ == '__main__':
    print("Kjører fra terminal")
    CN = ClassifyNews("MultinomialNB")
    print("Opprettet classifyer", time()-start, 'sekunder')

    print()
    print('It took', time()-start, 'seconds.')
