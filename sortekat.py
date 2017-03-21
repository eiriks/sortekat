#!/usr/bin/python3
# coding: utf-8
__author__ = 'eirikstavelin'
__version__ = '0.0.1a'

from sklearn.externals import joblib  # for pickleing
from sklearn.svm import LinearSVC
from time import time
from nltk.metrics.agreement import AnnotationTask

#from nltk.metrics import ConfusionMatrix

import sys
start = time()

# settings
DATA_NAME = 'l2LinearSVC_handcurated_synthetic_media_scientist_compTest'
#DATA_NAME = 'NAK10'  # "small_test_set"  # used in saved model
DATA_PATH = 'saved_models'


# class L1LinearSVC(LinearSVC):
#     # this is needed as it is not in the piclked data
#     ''' This is how it is done in the algo-tester:
#     http://scikit-learn.org/stable/auto_examples/document_classification_20newsgroups.html'''  # noqa
#     def fit(self, X, y):
#         # The smaller C, the stronger the regularization.
#         # The more regularization, the more sparsity.
#         self.transformer_ = LinearSVC(penalty="l1",
#                                       dual=False, tol=1e-3)
#         X = self.transformer_.fit_transform(X, y)
#         return LinearSVC.fit(self, X, y)
#
#     def predict(self, X):
#         X = self.transformer_.transform(X)
#         return LinearSVC.predict(self, X)


class SorteKat:
    '''Class to classify news text into categories
    This class gives you three models based on three
    different algorithms MultinomialNB, l2LinearSVC & SVC,
    all from the sklearn library.
    '''

    def __init__(self): # , flavour='MultinomialNB'
        self.labels = joblib.load('%s/%s_labels.pkl' % (DATA_PATH, DATA_NAME))
        self.clf = joblib.load('%s/%s.pkl' % (DATA_PATH, DATA_NAME))  # noqa
        # self.flavour = flavour

    def predict_one(self, text):
        return self.labels[self.clf.predict([text])]

    def predict_many(self, texts):
        '''Excpect a list of texts'''
        return self.labels[self.clf.predict(texts)]
        #return [self.labels[x] for x in self.clf.predict(texts)]
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
    from pympler import asizeof

    clf = SorteKat()
    print(asizeof.asizeof(clf))
    print("Laste classifyer tar lang tid:", time()-start, 'sekunder')

    print()
    print('It took', time()-start, 'seconds.')
    t1 = time()
    texts = ['Hardangervidda er midelertidig stengt mens brøytamanskap måker veien etter kraftig snøvær. Nedbør og sol. Solfaktor. Snø. Nedbør. Vind.',
         "Statsminister Stoltenberg besøkte bedrifter på vestlandet.",
        "Brann vant kveldens kamp not Drammen.",
        "Arbeidsledighetstallene går ned viser nye tall fra SSB.",
        "Ny forskning viser at profesorer forsker mer.",
        "Den nye storfilmen fra ",
        "– Eg kastar opp viss eg må gå på do der Over heile landet fortvilar bussjåførar over skitne toalett, manglande vatn og såpe eller ingen sanitære forhold i det heile."]
    print("Men klassifisering er raskt")
    pred = clf.predict_one(texts[0])
    print("..."+texts[0]+"...")
    print("er", pred[0], "og det tok bare", time()-t1, "sekunder")
    t1 = time()
    print()
    preds = clf.predict_many(texts)
    for p, t in zip(preds, texts):
        print(p,"==>", t)
    print()
    print("og", len(texts), "tok bare", time()-t1, "sekunder")
