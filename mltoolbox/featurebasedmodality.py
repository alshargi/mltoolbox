
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.utils import shuffle
import pandas as pd
import unicodedata as ud
import matplotlib.pyplot as plt


def untag(tagged_sentence):
	return [w for w, t in tagged_sentence]

def loadUnqList(p):
    klist = []
    with open(p) as fword:
        klist = fword.read().splitlines()
    return klist


def check_punctuation(xstr):
    res = False
    for i in xstr: 
        if ud.category(i).startswith('P'):
            res = True
    return res

def check_capitalized(xstr):
    res = False
    for f in xstr:
        if f.isupper():
            res = True
            break
    return res

def check_numeric(xstr):
    res = False
    for f in xstr:
        if f.isdigit():
            res = True
            break
    return res


def check_dot(xstr):
    res =  False
    res_dot = False
    res_cap = False
    cc = xstr.split(" ")
    for c in cc:
        if len(c) == 2:
            if '.' in c:
                res_dot = True
            if c.isupper() == False:
                res_cap = True
    if res_dot == True and  res_cap == True:
        res = True
    else:
        res = False
    return res



def check_single_lett_cap(xstr):
    res = False
    fg = xstr.split(' ')
    for f in fg:
        if len(f) == 1 and f.isupper():
            res = True
            break
    return res

def chec_frs_lett_cap(xstr):
    res = False
    tco = ''
    sff = xstr.split(' ')
    for f in sff:
        tco += f[0]
        #print(tco)
    if tco.upper() == tco:
        res = True

    return res

def transform_to_dataset(tagged_sentences):
    Xx, yy = [], []
    for tagged in tagged_sentences:
        Xx.append(features(tagged[0]))
        yy.append(tagged[1])
    return Xx, yy


def check_bbrivation(xstr):
    res = False
    pp = []
    dk = xstr.split(" ")
    
    for d in dk:
        res = d.isupper()
        pp.append(res)
        
    if True in pp and False in pp:
        res = True
        
    return res


def check_if_ascii(xstr):
    return xstr.isascii()

def check_if_isalnum(xstr):
    return xstr.isalnum()


def check_hyphen(xstr):
    res = False
    islower = False
    ishvnin = False
    btween = False

    if xstr.lower() == xstr:
        islower = True
        
    if '-' in xstr:
        ishvnin = True
            
    cc = xstr.split(" ")
    for c in cc:
        if "-" in c and len(c) > 2:
            btween = True
            break
        
    if islower == True and ishvnin == True   and btween == True :
       res = True
   
    return res



def features(sentence):
	return {
    	'entry': sentence,
        'is_capitalized': check_capitalized(sentence),
        'has_abbriv': check_bbrivation(sentence),
        #'is_all_caps': sentence.upper() == sentence,
        #'is_all_lower': sentence.lower() == sentence,
        'has_case': sentence.upper() != sentence.lower(),
        'has_hyphen' : check_hyphen(sentence),
        'is_ascii' : check_if_ascii(sentence),
        'is_alphanum' : check_if_isalnum(sentence),
        'has_dot': check_dot(sentence),
        'has_punctuation':  check_punctuation(sentence),
        'is_numeric': check_numeric(sentence),
        'is_sing_lett_cap':  check_single_lett_cap(sentence),
        	}





