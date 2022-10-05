import argparse
import codecs
import json
import re
import shutil
import scipy.spatial.distance as ssd
import subprocess
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.utils import shuffle
import pandas as pd
import unicodedata as ud
import matplotlib.pyplot as plt
import os
import joblib
import requests
from datetime import datetime
from colorama import Fore
from colorama import Style
import sys

import time
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.linear_model import LinearRegression





training_Entries = []
test_Entries =[]

def log(string):
    now = str(datetime.now())
    print(Fore.BLUE + now + ' ' + Style.RESET_ALL + string)

def popen_cmd(cmd):
    log(' '.join(cmd))
    p = subprocess.Popen(' '.join(cmd), text=True, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = p.communicate()[0:2]
    if p.returncode != 0:
        print(stdout)
        print(stderr)
    return stdout


def assume(role, region, account):
    cmd = ['isengardcli', 'creds', '--role', role, '--region', region, account]
    p = popen_cmd(cmd)
    for line in p.split('\n'):
        try:
            key = re.search(r'AWS_[A-Z_]+', line.split(' ')[1]).group()
            value = '='.join(line.split(' ')[1].split('=')[1:])
            os.environ[key] = value 
        except:
            pass

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
        #'has_abbriv': check_bbrivation(sentence),
        'is_all_caps': sentence.upper() == sentence,
        'is_all_lower': sentence.lower() == sentence,
        'has_case': sentence.upper() != sentence.lower(),
        'has_hyphen' : check_hyphen(sentence),
        'is_ascii' : check_if_ascii(sentence),
        'is_alphanum' : check_if_isalnum(sentence),
        'has_dot': check_dot(sentence),
        'has_punctuation':  check_punctuation(sentence),
        'is_numeric': check_numeric(sentence),
        'is_sing_lett_cap':  check_single_lett_cap(sentence),
        	}



# load and predict modality
def load_FeatureBased_w_sw_Model(xlink):
    xloaded_model_mod = joblib.load(xlink + '/models/fb_modality_2l_w_sw_sep_9_treng.sav')
    log("Model, loaded ")
    return xloaded_model_mod

def predict_modality_fb(clfx, input_entries):
    fb_result = []
    all_modality_mod = []
    log("predicting result ..")
    labels = {'s': 'SpokenOnly', 'w': 'WrittenOnly', 'sw': 'SpokenAndWritten'}
    for entry in  input_entries: 
        prd = clfx.predict(features(entry))[0]
        fb_result.append(labels[prd])
        all_modality_mod.append(labels[prd])
    log("Feature-Based Modality >> " + "#" *22) 
    
    return fb_result

# load and predict Rank
def Load_Rank_Mmodels(xlink):
    xloaded_model_mod = joblib.load(xlink + '/models/rank_model.sav')
    xloaded_cvec_mod = joblib.load(xlink + '/models/rank_countvectorizer.sav')
    xloaded_tfidf_transformer_mod= joblib.load(xlink + '/models/rank_tfidftransformer.sav')
    log("Models , loaded ")
    return xloaded_model_mod, xloaded_cvec_mod, xloaded_tfidf_transformer_mod

def Predict_Rank(model_mod, cvec_mod, tfidf_mod, input_entries):
    all_rank_mod = []
    rank_result = []
    rank_labels = {'0': 'Top', '1': 'Medium', '2': 'Low'}
    for entry in input_entries:
        new_x = [entry.strip()]
        x_val_vec_mod = cvec_mod.transform(new_x)
        X_val_tfidf_mod = tfidf_mod.transform(x_val_vec_mod)
        result_mod = model_mod.predict(X_val_tfidf_mod)
        rank_result.append(rank_labels[str(result_mod[0])])
        all_rank_mod.append(rank_labels[str(result_mod[0])])

    log("Rank >> " + "#" *22)

    return rank_result

# load and predict Types GIC
def Load_Types_GIC_Mmodels(xlink):
    xloaded_model_mod = joblib.load(xlink + '/models/Type_model_gci_v3.sav')
    xloaded_cvec_mod = joblib.load(xlink + '/models/Type_countvectorizer_gci_v3.sav')
    xloaded_tfidf_transformer_mod= joblib.load(xlink + '/models/Type_tfidftransformer_gci_v3.sav')
    log("Models , loaded ")
    return xloaded_model_mod, xloaded_cvec_mod, xloaded_tfidf_transformer_mod

def Predict_Types_GIC(model_mod, cvec_mod, tfidf_mod, input_entries): 
    types_result = []
    types_GIC_labels = {'0' : 'GTD-AlphaNumeric' , '1' : 'Intent-Cancel', '2' : 'Catalog-Country',
                       '3' : 'Catalog-City', '4' : 'GTD-Currency', '5' : 'GTD-Duration',
                       '6' : 'GTD-Date', '7' : 'GTD-DateInterval', '8' : 'GTD-EmailAddress',
                       '9' : 'Catalog-FirstName', '10' : 'Intent-Help', '11' : 'Catalog-LastName' ,
                       '12' : 'GTD-Number', '13' : 'Intent-No', '14' : 'GTD-Percentage', '15' : 'GTD-PhoneNumber',
                       '16' : 'Catalog-Airport', '17' : 'Intent-Pause', '18' : 'Intent-Resume', '19' : 'Intent-Repeat',
                       '20' : 'GTD-Speed', '21' : 'Intent-Stop' , '22' : 'Catalog-StreetName', '23' : 'Catalog-State' ,
                       '24' : 'Intent-StartOver', '25' : 'GTD-Time', '26' : 'GTD-Weight', '27' : 'Intent-Yes'}



    for entry in input_entries:
        new_x = [entry.strip()]
        x_val_vec_mod = cvec_mod.transform(new_x)
        X_val_tfidf_mod = tfidf_mod.transform(x_val_vec_mod)
        result_mod = model_mod.predict(X_val_tfidf_mod)
        types_result.append(types_GIC_labels[str(result_mod[0])])

    log("Types GIC >> " + "#" *22) 
    return types_result


# load and predict code-switching
def Load_Codeswitching_Mmodels(xlink):
    xloaded_model_mod = joblib.load(xlink + '/models/hiin_cs_model_v1.sav')
    xloaded_cvec_mod = joblib.load(xlink + '/models/hiin_cs_countvectorizer_v1.sav')
    xloaded_tfidf_transformer_mod= joblib.load(xlink + '/models/hiin_cs_tfidftransformer_v1.sav')
    log("Models , loaded ")
    return xloaded_model_mod, xloaded_cvec_mod, xloaded_tfidf_transformer_mod

def Predict_Codeswitching(model_mod, cvec_mod, tfidf_mod, input_entries):
    Codeswitching_result = []
    codeswitch_labels = {'0': 'no-codeswitch', '1': 'codeswitch'}

    for entry in input_entries:
        new_x = [entry.strip()]
        x_val_vec_mod = cvec_mod.transform(new_x)
        X_val_tfidf_mod = tfidf_mod.transform(x_val_vec_mod)
        result_mod = model_mod.predict(X_val_tfidf_mod)
        Codeswitching_result.append(codeswitch_labels[str(result_mod[0])])

    log("Codeswitching >> " + "#" *22)
    return Codeswitching_result


def check_scrp_swuch(xsnts):
    dontshow = ['NUMBER', 'DOLLAR', 'CIRCUMFLEX', 'AMPERSAND', 
    'ASTERISK', 'LOW', 'PLUS', 'EQUALS',
    'COLON', 'QUOTATION', 'LESS-THAN',
    'GREATER-THAN', 'FULL', 'LEFT-TO-RIGHT','PERCENT',
    'COMMERCIAL','EXCLAMATION','DIGIT','SPACE','HYPHEN-MINUS',
    'QUESTION', 'LEFT','APOSTROPHE','SEMICOLON', 'RIGHT', 
    'REVERSE']
    res  = [ud.name(c) for c in xsnts]
    unklangintxt = []
    for i in res:
        cc = i.split(' ')[0]
        if cc not in unklangintxt:
            if cc not in dontshow:
                unklangintxt.append(cc)

    return unklangintxt



def normlize_num(xx, list_allx):
    normalized = (xx-min(list_allx))/(max(list_allx)-min(list_allx))
    return normalized


def wiki_count(text):
    S = requests.Session()
    URL = "https://en.wikipedia.org/w/api.php"
    PARAMS = {
        "action": "query",
        "format": "json",
        "list": "search",
        "srsearch": text
    }

    r = S.get(url=URL, params=PARAMS)
    data = r.json()

    return data['query']['searchinfo']["totalhits"]


def is_code_switch_to_eng_lpzg(xsnt):
  xsnt_list_o = word_tokenize(xsnt)
  wl = []
  f_res = ""
  keep_words = []
  for w in xsnt_list_o:
    #print(w)
    if w.lower() in all_words and w not in all_span_words:
      wl.append("yes")
      keep_words.append(w)
    else:
      wl.append("no")

  if int(len(set(wl))) == 2 :
    f_res = "code_switch"

  if int(len(set(wl))) == 1 and "no" in wl:
    f_res = "no-code_switch"

  if int(len(set(wl))) == 1 and "yes" in wl:
    f_res = "English_Script"

  return f_res, keep_words



def classifyme(input_df, key, classifiers):
    clas_unq_name = ['salience', 'modality', 'rank', 'types_gic', 'code_switch', 'script', 'eng_spanish_cs']

    if len(set(classifiers)) != len(classifiers):
        log("Duplication in the classifier list, fix and try again")
        sys.exit()

    for c in classifiers:
        if c.lower() not in clas_unq_name:
            log(str(c) + " Not a correct classifer name. Please choose from these list. " + str(clas_unq_name))
            sys.exit()

    path_to_library = os.path.dirname(os.path.abspath(__file__))
    for i in classifiers:
        log("")
        if i.lower() == 'modality':
            log("Classify >> " +  (str(i)))
            modality_clf = load_FeatureBased_w_sw_Model(path_to_library)
            modality_result_df = predict_modality_fb(modality_clf, input_df[key])
            input_df['modality'] = modality_result_df
            log("")

        if i.lower() == 'rank':
            log("Classify >> " +  (str(i)))
            rank_model_mod, rank_cvec_mod, rank_tfidf_mod  = Load_Rank_Mmodels(path_to_library)
            rank_result_df = Predict_Rank(rank_model_mod, rank_cvec_mod, rank_tfidf_mod , input_df[key])
            input_df['rank'] = rank_result_df
            log("")

        if i.lower() == 'types_gic':
            log("Classify >> " +  (str(i)))
            type_model_mod, type_cvec_mod, type_tfidf_mod  = Load_Types_GIC_Mmodels(path_to_library)
            types_result_df = Predict_Types_GIC(type_model_mod, type_cvec_mod, type_tfidf_mod , input_df[key])
            input_df['types'] = types_result_df
            log("")

        if i.lower() == 'code_switch':
            log("Classify >> " +  (str(i)))
            cs_model_mod, cs_cvec_mod, cs_tfidf_mod  = Load_Codeswitching_Mmodels(path_to_library)
            cs_result_df = Predict_Codeswitching(cs_model_mod, cs_cvec_mod, cs_tfidf_mod , input_df[key])
            input_df['code_switch'] = cs_result_df
            log("")

        if i.lower() == 'script':
            log("Classify >> " +  str(i) )
            scc_result_df = []
            for x in input_df[key]:
                resc = check_scrp_swuch(x.strip())
                scc_result_df.append(str(resc))

            input_df['script'] = scc_result_df
            log("")

        if i.lower() == 'eng_spanish_cs':
            log("Classify >> " +  str(i) )
            log("")
            keep_result_labels =[]
            keep_result_words =[]
            eng_words = loadUnqList(path_to_library + '/models/en_from_nltk_unq_lower_rmv.txt')
            #eng_words = loadUnqList('models/eng_news_2020words.txt')
            #eng_words = loadUnqList('data/en_from_nltk_unq_lower_rmv.txt')

            for i in eng_words:
              #cc = i.split("\t")
              #all_words.append(cc[1].lower())
              all_words.append(i.lower())

            print(len(all_words))
            
            ########## load our tool
            span_words = loadUnqList(path_to_library + '/models/es_from_nltk_unq_lower.txt')
            #span_words = loadUnqList('models/spa_news_2020words.txt')
            #span_words = loadUnqList('data/es_from_nltk_unq_lower.txt')

            for i in span_words:
              #cc = i.split("\t")
              #all_span_words.append(cc[1].lower())
              all_span_words.append(i.lower()) 
             
            print(len(all_span_words))
            print("")
            for d in string.punctuation :
              all_span_words.append(d)
            
            keep_words = ""
            for i in input_df[key]:
              #f_res, keep_words = is_code_switch_to_eng_treebank(i.strip().lower())
              f_res, keep_words = is_code_switch_to_eng_lpzg(i.strip().lower())
              if keep_words == []:
                print(f_res, i)
                keep_result_labels.append(f_res)
                keep_result_words.append("x")
                
            
              if keep_words != []:
                print(f_res, keep_words, i)
                keep_result_labels.append(f_res)
                keep_result_words.append(keep_words)

        input_df['es-codeswitch'] = keep_result_labels
        input_df['en-words'] = keep_result_words
    
    return input_df


