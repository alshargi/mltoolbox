from datetime import datetime
import joblib
import json
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder    



def count_now(xnum):  
    for i in range(xnum):
        print("woow", i)



def save_file(xlist, xxpath):
    file1 = open(xxpath,"w")
    for i in xlist:
        file1.writelines("{}\n".format(i))
    file1.close()

    
def load_modality_Ngram4MaxLsvc_model(): 
    model_mod = joblib.load("mltoolbox/modality_model.sav")
    return model_mod


def load_countvectorizer_Ngram4MaxLsvc_model(): 
    countvectorizer_mod = joblib.load("mltoolbox/modality_countvectorizer.sav")
    return countvectorizer_mod


def load_fidftransformer_Ngram4MaxLsvc_model(): 
    fidftransformer_mod = joblib.load("mltoolbox/modality_fidftransformer.sav")
    return fidftransformer_mod


    
def read_json_originalText(fx):
    keepall = []
    f = open(fx,)
    data = json.load(f)
    for i in data:
        keepall.append(i['originalText'])
    f.close()
    return keepall


