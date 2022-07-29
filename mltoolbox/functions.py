from datetime import datetime
import joblib
import json
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder    
from colorama import Fore
from colorama import Style
from datetime import datetime


def count_now(xnum):  
    for i in range(xnum):
        print("woow", i)



def save_file(xlist, xxpath):
    file1 = open(xxpath,"w")
    for i in xlist:
        file1.writelines("{}\n".format(i))
    file1.close()
    
    
def log(string):
    now = str(datetime.now())
    print(Fore.BLUE + now + ' ' + Style.RESET_ALL + string)


def CountFrequency_labeles(my_list, inputfile_siz): 
    keep_result = []
    freq = {}
    for item in my_list:
        if (item in freq):
            freq[item] += 1
        else:
            freq[item] = 1
 
    for key, value in freq.items():
        perx = value / inputfile_siz * 100
        log("{}%\t{:03d}\t\t{}".format(round(perx, 3), value,  key))
        keep_result.append("{}%\t{:03d}\t\t{}".format(round(perx, 3), value, key ))
    return keep_result
        
     
    
def load_modality_Ngram4MaxLsvc(): 
    model_mod = joblib.load("mltoolbox/modality_model.sav")
    return model_mod

def load_countvectorizer_Ngram4MaxLsvc(): 
    countvectorizer_mod = joblib.load("modality_countvectorizer.sav")
    return countvectorizer_mod


def load_fidftransformer_Ngram4MaxLsvc(): 
    fidftransformer_mod = joblib.load("mltoolbox/modality_fidftransformer.sav")
    return fidftransformer_mod


def load_modelslenks(): 
    modelx = "https://github.com/alshargi/mltoolbox/blob/6f89db2a82a132353d7fc9f2d6f1df9297477c9a/mltoolbox/modality_model.sav"
    fidftransformerxx = "https://github.com/alshargi/mltoolbox/blob/6f89db2a82a132353d7fc9f2d6f1df9297477c9a/mltoolbox/modality_fidftransformer.sav"
    countvectorizerx = "https://github.com/alshargi/mltoolbox/blob/6f89db2a82a132353d7fc9f2d6f1df9297477c9a/mltoolbox/modality_countvectorizer.sav"
    return modelx, fidftransformerxx, countvectorizerx


def read_csv_file_comma(fnme):
    keepall_row = []
    with open(fnme) as from_LC:
        csvReader_lc = csv.reader(from_LC, delimiter=',')
        for row in csvReader_lc:
            keepall_row.append("{}".format(row[0]))
    return keepall_row

def read_csv_file_tab(fnme):
    keepall_row = []
    with open(fnme) as from_LC:
        csvReader_lc = csv.reader(from_LC, delimiter='\t')
        for row in csvReader_lc:
            keepall_row.append("{}".format(row[0]))
    return keepall_row


def read_json_originalText(fx):
    keepall = []
    f = open(fx,)
    data = json.load(f)
    for i in data:
        keepall.append(i['originalText'])
    f.close()
    return keepall


