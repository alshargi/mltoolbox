from datetime import datetime
import joblib
import json
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder    
from colorama import Fore
from colorama import Style
import requests
from zipfile import ZipFile
import os


def count_now(xnum):  
    for i in range(xnum):
        print("woow", i)



def save_file(xlist, xxpath):
    file1 = open(xxpath,"w")
    for i in xlist:
        file1.writelines("{}\n".format(i))
    file1.close()
    
  
def labels_details(clas_name):
    s_name = []
    l_name = [] 
    if clas_name == 'modality':
        s_name = ['s', 'sw','w'] #entry
        l_name = ['SpokenOnly', 'SpokenAndWritten','WrittenOnly']
    #--------------------
    # hindi CS
    if clas_name == 'cs':
        s_name = ['no', 'yes'] 
        l_name = ['no-cs', 'yes-cs']
    #--------------------
    if clas_name == 'rank':
        s_name = [1, 2, 3] 
        l_name = ['Top', 'Medium','Low']
    #--------------------
    if clas_name == 'type':
        s_name = ['ap', 'cn', 'cnt', 'cty', 'cu', 'dr', 'dt', 'dv', 'ed', 'fnm',
           'hp', 'lnm', 'nm', 'np', 'pg', 'pn', 'prt', 'ps', 'rm', 'rt', 'sp',
           'st', 'stn', 'stt', 'sv', 'tm', 'wg', 'yp']
        l_name =  ['AlphaNumeric-GTD' , 'Cancel-Intent', 'Country-Catalog',
                        'City-Catalog', 'Currency-GTD', 'Duration-GTD',
                        'Date-GTD', 'DateInterval-GTD', 'EmailAddress-GTD', 
                        'FirstName-Catalog', 'Help-Intent', 'LastName-Catalog' ,
                        'Number-GTD', 'Nope-Intent', 'Percentage-GTD', 'PhoneNumber-GTD', 
                        'Airport-Catalog', 'Pause-Intent', 'Resume-Intent', 'Repeat-Intent', 
                        'Speed-GTD', 'Stop-Intent' , 'StreetName-Catalog', 'State-Catalog' ,
                        'StartOver-Intent', 'Time-GTD', 'Weight-GTD','Yeppers-Intent']

    return s_name, l_name
    
 

# load from github
def load_model(xlinkmod):
    myfile = requests.get(xlinkmod)
    open('./modality_models.zip', 'wb').write(myfile.content)
    with ZipFile('./modality_models.zip', 'r') as zipObj:
       zipObj.extractall()
       
    if os.path.exists("./modality_models.zip"):
        os.remove("./modality_models.zip")
 
    xloaded_model_mod = joblib.load('./modality_model.sav')
    xloaded_cvec_mod = joblib.load('./modality_countvectorizer.sav')
    xloaded_tfidf_transformer_mod= joblib.load('./modality_fidftransformer.sav')
    print("Models , loaded ")
    print("")
    return xloaded_model_mod, xloaded_cvec_mod, xloaded_tfidf_transformer_mod




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



def load_modality_model(): 
    models_link = "https://github.com/alshargi/mltoolbox/raw/main/mltoolbox/models/modality_ngram_4_max_lsvc/modality.zip"
    return models_link


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


