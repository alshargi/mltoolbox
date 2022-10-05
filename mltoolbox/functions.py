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
import csv
from deep_translator import GoogleTranslator,MyMemoryTranslator 


def count_now(xnum):  
    for i in range(xnum):
        print("woow", i)



def load_txt_file(p):
    klist = []
    with open(p) as fword:
        klist = fword.read().splitlines()
    return klist



    
def load_txt_file_ISO(p):
    all_data = []
    with open(p, encoding = "ISO-8859-1") as fword:
        all_data = fword.read().splitlines()
    return all_data


def load_txt_file_utf8(p):
    all_data = []
    with open(p, encoding="utf-8") as fword:
        all_data = fword.read().splitlines()
    return all_data



# load our tool
def loadUnqList(p):
    klist = []
    with open(p) as fword:
        klist = fword.read().splitlines()
    return klist




def save_file(xlist, xxpath):
    file1 = open(xxpath,"w")
    for i in xlist:
        file1.writelines("{}\n".format(i))
    file1.close()
    

def create_new_auterance_GoogleTranslator(fromx, toz, tx):
    #auto, en, ar
    translated = GoogleTranslator(source=fromx, target=toz).translate(tx)
    return translated

    
def create_new_auterance_MyMemoryTranslator(fromx, toz, tx):
    #auto, en, ar
    translated = MyMemoryTranslator(source=fromx, target=toz).translate(tx)
    return translated 
    

def get_predict(strings, model_mod, cvec_mod, tfidf_mod, xl_labels_mod): 
    count_id = 0
    keepAll = []
    all_mod = []
    for line in strings:
        count_id += 1
        new_x = [line.strip()]
        
        x_val_vec_mod = cvec_mod.transform(new_x)
        X_val_tfidf_mod = tfidf_mod.transform(x_val_vec_mod)
        result_mod = model_mod.predict(X_val_tfidf_mod)
        
        log(str(count_id ) + '\t'  + xl_labels_mod[result_mod[0]] +  '\t' + str(new_x[0]))
        keepAll.append("{:03d}\t{}\t{}".format(count_id,  xl_labels_mod[result_mod[0]], new_x[0]))
        all_mod.append(xl_labels_mod[result_mod[0]])
    
    log("#" *30) 
    log("# Entries: " + str(len(strings)))
    keepAll.append("{}\t{}".format("# Entries: ", len(strings)))
    log("") 

    log("Modality >> ")
    keepAll.append("Modality >> ")
    keep_rep = CountFrequency_labeles(all_mod,len(strings) )
    for ix in keep_rep:
        keepAll.append(ix)
        
    keepAll.append("")     
    log("#" *30)   
    
    return keepAll



#--------------------
def Modality_labels_details():
    s_name= ['sw', 'w']
    l_name = ['SpokenAndWritten', 'WrittenOnly']
    return s_name, l_name 
#--------------------
def CodeSwitch_labels_details():
      s_name = ['no', 'yes'] 
      l_name = ['no-cs', 'yes-cs']
      return s_name, l_name 
  
#--------------------  
def Types_labels_details():   
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

#--------------------  
def Ranks_labels_details():
    s_name = [1, 2, 3] 
    l_name = ['Top', 'Medium','Low']
    return s_name, l_name
 
 

# load from github
def load_model(xlinkmod):
    myfile = requests.get(xlinkmod)
    open('./modality_models.zip', 'wb').write(myfile.content)
    with ZipFile('./modality_models.zip', 'r') as zipObj:
       zipObj.extractall()

 
    xloaded_model_mod = joblib.load('./modality_model.sav')
    xloaded_cvec_mod = joblib.load('./modality_countvectorizer.sav')
    xloaded_tfidf_transformer_mod= joblib.load('./modality_fidftransformer.sav')
           
    if os.path.exists("./modality_models.zip"):
        os.remove("./modality_models.zip")
    if os.path.exists("./modality_model.sav"):
        os.remove("./modality_model.sav")      
    if os.path.exists("./modality_fidftransformer.sav"):
        os.remove("./modality_fidftransformer.sav")       
    if os.path.exists("./modality_countvectorizer.sav"):
        os.remove("./modality_countvectorizer.sav")
     
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
        
     
    
#def load_modality_Ngram4MaxLsvc(): 
#    model_mod = joblib.load("mltoolbox/modality_model.sav")
#    return model_mod

#def load_countvectorizer_Ngram4MaxLsvc(): 
#    countvectorizer_mod = joblib.load("modality_countvectorizer.sav")
#    return countvectorizer_mod


#def load_fidftransformer_Ngram4MaxLsvc(): 
#    fidftransformer_mod = joblib.load("mltoolbox/modality_fidftransformer.sav")
#    return fidftransformer_mod



def load_modality_model(): 
    models_link = "https://github.com/alshargi/mltoolbox/raw/main/mltoolbox/modality_sw_w.zip"
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

import sys
def classify_now(input_df, key, classifiers):
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
        if i.lower() == 'eng_spanish_cs':
            log("Classify >> " +  str(i) )
            log("")
            keep_result_labels =[]
            keep_result_words =[]
            eng_words = loadUnqList(path_to_library + '/models/en_from_nltk_unq_lower_rmv.txt')
            for i in eng_words:
              all_words.append(i.lower())

            print(len(all_words))
            
            ########## load our tool
            span_words = loadUnqList(path_to_library + '/models/es_from_nltk_unq_lower.txt')
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



    
    
