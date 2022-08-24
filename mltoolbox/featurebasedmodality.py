
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.utils import shuffle
import pandas as pd
import unicodedata as ud
import matplotlib.pyplot as plt
from mltoolbox import log

training_Entries = []
test_Entries =[]


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


def save_model_f(xmodel, xunq_name, xpath, xfoldename):
    try:
        mod_name = xfoldename + "/" +  xunq_name + "_model.sav"
        joblib.dump(xmodel, mod_name)
        log("Model saved")
    except:
        print("Error")
        
	
def Create_feature_based_modality(xAlg_used,
                    xtrain_locales, 
                    xtest_locale, 
                    xsplit_test_percentage,
                    xTraining_file, 
                    xfull_names,
                    xshort_names,
                    xoutput_path, xuniq_model_name, xsavemodel):
    
    training_Entries = []
    test_Entries =[]
    
    locales_in_training_file = xtrain_locales
    test_locale = xtest_locale 
    cut_off_percentage = xsplit_test_percentage
    algor_used = xAlg_used
    #load file
    names = ['entry', 'label', 'locales']
    df_all_dataset = shuffle(shuffle(pd.read_csv(xTraining_file,  delimiter='\t', names=names)))    
    #df_all_dataset.head(10)
    print("Dataset loaded")
    
    for xentry, xlabel, xlocal in zip(df_all_dataset['entry'], df_all_dataset['label'], df_all_dataset['locales']):    
        if xlocal in locales_in_training_file:
            training_Entries.append(tuple([xentry, xlabel,  xlocal]))
         
        if xlocal in test_locale:
            test_Entries.append(tuple([xentry,xlabel,  xlocal]))#


    # cut test percentage

    cutoff = int(cut_off_percentage * len(test_Entries))
    test_Entries = test_Entries[cutoff:]
        
    log ("Locales in training dataset: " +  str(locales_in_training_file))
    log ("All Entries: " +  str(len(training_Entries)))
    log("")
    #print ("All Entries: ", len(test_Entries))
    cutoff_test = int(cut_off_percentage * len(test_Entries))
    log("Test-Dataset Avaliable: " + str(test_locale)  + "\t" + str(len(test_Entries)) + " Entries") 
    test_Entries = test_Entries[:cutoff_test]
    log("Test-Dataset Used:" + str(test_locale) + "\t" + str(cut_off_percentage) + "% >> "  + "\t" + str(len(test_Entries)) + " Entries") 
    log("")
    # model
    clf = Pipeline([('vectorizer', DictVectorizer(sparse=False)),
	            ('classifier', algor_used)     ])
    
    X, y = [], []
    X, y = transform_to_dataset(training_Entries)
    #clf.fit(X[:2000], y[:2000])
    clf = clf.fit(X, y)
    log("size" +  str(len(X))  + ", " + str(len(y)))
     
    X_test, y_test = transform_to_dataset(test_Entries)
    predicted = clf.predict(X_test)
    
    # Print the precision and recall, among other metrics
    class_rep = classification_report(y_test, predicted, digits=3)
    
    log(class_rep)
    # accuracy
    Average_accuracy_on_test = clf.score(X_test, y_test)
    log ("Accuracy:" + str(Average_accuracy_on_test))

    ## print and plot
    #y_true = y_test
    #y_pred = predicted 

    if xsavemodel.lower() == "yes": 
        ffolder = output_path + xuniq_model_name + "_model"
        isdir = os.path.isdir(ffolder) 
        #print(isdir) 
        if isdir == False:
            os.mkdir(ffolder) 
            save_model_f(clf, xuniq_model_name, output_path, ffolder)
        
        else:
            save_model_f(clf, xuniq_model_name, output_path, ffolder)


    




