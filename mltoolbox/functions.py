from datetime import datetime
import joblib
import json



def count_now(xnum):  
    for i in range(xnum):
        print("woow", i)



def save_file(xlist, xxpath):
    file1 = open(xxpath,"w")
    for i in xlist:
        file1.writelines("{}\n".format(i))
    file1.close()

    


def load_Model_modalityNgram4MaxLsvc():   
    model_mod = joblib.load("./modality_model.sav")
    #vec_mod = joblib.load("https://github.com/alshargi/mltoolbox/blob/5ab89f89bea1610e5cd14186fd09ede70752552b/mltoolbox/modality_countvectorizer.sav")
    #tfidf_transformer_mod= joblib.load("https://github.com/alshargi/mltoolbox/blob/5ab89f89bea1610e5cd14186fd09ede70752552b/mltoolbox/modality_fidftransformer.sav")
    print("Models , loaded ") 

    
def read_json_originalText(fx):
    keepall = []
    f = open(fx,)
    data = json.load(f)
    for i in data:
        keepall.append(i['originalText'])
    f.close()
    return keepall


