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
<<<<<<< Updated upstream
    model_mod = joblib.load("models/modality_ngram_4_max_lsvc_model.sav")
    vec_mod = joblib.load("models/modality_ngram_4_max_lsvc_countvectorizer.sav")
    tfidf_transformer_mod= joblib.load("models/modality_ngram_4_max_lsvc_tfidftransformer.sav")
=======
    model_mod = joblib.load("models/modality_ngram_4_max_lsvc/modality.sav")
    vec_mod = joblib.load("models/modality_ngram_4_max_lsvc/countvectorizer.sav")
    tfidf_transformer_mod= joblib.load("models/modality_ngram_4_max_lsvc/tfidftransformer.sav")
>>>>>>> Stashed changes
    print("Models , loaded ")
  

def read_json_originalText(fx):
    keepall = []
    f = open(fx,)
    data = json.load(f)
    for i in data:
        keepall.append(i['originalText'])
    f.close()
    return keepall


