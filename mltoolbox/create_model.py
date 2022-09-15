# -*- coding: utf-8 -*-

from datetime import datetime
import numpy as np
from colorama import Fore
from colorama import Style

from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
    
from sklearn.utils import shuffle
from pandas import np

from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV

from sklearn.svm import LinearSVC                              
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import os
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import joblib


 
print_to_file = []



def log(string):
    now = str(datetime.now())
    print(Fore.BLUE + now + ' ' + Style.RESET_ALL + string)


    
def balance_training_data_asynthatec(X_trainx, y_trainx, X_testx, y_testx, ngram_range, analyzer_type):
    smote = SMOTE(random_state = 10)
    #smote = SMOTE("minority")
    df_train_only = pd.DataFrame({'entry': X_trainx, 'label': y_trainx})  
    count_vect = CountVectorizer(ngram_range= ngram_range, 
                                 lowercase=False, analyzer=analyzer_type)      

    #train transform
    X_train_counts = count_vect.fit_transform(X_trainx)
    tf_transformer = TfidfTransformer().fit(X_train_counts)
    X_train_transformed = tf_transformer.transform(X_train_counts)
        
    labels = LabelEncoder()     
    y_train_labels_fit = labels.fit(y_trainx)
    y_train_lables_trf = labels.transform(y_trainx)
        
    # test transform
    X_test_counts = count_vect.transform(X_testx)
    X_test_transformed = tf_transformer.transform(X_test_counts)

    y_test_transformed = labels.transform(y_testx)
    #####
    log("Running , Please wait ...") 
    
    X_train_transformed_syn, y_train_lables_trf_syn = smote.fit_resample(X_train_transformed, y_train_lables_trf)
    return X_train_transformed_syn, y_train_lables_trf_syn, X_test_transformed, y_test_transformed
    


def balance_training_data_min(X_all_entry, y_all_entry,  ngram_range, analyzer_type, split_test_percentagex):
    df_train_only = pd.DataFrame({'entry': X_all_entry, 'label': y_all_entry})  
    count_vect = CountVectorizer(ngram_range= ngram_range, 
                                 lowercase=False, analyzer=analyzer_type)  

    
    min_class_siz = df_train_only.groupby('label').count()
    fmin_c =  min_class_siz.min()[0]
    log("Minimum class # = " + str(fmin_c))
    log("Minimum class: " + str(fmin_c))
    split_by = int(min_class_siz.min()[0])
    print(min_class_siz)
    uniq_labels = df_train_only.label.unique()
    print(uniq_labels)
    entry_balance_data = []
    label_balance_data = []
    cnow = 0
    for ib in uniq_labels: 
        print("label - ", ib, cnow )
        cnow = 0
        for x, z in zip(df_train_only['entry'], df_train_only['label']):
            if ib == z:       
                if cnow < split_by:
                    #print(x + "\t" +z)
                    entry_balance_data.append(x)
                    label_balance_data.append(z)
                    cnow += 1 
                   
                    
    balance_list = pd.DataFrame(
      {'entry': entry_balance_data,
       'label': label_balance_data
      })
    balance_alldata = shuffle(shuffle(balance_list)) # 3 x
    balance_list = []  # remove for memory
    
    print(balance_alldata.head())
    df_all_dataset = balance_alldata
    X_train, X_test, y_train, y_test = train_test_split( df_all_dataset['entry'],  df_all_dataset['label'], random_state=0, test_size= split_test_percentagex)
  

    
    #train transform
    X_train_counts = count_vect.fit_transform(X_train)
    tf_transformer = TfidfTransformer().fit(X_train_counts)
    X_train_transformed = tf_transformer.transform(X_train_counts)
        
    labels = LabelEncoder()     
    y_train_labels_fit = labels.fit(y_train)
    y_train_lables_trf = labels.transform(y_train)
  
    # test transform
    X_test_counts = count_vect.transform(X_test)
    X_test_transformed = tf_transformer.transform(X_test_counts)

    y_test_transformed = labels.transform(y_test)
    
    
    return X_train_transformed, y_train_lables_trf, X_test_transformed, y_test_transformed
    

def balance_training_data_max(X_trainx, y_trainx, X_testx, y_testx, ngram_range, analyzer_type):
    count_vect = CountVectorizer(ngram_range= ngram_range, 
                                 lowercase=False, analyzer=analyzer_type)  

    
    #train transform
    print_to_file = []
    print_to_file.append("#### All dataset will be used regardless balanced or not")


    X_train_counts = count_vect.fit_transform(X_trainx)
    tf_transformer = TfidfTransformer().fit(X_train_counts)
    X_train_transformed = tf_transformer.transform(X_train_counts)
        
    labels = LabelEncoder()     
    y_train_labels_fit = labels.fit(y_trainx)
    y_train_lables_trf = labels.transform(y_trainx)
  
    # test transform
    X_test_counts = count_vect.transform(X_testx)
    X_test_transformed = tf_transformer.transform(X_test_counts)
    y_test_transformed = labels.transform(y_testx)
    
    
    return X_train_transformed, y_train_lables_trf, X_test_transformed, y_test_transformed
    



def create_model(dfx, alg_name, split_test_percentage, balance_type, ngram_range, analyzer_type, output_path, save_model):
    if balance_type.lower() == 'synthatic':
        X_train, X_test, y_train, y_test = train_test_split( dfx['entry'], dfx['label'],
                                                         random_state=0, test_size= split_test_percentage) 
    
        print("We will use synthatec SMOTE")
        X_train_trans_, y_train_trf_, X_test_transformedx, y_test_transformedx = balance_training_data_asynthatec(X_train, y_train, X_test, y_test,
                                                                    ngram_range, analyzer_type)
        # create the Model with syn data
        clf = alg_name.fit(X_train_trans_,y_train_trf_)# 
        calibrated_svc = CalibratedClassifierCV(base_estimator=clf,cv="prefit")    
        calibrated_svc.fit(X_train_trans_,y_train_trf_)
        predicted = calibrated_svc.predict(X_test_transformedx)
        Average_accuracy_on_test= np.mean(predicted == y_test_transformedx)
        log("Accuracy : " + str(Average_accuracy_on_test)  + '%')
    
    if balance_type.lower() == 'min':
        print("We will use MIN")
        X_train_trans_, y_train_trf_, X_test_transformedx, y_test_transformedx = balance_training_data_min(dfx['entry'], dfx['label'],
                                                               ngram_range, analyzer_type, split_test_percentage)
      
        print(df_all_dataset.head())  
        log("Running , Please wait ...") 
        # create the Model with MIN data
        clf = alg_name.fit(X_train_trans_,y_train_trf_)# 
        calibrated_svc = CalibratedClassifierCV(base_estimator=clf,cv="prefit")    
        calibrated_svc.fit(X_train_trans_,y_train_trf_)
        predicted = calibrated_svc.predict(X_test_transformedx)
        Average_accuracy_on_test= np.mean(predicted == y_test_transformedx)
        log("Accuracy : " + str(Average_accuracy_on_test)  + '%') 
    
    if balance_type.lower() == 'max':
        log("We will use MAX")
        X_train, X_test, y_train, y_test = train_test_split( dfx['entry'], dfx['label'],
                                                         random_state=0, test_size= split_test_percentage) 
        #X_train_trans_, y_train_trf_, X_test_transformedx, y_test_transformedx = balance_training_data_max(X_train, y_train, X_test, y_test,
        #                                                       ngram_range, analyzer_type)
        
        count_vect = CountVectorizer(ngram_range= ngram_range, 
                                     lowercase=False, analyzer=analyzer_type)  

        
        #train transform
        print_to_file = []
        print_to_file.append("#### All dataset will be used regardless balanced or not")


        X_train_counts = count_vect.fit_transform(X_train)
        tf_transformer = TfidfTransformer().fit(X_train_counts)
        X_train_transformed = tf_transformer.transform(X_train_counts)
            
        labels = LabelEncoder()     
        y_train_labels_fit = labels.fit(y_train)
        y_train_lables_trf = labels.transform(y_train)
      
        # test transform
        X_test_counts = count_vect.transform(X_test)
        X_test_transformed = tf_transformer.transform(X_test_counts)
        y_test_transformed = labels.transform(y_test)
        
        
        labels_list = dfx.groupby('label').count()
        uniq_labels = dfx.label.unique()
         
        print(df_all_dataset.head())  
        log("Running , Please wait ...") 
        # create the Model with MIN data
        clf = alg_name.fit(X_train_transformed,y_train_lables_trf)# 
        calibrated_svc = CalibratedClassifierCV(base_estimator=clf,cv="prefit")    
        calibrated_svc.fit(X_train_transformed,y_train_lables_trf)
        predicted = calibrated_svc.predict(X_test_transformed)
        Average_accuracy_on_test= np.mean(predicted == y_test_transformed)
 
    
        # Print the confusion matrix 
        y_true = y_test_transformed 
        y_pred = predicted 
             
        cf_matrix_3x3 = confusion_matrix(y_true, y_pred)
        log("Accuracy : " + str(Average_accuracy_on_test) + '%')
        # Print the precision and recall, among other metrics
        
        uniq_labels_str = [] 
        for i in uniq_labels:
            uniq_labels_str.append(str(i))

        
        clas_rep = classification_report(y_true, y_pred, digits=3, target_names= uniq_labels_str)  
        ########### sensitivity','specificity'
        #Sensitivity, also known as the true positive rate (TPR), is the same as recall. Hence, it measures the proportion of positive class that is correctly predicted as positive.
        #Specificity is similar to sensitivity but focused on negative class. It measures the proportion of negative class that is correctly predicted as negative.
        
        res = []
        for l in range(0, len(uniq_labels_str)):
             prec,recall,_,_ = precision_recall_fscore_support(np.array(y_true)==l,
                                                           np.array(y_pred)==l,
                                                           pos_label=True,average=None)
             
             res.append([l, recall[0], recall[1]])
        

        print(uniq_labels)
        print_to_file.append("#### All entries: " + str(len(dfx['entry'])) )   
        print_to_file.append("#### Training entries: " + str(len(X_train)) )
        print_to_file.append("#### Testing entries: " + str(len(X_test)) )
        print_to_file.append("")
        print_to_file.append("#### Labeles: " + str(labels_list))
        print_to_file.append("")
        print_to_file.append("#### Uniq labels: " + str(uniq_labels))
        print_to_file.append("")
        print_to_file.append("#### Accuracy: " + str(Average_accuracy_on_test) + '%')
        print_to_file.append("")
        print_to_file.append("c#### lassification_report : " + str(clas_rep) )
        print_to_file.append("")
        print_to_file.append("#### confusion Matrix : " + str(cf_matrix_3x3) )
        print_to_file.append("")
        print_to_file.append("#### sensitivity and specificity")
        print_to_file.append(pd.DataFrame(res,columns = ['class','sensitivity','specificity']))
        print_to_file.append("################")   
        if save_model.lower() == "yes": 
            return print_to_file, calibrated_svc,  count_vect, tf_transformer
        
        if save_model.lower() == "no": 
            return print_to_file
        
