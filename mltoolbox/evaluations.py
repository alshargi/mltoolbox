import argparse
import numpy as np
from colorama import Fore
from colorama import Style
from datetime import datetime
import pandas as pd
import os
from sklearn.utils import shuffle
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
import matplotlib.pyplot as plt
import sys
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support
from collections import Counter
from imblearn.over_sampling import SMOTE
import joblib


def save_file(xlist, xxpath):
    file1 = open(xxpath,"w")
    for i in xlist:
        file1.writelines("{}\n".format(i))
    file1.close()
    
    
def log(string):
    now = str(datetime.now())
    print(Fore.BLUE + now + ' ' + Style.RESET_ALL + string)


def save_model(xcalibrated_svc,  xcount_vect, xtf_transformer, xunq_name, xpath, xfoldename):
    try:
        mod_name = xfoldename + "/" +  xunq_name + "_model.sav"
        vec_modname = xfoldename + "/" +  xunq_name + "_countvectorizer.sav" 
        tfidf_modname = xfoldename + "/" +  xunq_name + "_tfidftransformer.sav"
            
            #filename = 'Type_model_gci_v1.sav'
        joblib.dump(xcalibrated_svc, mod_name)
            
        #filename = 'Type_countvectorizer_gci_v1.sav'
        joblib.dump(xcount_vect, vec_modname)
        
        #filename = 'Type_tfidftransformer_gci_v1.sav'
        joblib.dump(xtf_transformer, tfidf_modname)
        log("Model saved")
    except:
        print("Error")
   
     
def Create_model_Synthatic(Model_used, Training_file, delimiter,ngram_range, analyzer_type,  split_test_percentage, s_labels_mod, full_names,output_path,uniq_model_name, savemodel ):    
    balance_type = "Synthatic"
    print_to_file = []
    log("Balance: " + str(balance_type))
    log("We will use SMOTE to balance the training file ")
    print_to_file.append("Balance: " + str(balance_type))
    print_to_file.append("We will use SMOTE to balance the training file ")
    print_to_file.append("") 
    
    names = ['entry', 'label']
    df_all_dataset = shuffle(pd.read_csv(Training_file,  delimiter=delimiter, names=names))
    X_train, X_test, y_train, y_test = train_test_split( df_all_dataset['entry'],  df_all_dataset['label'], random_state=0, test_size= split_test_percentage) 
    log("Balance: " + str(balance_type))
    log("We will use synthatec SMOTE")
    
    
    
    smote = SMOTE(random_state = 10)
    #smote = SMOTE("minority")

    df_test_only = pd.DataFrame({'entry': X_test, 'label': y_test})
    # Train only Dataset
    df_train_only = pd.DataFrame({'entry': X_train, 'label': y_train})  
    
    count_vect = CountVectorizer(ngram_range= ngram_range,  lowercase=False, analyzer=analyzer_type)      

    #train transform
    X_train_counts = count_vect.fit_transform(X_train)
    tf_transformer = TfidfTransformer().fit(X_train_counts)
    X_train_transformed = tf_transformer.transform(X_train_counts)
        
    # test transform
    X_test_counts = count_vect.transform(X_test)
    X_test_transformed = tf_transformer.transform(X_test_counts)
        
    labels = LabelEncoder()     
    y_train_labels_fit = labels.fit(y_train)
    y_train_lables_trf = labels.transform(y_train)
        
        
    #####
    print(df_all_dataset.head())  
    log("Running , Please wait ...") 
    
    # check mislabeled
    print_to_file.append("")
    log(" Check Mislabeled entry")
    print_to_file += check_mislabeled(df_all_dataset["label"], s_labels_mod )
 
    X_train_transformed_syn, y_train_lables_trf_syn = smote.fit_resample(X_train_transformed, y_train_lables_trf)
    # create the Model with syn data
    clf = Model_used.fit(X_train_transformed_syn,y_train_lables_trf_syn)# 
    calibrated_svc = CalibratedClassifierCV(base_estimator=clf,cv="prefit")    
    calibrated_svc.fit(X_train_transformed_syn,y_train_lables_trf_syn)
         
    predicted = calibrated_svc.predict(X_test_transformed)
    Average_accuracy_on_test= np.mean(predicted == labels.transform(y_test))

    

    # Print the confusion matrix
    y_true = labels.transform(y_test) 
    y_pred = predicted 
        
    cf_matrix_3x3 = confusion_matrix(y_true, y_pred)
        
    # Print the precision and recall, among other metrics
    clas_rep = classification_report(y_true, y_pred, digits=3, target_names= full_names)

    ########### sensitivity','specificity'
    #Sensitivity, also known as the true positive rate (TPR), is the same as recall. Hence, it measures the proportion of positive class that is correctly predicted as positive.
    #Specificity is similar to sensitivity but focused on negative class. It measures the proportion of negative class that is correctly predicted as negative.
       
    res = []
    for l in range(0, len(s_labels_mod)):
        prec,recall,_,_ = precision_recall_fscore_support(np.array(y_true)==l,
                                                          np.array(y_pred)==l,
                                                          pos_label=True,average=None)
        res.append([full_names[l],recall[0],recall[1]])
        
  # print info
    prnt_bfor = []   
    prntaftr = []
        
    #############
    #from collections import Counter
    counter_bfr = Counter(y_train_lables_trf) 
        #print(counter_bfr)

    log("")
    log("Training file before synthatic: ")        
    counter_bfr =   Counter(y_train_lables_trf)
        
    for k,v in counter_bfr.items():
        per = v / len(y_train_lables_trf) * 100
        log('Class=%s, n=%d (%.3f%%)' % (label_like_the_file[k], v, per))
        prnt_bfor.append('Class=%s, n=%d (%.3f%%)' % (label_like_the_file[k], v, per))
        
    print_to_file.append("Training file before synthatic: "  )
    log("----------------------")        
    log("") 
    print_to_file.append("")
    print_to_file += prnt_bfor
    print_to_file.append("")
    ########### 
    log("")
    log("Training file after synthatic: ")        
    counter_aftr =   Counter(y_train_lables_trf_syn)
    #print(counter_aftr)
    for k,v in counter_aftr.items():
        per = v / len(y_train_lables_trf_syn) * 100
        log('Class=%s, n=%d (%.3f%%)' % (s_labels_mod[k], v, per))
        prntaftr.append('Class=%s, n=%d (%.3f%%)' % (s_labels_mod[k], v, per))
        
    print_to_file.append("")
    print_to_file.append("Training file after synthatic" )
    log("----------------------")        
    log("") 
    print_to_file += prntaftr
    print_to_file.append("")

    ###################### plot data and confusion matrix  
    chart_path = output_path + "_" + uniq_model_name + "_" + "_ngram"  + "_" +  str(balance_type) +  "_charts.pdf"
       
    with PdfPages(chart_path) as export_pdf:
        fig = pyplot.figure(figsize=(8,6))                  
        plt.bar(counter_bfr.keys(), counter_bfr.values())
        plt.title('Training before synthatic')
        plt.grid(True)
        export_pdf.savefig()
        plt.close()
        plt.bar(counter_aftr.keys(), counter_aftr.values())
        plt.title('Training after synthatic')
        plt.grid(True)
        export_pdf.savefig()
        plt.close()
        
        make_confusion_matrix(cf_matrix_3x3, 
                                      categories=full_names,
                                      figsize=(8,6), 
                                      cbar=False,
                                      title="Confusion_matrix - with synthatic data")
        plt.grid(True)
        export_pdf.savefig()
        plt.close()
       
     
    print_to_file.append(" ")
    print_to_file.append("------------------------------- ")
    print_to_file += show_report(split_test_percentage,
                            len(df_test_only),
                            len(df_train_only),
                            len(df_all_dataset),
                            labels.classes_,
                            analyzer_type,
                            ngram_range,
                            Model_used ,
                            cf_matrix_3x3,
                            Average_accuracy_on_test, clas_rep,
                            pd.DataFrame(res,columns = ['class','sensitivity','specificity']))
              

    print_to_file.append("") 
     #################################################### 
     # check wrong predictions  
    log("wrong predicted list in the result file")
    print_to_file.append("")
    print_to_file += show_wrong_prred(X_test, labels.transform(y_test), predicted , full_names, label_like_the_file)
       
            
    save_file( print_to_file, output_path + "_" + uniq_model_name + "_"  + "_" +  str(balance_type) +  "_eval_report.txt" )                 
    log("Report (pdf charts) and text saved in this path " + str(output_path)  )
    log("--------------------------------")
    log("")
    
    if savemodel.lower() == "yes": 
        un_name =  uniq_model_name + "_ngram_" +  str(balance_type) 
        ffolder = output_path + un_name + "_model"
        isdir = os.path.isdir(ffolder) 
        #print(isdir) 
        if isdir == False:
            os.mkdir(ffolder) 
            save_model(calibrated_svc,  count_vect, tf_transformer, un_name, output_path, ffolder)
        else:
            save_model(calibrated_svc,  count_vect, tf_transformer, un_name, output_path, ffolder)

            
            
def Create_model_max(Model_used, Training_file, delimiter,ngram_range, analyzer_type,  split_test_percentage, s_labels_mod, full_names,output_path,uniq_model_name, savemodel ):    
    balance_type = "Max"
    rep_print = []
    names = ['entry', 'label']
    df_all_dataset = shuffle(pd.read_csv(Training_file,  delimiter=delimiter, names=names))
    X_train, X_test, y_train, y_test = train_test_split( df_all_dataset['entry'],  df_all_dataset['label'], random_state=0, test_size= split_test_percentage) 
    log("Balance: " + str(balance_type))
    log("We will use all the imbalance train file")
    
    # check check mislabeled
    rep_print.append("")
    log(" Check Mislabeled entry")
    rep_print += check_mislabeled(df_all_dataset["label"], s_labels_mod )
 
    # Test only Dataset
    df_test_only = pd.DataFrame({'entry': X_test, 'label': y_test})
    # Train only Dataset
    df_train_only = pd.DataFrame({'entry': X_train, 'label': y_train})  
    count_vect = CountVectorizer(ngram_range= ngram_range,  lowercase=False, analyzer=analyzer_type)      
    
    #train transform
    X_train_counts = count_vect.fit_transform(X_train)
    tf_transformer = TfidfTransformer().fit(X_train_counts)
    X_train_transformed = tf_transformer.transform(X_train_counts)
    
    # test transform
    X_test_counts = count_vect.transform(X_test)
    X_test_transformed = tf_transformer.transform(X_test_counts)
    
    labels = LabelEncoder()
    
    y_train_labels_fit = labels.fit(y_train)
    y_train_lables_trf = labels.transform(y_train)
    
    log(">> Training file - loaded" + str(len(df_all_dataset)) + ' Entries')
    log("")
    
    clf = Model_used.fit(X_train_transformed,y_train_lables_trf)#
         
    calibrated_svc = CalibratedClassifierCV(base_estimator=clf,cv="prefit") 
    calibrated_svc.fit(X_train_transformed,y_train_lables_trf)
          
    predicted = calibrated_svc.predict(X_test_transformed)
    Average_accuracy_on_test= np.mean(predicted == labels.transform(y_test))

    ## print and plot
    y_true = labels.transform(y_test) 
    y_pred = predicted 
         
    # Print the confusion matrix
    cf_matrix_3x3 = confusion_matrix(y_true, y_pred)
         
    # Print the precision and recall, among other metrics
    clas_rep = classification_report(y_true, y_pred, digits=3, target_names= full_names)


    ########### sensitivity','specificity'
    #Sensitivity, also known as the true positive rate (TPR), is the same as recall. Hence, it measures the proportion of positive class that is correctly predicted as positive.
    #Specificity is similar to sensitivity but focused on negative class. It measures the proportion of negative class that is correctly predicted as negative.
   
    res = []
    for l in range(0, len(s_labels_mod)):
        prec,recall,_,_ = precision_recall_fscore_support(np.array(y_true)==l,
                                                      np.array(y_pred)==l,
                                                      pos_label=True,average=None)
        res.append([full_names[l],recall[0],recall[1]])

    # print info
   
    rep_print += show_report(split_test_percentage,
                   len(df_test_only),
                   len(df_train_only),
                   len(df_all_dataset),
                   labels.classes_,
                   analyzer_type,
                   ngram_range,
                   Model_used ,
                   cf_matrix_3x3,
                   Average_accuracy_on_test, clas_rep, 
                   pd.DataFrame(res,columns = ['class','sensitivity','specificity']))




    ###################### plot data and confusion matrix  
    chart_path = output_path + "_" + uniq_model_name + "_"  + "_" +  str(balance_type) +  "_charts.pdf"
    chart_data(chart_path, df_all_dataset, df_train_only, df_test_only, full_names, cf_matrix_3x3)
    
    #from yellowbrick.classifier import class_prediction_error   
    #visualizer = class_prediction_error(calibrated_svc, 
    #                            X_train_transformed,y_train_lables_trf, 
    #                            classes=full_names)
    #visualizer.show()

    
   #################################################### 
    # check wrong predictions
    rep_print += show_wrong_prred(X_test, labels.transform(y_test), 
                     predicted , full_names, s_labels_mod)



    rep_print.append("")  
    save_file(rep_print, output_path + "_" + uniq_model_name + "_"  + "_" +  str(balance_type) +  "_eval_report.txt" )                 
    log("Report (pdf charts) and text saved in this path " + str(output_path)  )
    log("--------------------------------")
    log("")
    
    if savemodel.lower() == "yes": 
        un_name =  uniq_model_name + "_ngram_" +  str(balance_type) 
        ffolder = output_path + un_name + "_model"
        isdir = os.path.isdir(ffolder) 
        #print(isdir) 
        if isdir == False:
            os.mkdir(ffolder) 
            save_model(calibrated_svc,  count_vect, tf_transformer, un_name, output_path, ffolder)
        else:
            save_model(calibrated_svc,  count_vect, tf_transformer, un_name, output_path, ffolder)

   

def check_mislabeled(xdf_all_dataset, slabel):
    print_to_filex = []
    stat_miss = 1
    print_to_filex.append("------------------------------- ")
    mislabeled_yes =[]
    mislabeled_yes.append("###################### mislabeled Entries")
    for i in  xdf_all_dataset:
        if i not in slabel:
            log( "mislabeled >>" +  str(i))
            mislabeled_yes.append(i)
            stat_miss = 0
    if stat_miss == 0:
        log("The data include mislabeled, please check and fix. ")
        print_to_filex.append("###### The data include mislabeled, please check and fix. ")
        print_to_filex.append(mislabeled_yes)
        print_to_filex.append("-------------------------")
                      
    else:
        log("NO mislabeled, PASS")
        print_to_filex.append("NO mislabeled, PASS")
        print_to_filex.append("-------------------------")
    
    print_to_filex.append("")
    return print_to_filex

    
def show_report(xsplit_test_percentage,xsizetest, xsiztrain, xsizalldata, xlabelsclass,xanalyzer_type,xngram_range,xModel_used ,xcf_matrix_3x3,xAverage_accuracy_on_test, xclas_rep, pd_senrtv):
    print_to_file = []
    log("")
    log("##############")
    log("Classifier Name: " + str(xModel_used))
    log("ngram range: " + str(xngram_range))
    log("analyzer type: "+ str(xanalyzer_type))
    log("Dataset Labels: "+ str(xlabelsclass))
    log("All Dataset " + str(xsizalldata))
    log("Train Dataset " + str(xsiztrain) + " >> % " + str(1 - xsplit_test_percentage))
    log("Test  Dataset " + str(xsizetest)  + " >> % " + str( xsplit_test_percentage))
    log("-------------------------------------")
    log("")
    print_to_file.append("{}\t{}".format("Classifier Name: ", xModel_used))
    print_to_file.append("{}\t{}".format("Ngram range: ", xngram_range))
    print_to_file.append("{}\t{}".format("Analyzer type: ", xanalyzer_type))
    print_to_file.append("{}\t{}".format("Dataset Labels: ", xlabelsclass))
    print_to_file.append("{}\t{}".format("All Dataset ", xsizalldata))
    print_to_file.append("{}\t{}\t{}".format("Train Dataset ", xsiztrain, 1 - xsplit_test_percentage))
    print_to_file.append("{}\t{}\t{}".format("Test Dataset ", xsizetest, xsplit_test_percentage))
    print_to_file.append("-------------------------------------")
    log("")
    log('Average accuracy on test set = {} % '.format(xAverage_accuracy_on_test ))
    log("Confusion Matrix:")
    print("")
    print(xcf_matrix_3x3)
    log("")
    print_to_file.append("Confusion Matrix:")
    print_to_file.append(xcf_matrix_3x3)
    print_to_file.append("") 
    log("")
    log("classification report:")
    print(xclas_rep)
    log("")
    print_to_file.append("")
    print_to_file.append("classification report:")
    print_to_file.append(xclas_rep)
    print_to_file.append("")
    print_to_file.append("{}\t{}".format("Average accuracy on test set ", xAverage_accuracy_on_test))
    print_to_file.append("")
    
    
    log(" Sensitivity and specificity ")
    log("-----------------------------")
    print(pd_senrtv)
    print_to_file.append(" Sensitivity and specificity ")
    print_to_file.append("-----------------------------")
    print_to_file.append(pd_senrtv)
    return print_to_file

def chart_data(xhart_path, xdf_all_dataset, xdf_train_only, xdf_test_only, xfull_names, cf_matrix_3x3x):
    with PdfPages(xhart_path) as export_pdf:
        fig = plt.figure(figsize=(8,6))  



        xdf_all_dataset.groupby('label').count().plot.bar(ylim=0)
        plt.title('All Dataset')
        plt.grid(True)
        export_pdf.savefig()
        plt.close()
        ###    
        #fig = plt.figure(figsize=(8,6))
        xdf_train_only.groupby('label').count().plot.bar(ylim=0)
        plt.title('Train Dataset')
        plt.grid(True)
        export_pdf.savefig()
        plt.close()
        ###   
        #fig = plt.figure(figsize=(8,6))
        xdf_test_only.groupby('label').count().plot.bar(ylim=0)
        plt.title('Test Dataset')
        plt.grid(True)
        export_pdf.savefig()
        plt.close()
            
        make_confusion_matrix(cf_matrix_3x3x, 
                                  categories=xfull_names,
                                  figsize=(8,6), 
                                  cbar=False,
                                  title="confusion_matrix")
        plt.grid(True)
        export_pdf.savefig()
        plt.close()
        
def make_confusion_matrix(cf,
                          group_names=None,
                          categories='auto',
                          count=True,
                          percent=True,
                          cbar=True,
                          xyticks=True,
                          xyplotlabels=True,
                          sum_stats=True,
                          figsize=None,
                          cmap='Blues',
                          title=True):
 
    # CODE TO GENERATE TEXT INSIDE EACH SQUARE
    blanks = ['' for i in range(cf.size)]

    if group_names and len(group_names)==cf.size:
        group_labels = ["{}\n".format(value) for value in group_names]
    else:
        group_labels = blanks

    if count:
        group_counts = ["{0:0.0f}\n".format(value) for value in cf.flatten()]
    else:
        group_counts = blanks

    if percent:
        group_percentages = ["{0:.2%}".format(value) for value in cf.flatten()/np.sum(cf)]
    else:
        group_percentages = blanks

    box_labels = [f"{v1}{v2}{v3}".strip() for v1, v2, v3 in zip(group_labels,group_counts,group_percentages)]
    box_labels = np.asarray(box_labels).reshape(cf.shape[0],cf.shape[1])


    # CODE TO GENERATE SUMMARY STATISTICS & TEXT FOR SUMMARY STATS
    if sum_stats:
        #Accuracy is sum of diagonal divided by total observations
        accuracy  = np.trace(cf) / float(np.sum(cf))

        #if it is a binary confusion matrix, show some more stats
        if len(cf)==2:
            #Metrics for Binary Confusion Matrices
            precision = cf[1,1] / sum(cf[:,1])
            recall    = cf[1,1] / sum(cf[1,:])
            f1_score  = 2*precision*recall / (precision + recall)
            stats_text = "\n\nAccuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}".format(
                accuracy,precision,recall,f1_score)
        else:
            stats_text = "\n\nAccuracy={:0.3f}".format(accuracy)
    else:
        stats_text = ""


    # SET FIGURE PARAMETERS ACCORDING TO OTHER ARGUMENTS
    if figsize==None:
        #Get default figure size if not set
        figsize = plt.rcParams.get('figure.figsize')

    if xyticks==False:
        #Do not show categories if xyticks is False
        categories=False


    # MAKE THE HEATMAP VISUALIZATION
    plt.figure(figsize=figsize)
    sns.heatmap(cf,annot=box_labels,fmt="",cmap=cmap,cbar=cbar,xticklabels=categories,yticklabels=categories)

    if xyplotlabels:
        plt.ylabel('True label')
        plt.xlabel('Predicted label' + stats_text)
    else:
        plt.xlabel(stats_text)
    
    if title:
        plt.title(title)
   
  

def show_wrong_prred(xX_test, labelstransform_y_test, xpredicted , xfull_names, xlabel_like_the_file):
    hh = 1
    correct_predict = []
    wrong_predict = []
    print_to_filex = []
    #log("Sample of wrong predictions, full list in the report")
    #log("True_label" + "\t" + "Pred_label" + "\t" + "Entry")
       
    for entry, true_label,  pred_label in zip(xX_test, labelstransform_y_test, xpredicted ):
        if true_label != pred_label:
            true_xx = xfull_names[xlabel_like_the_file.index(xlabel_like_the_file[true_label])]
            pred_xx = xfull_names[xlabel_like_the_file.index(xlabel_like_the_file[pred_label])]
           
            wrong_predict.append("{}\t{}\t\t{}".format("true: " + true_xx,  "pred: " + pred_xx, entry ))
            hh += 1       
        else:
            correct_predict.append("{}\t{}\t{}".format(xlabel_like_the_file[true_label], xlabel_like_the_file[pred_label],  entry))       
    
    
    #log("-------------------------------------")  
    log("") 
    log(str(len(xpredicted)) + "  All predict Entries " )
    log(str(len(wrong_predict)) + "  Wrong predict Entries " )
    log(str(len(correct_predict)) + "  Correct predict Entries " )
    print_to_filex.append("")
    print_to_filex.append("-------------------------------------")
    print_to_filex.append("{}\t{}".format(len(xpredicted), " All predict Entries" ))
    print_to_filex.append("{}\t{}".format(len(wrong_predict), " Wrong predict Entries"))
    print_to_filex.append("{}\t{}".format(len(correct_predict), " Correct predict Entries"))
    print_to_filex.append("")
    print_to_filex.append("#### Wrong predict")
    print_to_filex.append("")
    print_to_filex.append("{}\t{}\t{}".format("True_label",  "Pred_label" , "Entry"))                                 

    
    for w in wrong_predict:
        print_to_filex.append(w)
        
    return print_to_filex

    

