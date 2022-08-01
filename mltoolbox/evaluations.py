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



def log(string):
    now = str(datetime.now())
    print(Fore.BLUE + now + ' ' + Style.RESET_ALL + string)

    
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
    print_to_filex.append("{}\t{}".format(len(predicted), " All predict Entries" ))
    print_to_filex.append("{}\t{}".format(len(wrong_predict), " Wrong predict Entries"))
    print_to_filex.append("{}\t{}".format(len(correct_predict), " Correct predict Entries"))
    print_to_filex.append("")
    print_to_filex.append("#### Wrong predict")
    print_to_filex.append("")
    print_to_filex.append("{}\t{}\t{}".format("True_label",  "Pred_label" , "Entry"))                                 

    
    for w in wrong_predict:
        print_to_filex.append(w)
        
    return print_to_filex

    

