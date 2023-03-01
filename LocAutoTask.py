# -*- coding: utf-8 -*-
# # Automation Localization-task
# author: Lex Pod falsharg@ Faisal ALSHARGI
# last updated: Mar/1/2023
####################################

from tempfile import mkstemp
from shutil import move, copymode
from os import fdopen, remove
import shutil
import os
#import json
import csv


validate_report = []
log_file = []
unk = []
sample_all = []

valid_list = [
    "BookFlightAirline\t{ArrivalAirportName}\t{DepartureAirportName}",
    "FlightCheckInAirline\t{ArrivalAirportName}\t{DepartureAirportName}",
    "SearchForFlightsAirline\t{DepartureCityName}\t{ArrivalCityName}",
    "ScheduleAirportPickupTravel\t{ArrivalAirportName}",
    "BookHotelTravel\t{CityName}",
    "SearchForHotelsTravel\t{CityName}"]


def replace(file_path, pattern, subst):
    #Create temp file
    fh, abs_path = mkstemp()
    with fdopen(fh,'w') as new_file:
        with open(file_path) as old_file:
            for line in old_file:
                new_file.write(line.replace(pattern, subst))
    #Copy the file permissions from the old file to the new file
    copymode(file_path, abs_path)
    #Remove original file
    remove(file_path)
    #Move new file
    move(abs_path, file_path)
    
    
def save_file(xlist, xxpath):
    file1 = open(xxpath,"w")
    for i in xlist:
        file1.writelines("{}\n".format(i))
    file1.close()
    
def loadUnqList(p):
    klist = []
    with open(p) as fword:
        klist = fword.read().splitlines()
    return klist
  
def git_tags(vv):
    res = []
    for i in valid_list:
        xx = i.split("\t")
        if xx[0] == vv:
            res = xx[1:]
    return  res

def check_validate(lsttags, sentries):
    rescount = 0
    for t in sentries:
        if lsttags in t :
            rescount += 1    
    return lsttags + " " + str(rescount) + " from " + str(len(sentries)) + " samples" 
    

###### P1   -----------------------

def CreateTaskFiles(task_path, task_path_temp, task_path_input, SAMPLES_file,DOMAIN_name, LOCAL_name, STYLEL_name, SOURCE_name, MODALITY_name, version, voiceId, SIM):
    ####------------------------------------------ Empty files, folder
 
    mainfolder = task_path + DOMAIN_name.lower()
    if os.path.exists(mainfolder):
        shutil.rmtree(mainfolder)
        print("The folder has been deleted successfully!")
        log_file.append("The folder has been deleted successfully!")
    
    
    if not os.path.exists(mainfolder):
        os.makedirs(mainfolder)
        print("main folder created")
        log_file.append(str(DOMAIN_name.lower() + " - Main folder created" ))
                        
    # update path add domain folder
    task_path = task_path + DOMAIN_name.lower() + '/'
    log_file.append("Path updated to : " + str(task_path))
    
    
    # 1 # empty_dialog_act_map.csv       #   utterance,dialog_act,slot_to_elicit
    col_headder = []
    col_headder.append("utterance,dialog_act,slot_to_elicit")
    save_file(col_headder, task_path + "empty_dialog_act_map.csv")
    print("empty_dialog_act_map.csv created")
    log_file.append(" empty_dialog_act_map.csv file - created" )
    
    
    
    # 2 # empty_conversations.csv #conversation_id,turn_no,uid,author_role,intent,dialog_act,elicited_slot,utterance
    col_headder = []
    col_headder.append("conversation_id,turn_no,uid,author_role,intent,dialog_act,elicited_slot,utterance")
    save_file(col_headder, task_path + "empty_conversations.csv")
    print("empty_conversations.csv created")
    log_file.append(" empty_conversations.csv file - created" )
    
                        
    # 3 # Create empty_file.json
    col_headder = []
    save_file(col_headder, task_path + "empty_file.json")
    print("empty_file.json created")
    log_file.append(" empty_file.json file - created" )
    
                        
    # 4 # Create output folder
    newpath = task_path + r'output' 
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    print("output folder created")
    log_file.append(" output folder - created" )
    
    ##---------------------------------------
    
    ###### ---- Copy job_config from template 
    src = task_path_temp + "job_config.yaml"
    dst = task_path + "job_config.yaml"
    shutil.copyfile(src, dst)
    print("job_config.yaml - done")
    log_file.append("job_config.yaml - copy from temp folder - Done" )
    
    
    
    ###### ---- copy bot_<domain> from template 
    src = task_path_temp + "bot_" + DOMAIN_name.lower() + ".json"
    dst = task_path + "bot_" + DOMAIN_name.lower() + ".json"
    shutil.copyfile(src, dst)
    print("bot_" + DOMAIN_name.lower() + ".json", " created")
    log_file.append("bot_" + DOMAIN_name.lower() + ".json - copy from temp folder - Done")
    
    
    
    
    ###### P2 ---------------------------------------- Create job_config.yaml
    #### ------ change local
    ptrn = "locale_xx" 
    newx = 'locale: '  + LOCAL_name 
    replace(task_path + 'job_config.yaml', ptrn, newx)#
    print("wow, UPDATE LOCAL IN  yaml")  
    log_file.append("Update local in job_config.yaml -  Done")
    
    
    #### ------ change domain
    ptrn = "domain_xx" 
    newx = 'domain: '  + DOMAIN_name 
    replace(task_path + 'job_config.yaml', ptrn, newx)#
    print("wow, UPDATE DOMAIN_name IN  yaml")  
    log_file.append("Update Domain name in job_config.yaml -  Done")
    
    #### ------ change modality
    ptrn = "modality_xx" 
    newx = 'modality: '  + MODALITY_name 
    replace(task_path + 'job_config.yaml', ptrn, newx)#
    print("wow, UPDATE MODALITY_name IN  yaml")  
    log_file.append("Update Modality in job_config.yaml -  Done")
    
    
    #### ------ change style
    ptrn = "style_xx" 
    newx = 'style: '  + STYLEL_name 
    replace(task_path + 'job_config.yaml', ptrn, newx)#
    print("wow, UPDATE STYLEL_name IN  yaml")  
    log_file.append("Update Style in job_config.yaml -  Done")
    
    
    #### ------ change SOURCE
    ptrn = "source_xx" 
    newx = 'source: '  + SOURCE_name 
    replace(task_path + 'job_config.yaml', ptrn, newx)#
    print("wow, UPDATE SOURCE_name IN  yaml")  
    log_file.append("Update Source in job_config.yaml -  Done")
    
    
    #### ------ change description
    ptrn = "description_xx" 
    newx = 'description:  >-\n  This is an ' + LOCAL_name + ' ' + DOMAIN_name + ' domain with customer utterances. SIM: ' + SIM 
    replace(task_path + 'job_config.yaml', ptrn, newx)#
    print("wow, UPDATE description IN  yaml")  
    log_file.append("Update description in job_config.yaml -  Done")
    
    
    #### ------ change bot_definition link
    ptrn = "bot_definition_xx" 
    botxx = '"{}"'.format('./bot_' + DOMAIN_name.lower()  + ".json")
    newx = 'bot_definition: '  + botxx
    replace(task_path + 'job_config.yaml', ptrn, newx)#
    print("wow, UPDATE bot_definition IN  yaml")  
    log_file.append("Update bot_definition link in job_config.yaml -  Done")
    
    #### ------ change extra_ic_sl_annotations link
    ptrn = "extra_ic_sl_annotations_xx" 
    extraxx = '"{}"'.format('./' + DOMAIN_name.lower()  + ".csv")
    newx = 'extra_ic_sl_annotations: '  + extraxx
    replace(task_path + 'job_config.yaml', ptrn, newx)#
    print("wow, UPDATE extra_ic_sl_annotations IN  yaml")  
    log_file.append("Update extra_ic_sl_annotations link in job_config.yaml -  Done")
    
    
    ###### P3---------------------------------------- create bot_<Domain>.json
                                               
    with open(task_path_input + SAMPLES_file, 'r') as file:
        next(file) # skip header
        csvreader = csv.reader(file, delimiter=',')
        for row in csvreader:   
            sample_all.append(row)
            if "{}\t{}".format(row[0], row[1]) not in unk:
                unk.append("{}\t{}".format(row[0], row[1]))
    
          
    unk_slots = []
    for t in unk:
        cc = t.split("\t")
        if DOMAIN_name == cc[0]:
            unk_slots.append( cc[1])
        
    print(unk_slots)
    log_file.append("Unique intents "  + str(unk_slots))
    log_file.append("----------------------------------")
    
    #xx = []
    cc = ""
        
    for unks in unk_slots:
        samples_entry = []
        print("####", unks)
        for sm in sample_all:
            if (sm[0] == DOMAIN_name) and (sm[1] == unks):
                samples_entry.append(sm[2])
        
        print(unks)   
        log_file.append("##### " + str(unks))
        check_this = git_tags(unks)
        print("") 
        print("check this tags", check_this)
        log_file.append("##### check these tags : " + str(check_this))
        for tt in check_this:
            print(check_validate(tt, samples_entry ))
            log_file.append(check_validate(tt, samples_entry ))
    
        print("")  
        log_file.append("")
        #####
        cc = ""
        for i in samples_entry:
            cc += '                    "{}"'.format(i) + ',\n'
        cc = cc[:-2]#
        ptrn = unks + '_sampleUtterances'
        ptrn = '"{}"'.format(ptrn)
        #print(ptrn)
        replace(task_path + 'bot_'  + DOMAIN_name.lower() + ".json", ptrn, "[\n" + cc + "\n                  ],")#
        #print("wow, update")
        log_file.append("Update bot json  file " + str(ptrn))
    
       
        
    log_file.append("")
    #### ------ change VoiceID
    ptrn = '"voiceId_xx"'
    v_id = '"{}"'.format(voiceId)
    newx = '"voiceId": ' + v_id + ','
    replace(task_path + 'bot_'  + DOMAIN_name.lower() + ".json", ptrn, newx)#
    print("wow, voiceid changes")  
    log_file.append("Update  voiceID - Done")
    
        
    #### ------ change Locale
    ptrn = '"locale_xx"'
    loc_com = '"{}"'.format(LOCAL_name)
    newx = '"locale": ' + loc_com + ','
    replace(task_path + 'bot_'  + DOMAIN_name.lower() + ".json", ptrn, newx)#
    print("wow, local changes")  
    log_file.append("Update  Local - Done")
    
     
    #### ------ change Locale
    ptrn = '"version_xx"'
    vers_com = '"{}"'.format(version)
    newx = '"version": ' + vers_com + ','
    replace(task_path + 'bot_'  + DOMAIN_name.lower() + ".json", ptrn, newx)#
    print("wow, version changes")  
    log_file.append("Update  version - Done")
    
    
    ################### save log file
    save_file(log_file, "./" + DOMAIN_name + "_log.txt")
    
    
    # END
