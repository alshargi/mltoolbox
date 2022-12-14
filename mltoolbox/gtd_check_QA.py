

from .functions import log
import re

def load_txt_file(p):
    klist = []
    with open(p) as fword:
        klist = fword.read().splitlines()
    return klist

  
def check_duplication(welo_entry,  xkeepall_welo):
  xres = 0
  for x in xkeepall_welo:
    if welo_entry.strip() == x.strip():
      xres += 1
  return xres


def check_GTD_for_CR(file_from_vendor):
    printtofile = []
    keepall_welo = []
    keepall_welo = load_txt_file(file_from_vendor)

    log("File Name: " + str(file_from_vendor))
    printtofile.append("File Name: " + str( file_from_vendor)) 
    printtofile.append("# Entries : " + str(len(keepall_welo))) 
    log("# Entries : " + str(len(keepall_welo)))


    ######-------------------------- check the Length of Entries
    log("########----- Entries Length")
    printtofile.append("########----- Entries Length") 
    alllength = []
    for tx in keepall_welo:
        wordsnum = tx.split()
        alllength.append(len(wordsnum))
 
 
    greatellf = 0
    for u in range(1,11):
        log('length #  ' + str(u) + ' words \t\t' + str(alllength.count(u)) + '\t' + 'Entries')
        printtofile.append('length #  ' + str(u) + ' words \t\t' + str(alllength.count(u)) + '\t' + 'Entries')
    
    
    for p in alllength:
        if p >= 11:
            greatellf += 1
    log('length>=  ' + str(11) + ' words \t\t' + str(greatellf) + '\t' + 'Entries')
    printtofile.append('length>=  ' + str(11) + ' words \t\t' + str(greatellf) + '\t' + 'Entries')
    log("")
    printtofile.append("")
      
    ######-------------- Check the multi spaces
    log("########----- Spaces")
    printtofile.append("########----- Spaces")
    
    bb = 1
    bb_st = 0
    for tx in keepall_welo:
        bb += 1
        res = bool(re.search(r"\s\s", tx))
        if res == True:
            bb_st = 1
            log('FAILED \t Row# ' + str(bb), res, tx)
            printtofile.append("{}\t{}\t{}".format('FAILED Row# ' + str(bb), res, tx))
    
    
    if bb_st == 0:
        log("PASSed \t No Extra spaces in the entries")
        printtofile.append("PASSed \t No Extra spaces in the entries")
    
    log("")
    printtofile.append("")
    
    ######-------------------------- Check duplicate entries
    log("")
    log("#######--------- Duplicate Entries ")
    printtofile.append("")
    printtofile.append("#######--------- Duplicate Entries ")
    
    keepunientry = []
    ccd = 0
    st2 = 0
    for y in keepall_welo:
      check_res = check_duplication(y, keepall_welo)   
      ccd += 1
      if check_res > 1:
            if y not in keepunientry:
                st2 += 1
                keepunientry.append(y)
                log(str(check_res) + ' times' +  " in Row# " + str(ccd)+ '\t' +  " Entry: " + str(y))
                printtofile.append("{}\t{}\t{}".format(str(check_res) + ' times', "in Row# " + str(ccd)+'\t', "Entry: " + y))
    
    if st2 == 0:
        log('No Duplicate Entries in the file')
        printtofile.append("No Duplicate Entries in the file")
        
    
    printtofile.append("##########################################")
    printtofile.append("")
    return printtofile

   
