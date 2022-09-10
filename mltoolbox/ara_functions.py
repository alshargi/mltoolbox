

# -*- coding: utf-8 -*-


# -*- coding: utf-8 -*-


ara_letters = {u"\u0627":'A',
u"\u0628":'b', u"\u062A":'t', u"\u062B":'v', u"\u062C":'j',
u"\u062D":'H', u"\u062E":'x', u"\u062F":'d', u"\u0630":'*', u"\u0631":'r',
u"\u0632":'z', u"\u0633":'s', u"\u0634":'$', u"\u0635":'S', u"\u0636":'D',
u"\u0637":'T', u"\u0638":'Z', u"\u0639":'E', u"\u063A":'g', u"\u0641":'f',
u"\u0642":'q', u"\u0643":'k', u"\u0644":'l', u"\u0645":'m', u"\u0646":'n',
u"\u0647":'h', u"\u0648":'w', u"\u0649":'y', u"\u064A":'Y',
u"\u0622":'|', u"\u064E":'a', u"\u064F":'u', u"\u0650":'i',
u"\u0651":'~', u"\u0652":'o', u"\u064B":'F', u"\u064C":'N',
u"\u064D":'K', u"\u0621":'\'', u"\u0623":'>', u"\u0625":'<',
u"\u0624":'&', u"\u0626":'}', u"\u0629":'p', " ":' '
}




def split(word): 
    return [char for char in word]  
      


def get_key(v, aralettersx): 
    for key, value in aralettersx.items(): 
         if v == value: 
             return key 

    return ""
 

def get_val(k, aralettersx): 
    for key, value in aralettersx.items(): 
         if k == key: 
             return value 

    return ""


def convert_bw_to_ara(xsent):
    res = ''
    splstr = split(xsent)
    for x in splstr:
        cara = get_key(x, xsent)
        if cara != '':
            res += cara
        else:
            res += x

    return res


def convert_ara_to_bw(xsent):
    res = ''
    splstr = split(xsent)
    print(splstr)
    for x in splstr:
        cara = get_val(x, xsent)
        if cara != '':
            res += cara
        else:
            res += x
    return res
