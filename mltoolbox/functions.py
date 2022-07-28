from datetime import datetime
  

def count_now(xnum):  
    for i in range(xnum):
        print("woow", i)



def save_file(xlist, xxpath):
    file1 = open(xxpath,"w")
    for i in xlist:
        file1.writelines("{}\n".format(i))
    file1.close()



def read_json_originalText(fx):
    keepall = []
    f = open(fx,)
    data = json.load(f)
    for i in data:
        keepall.append(i['originalText'])
    f.close()
    return keepall


