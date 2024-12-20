def yVal(file, n_lev):
    file = open(file, "r")
    a = file.read()
    file.close()
    a = a.split(" ") #Ancora array di stringhe
    offset = n_lev*1000
    y = [0]*100
    for i in range(100):
        for j in range(20):
            if a[offset+i+j*100]=="\n":
                break
            y[i]+=float(a[offset+i+j*100]) #Offset per il bug di scrittura nei file
    y1 = [float(x)/20 for x in y]
    return y1
def yLast(file, n_lev):
    file = open(file, "r")
    a = file.read()
    file.close()
    a = a.split(" ") #Ancora array di stringhe
    offset = n_lev*1000
    i = 99
    y = []
    for j in range(20):
        #if a[offset+i+j*100]=="\n":
         #   break
        y.append(float(a[offset+i+j*100])) #Offset per il bug di scrittura nei file
    return y