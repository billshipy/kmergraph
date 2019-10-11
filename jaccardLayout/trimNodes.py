tmerList=[]
finalList=[]

#CHANGE NODE NUM RESTRICTION HERE:
nodeNum = 20

for i in range(0,1500):
    f = open("/Users/shipengyi/Desktop/largeData/PrxQ/sample"+str(i)+"JaccSorted.txt","r")
    #print("reading "+str(i)+" JaccSorted")
    for x in f:
        #print(x)
        if x[:3] not in tmerList and len(tmerList)<nodeNum:
            tmerList.append(x[:3])
            if len(tmerList) == nodeNum:
                break
        if x[4:7] not in tmerList and len(tmerList)<nodeNum:
            tmerList.append(x[4:7])
            if len(tmerList) == nodeNum:
                break
    finalList.append(tmerList)
    tmerList=[]
    #print(tmerList)
    print(len(finalList[i]))
    #print("\n")


for i in range(0,1500):
    f = open("/Users/shipengyi/Desktop/largeData/PrxQ/sample"+str(i)+"JaccSorted.txt","r")
    j = open("/Users/shipengyi/Desktop/largeData/PrxQTrim/sample"+str(i)+"JaccSorted.txt", "w")
    #print("reading "+str(i)+" JaccSorted")
    #print(finalList[i])
    for x in f:
        if x[:3] in finalList[i] and x[4:7] in finalList[i]:
            j.write(x)
            #print("123")

