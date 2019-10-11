import networkx as nx
import matplotlib.pyplot as plt
import operator
from colormap import rgb2hex

G= nx.read_edgelist("/Users/shipengyi/Desktop/Kmergraphs/sampleJaccSorted/sample0JaccSorted.txt", data=(('weight',float),))
nx.draw_kamada_kawai(G, with_labels=True)
plt.show()
tmerList = []
temp = ""
for i in range(0,100):
    f = open("/Users/shipengyi/Desktop/Kmergraphs/sampleJaccSorted/sample"+str(i)+"JaccSorted.txt","r")
    print("reading "+str(i)+" JaccSorted")
    for x in f:
        #print(x)
        temp =x
        if x[:3] not in tmerList:
            tmerList.append(x[:3])
        if x[4:7] not in tmerList:
            tmerList.append(x[4:7])

    f = open("/Users/shipengyi/Desktop/Kmergraphs/QsampleJaccSorted/sample" + str(i) + "JaccSorted.txt", "r")
    print("reading " + str(i) + " QJaccSorted")
    for x in f:
        #print(x)
        temp = x
        if x[:3] not in tmerList:
            tmerList.append(x[:3])
        if x[4:7] not in tmerList:
            tmerList.append(x[4:7])

print(tmerList)
print(len(tmerList))
