import networkx as nx
import matplotlib.pyplot as plt
import operator
from colormap import rgb2hex
import csv

## These are example data sets just for testing out how things work
# nodeList = ["A","B","C","D"]
# edgeList = [("A","B",0.9),("A","C",1.0),("C","D",0.7)]
# nodeList = ["KWS","WSV","CPT","PTE","FVC","VCP"]
# edgeList = [("KWS","WSV",1.0),("CPT","PTE",0.9925),("FVC","VCP", 0.9857)]
color_map = []
colorR = 255
colorG = 255
colorB = 255
colorDic = {}
def assign3merColor(tmerName):
    global colorR
    global colorB
    global colorG
    global colorDic
    if tmerName not in colorDic:
        if len(colorDic)%3==0:
            colorR -= 52
            if colorR < 0:
                colorR += 255
            colorDic[tmerName] = rgb2hex(colorR, 0, 0)

        elif len(colorDic)%3==1:
            colorG -= 52
            if colorG < 0:
                colorG += 255
            colorDic[tmerName] = rgb2hex(0, colorG, 0)

        elif len(colorDic)%3== 2:
            colorB -= 52
            if colorB < 0:
                  colorB += 255
            colorDic[tmerName] = rgb2hex(0, 0, colorB)
    return colorDic


def dicWriter(dic):
    w = csv.writer(open("/Users/shipengyi/Desktop/largeData/colors.txt", "w"))
    for key, val in dic.items():
        w.writerow([key, val])


def dicReader(filename, returnDic):
    with open(filename, 'rb') as f:
        reader = csv.reader(f)
        returnDic= dict(reader)
        return returnDic



# could be easier with 1 single color var but could cause trouble, the following
# code is for creating a color map from node name to rgb color
# FIX ME: creating a dictionary is easy but how to keep this assigning process running, keep color RGB? or anything else
tmerList = []
temp = ""
for i in range(0,1500):
    f = open("/Users/shipengyi/Desktop/largeData/PrxQTrim/sample"+str(i)+"JaccSorted.txt","r")
    #print("reading "+str(i)+" JaccSorted")
    for x in f:
        #print(x)
        temp =x
        if x[:3] not in tmerList:
            tmerList.append(x[:3])
        if x[4:7] not in tmerList:
            tmerList.append(x[4:7])

    f = open("/Users/shipengyi/Desktop/largeData/Prx1Trim/sample"+str(i)+"JaccSorted.txt","r")
    #print("reading " + str(i) + " QJaccSorted")
    for x in f:
        #print(x)
        temp = x
        if x[:3] not in tmerList:
            tmerList.append(x[:3])
        if x[4:7] not in tmerList:
            tmerList.append(x[4:7])

for i in range(0, len(tmerList)):
    assign3merColor(tmerList[i])
print(colorDic)

# testing dicwriter and dicreader
dicWriter(colorDic)
#name = "/Users/shipengyi/Desktop/jaccLayout/colors.txt"
#rd = {}
#dicReader(name, rd)
#print(rd)

# nx.draw_kamada_kawai(exampleGraph, node_color=color_map, with_labels=True, font_weight='normal', font_size=6)
# plt.show()
# # print(color_map)
# graphDegreeView = exampleGraph.degree()
# clusteringCoefficients = nx.clustering(exampleGraph)

# print(graphDegreeView)
# print(clusteringCoefficients)
# jaccardEdgeData = []
# for (u, v, jaccard) in exampleGraph.edges.data('jaccard'):
# print('(%s, %s, %.4f)' % (u, v, jaccard))
# jaccardEdgeData.append((u,v,jaccard))
# print(jaccardEdgeData)
# sortedJaccardEdgeData = sorted(jaccardEdgeData, key=operator.itemgetter(2))
# print(sortedJaccardEdgeData)