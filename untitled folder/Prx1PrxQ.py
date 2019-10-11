import matplotlib.pyplot as plt
import statistics as st
finalAccList = []
finalValList = []
accList=[]
val_accList=[]
avgAcc = []
avgVal = []
numOfEpoch = 300

for i in range (5, 25):
    reader = open("/Users/shipengyi/Desktop/largeData/NoEdgeOutput/output"+str(i)+".txt","r")
    for x in reader:
        w, h = map(float, x.split())
        accList.append(w)
        val_accList.append(h)
    finalAccList.append(accList)
    accList= []
    finalValList.append(val_accList)
    val_accList = []


for i in range(0,300):
    avgAcc.append((finalAccList[0][i]+finalAccList[1][i]+finalAccList[2][i]+finalAccList[3][i]+finalAccList[4][i])/5)
    avgVal.append((finalValList[0][i]+finalValList[1][i]+finalValList[2][i]+finalValList[3][i]+finalValList[4][i])/5)

#plt.plot(range(len(avgAcc)), avgAcc, 'r', label='Training acc')
#plt.plot(range(len(avgVal)), avgVal, 'b', label='Validation acc')
#plt.title('Training and validation loss')
#print(avgAcc[299])
#print(avgVal[299])
#plt.figure()
#plt.savefig("NoColorOutput.png")
#plt.show()

sAcc = sorted(avgAcc)
sVal = sorted(avgVal)
plt.plot(range(len(sAcc)), sAcc, 'r', label='Training Acc')
plt.plot(range(len(sVal)), sVal, 'b', label='Validation Acc')
plt.title('Training and validation Acc')
plt.savefig("NoEdgeAcc.png")
#plt.show()
print(str(st.mean(avgAcc))+" "+str(st.median(avgAcc))+" "+str(st.stdev(avgAcc))+" "+str(max(avgAcc)))

print(str(st.mean(avgVal))+" "+str(st.median(avgVal))+" "+str(st.stdev(avgVal))+" "+str(max(avgVal)))

print((sAcc[149]+sAcc[150])/2)
#median should be in sorted order so whether we shouold use median or just the original list?
