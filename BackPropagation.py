# -*- coding: utf-8 -*-
"""
Created on Thu May 17 00:38:28 2018

@author: Antraxiana
"""

# -*- coding: utf-8 -*-
"""
Created on Thu May 16 16:25:08 2018

@author: Axel
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

#Dataset
data = pd.read_csv("iris - Copy.csv", header = None, names = ["X1", "X2", "X3", "X4", "Class"])
data["Setosa"] = np.where(data["Class"]=="Iris-setosa",1,0)
data["Versicolor"] = np.where(data["Class"]=='Iris-versicolor',1,0)
data["Virginica"] = np.where(data["Class"]=='Iris-virginica',1,0)
dataset=data.values.tolist()


#Function
def fH(X, w, b):
    ans = []
    for j in range(0, len(w)):
        H=0
        for i in range(0,len(w[j])):
            H+=X[i]*w[j][i]
        H+= b[j]
        ans.append(H)
    return ans

def sigmoid(h):
    ans = []
    for i in range(0, len(h)):
        ans.append(1/(1+math.e**-h[i]))
    return ans

def softmax(h):
    ans = []
    for i in range(0, len(h)):
        ans.append(math.e**h[i]/(math.e**h[0]+math.e**h[1]+math.e**h[2]))
    return ans
    
def error(dataset, sigo):
    ans = []
    for i in range(0, len(sigo)):
        ans.append((dataset[i+5]-sigo[i])**2)
    return ans

def predic(val):
    if (val[0] > val[1]) and (val[0] > val[2]):
        return "Iris-setosa"
    elif (val[1] > val[0]) and (val[1] > val[2]):
        return "Iris-versicolor"
    elif (val[2] > val[0]) and (val[2] > val[1]):
        return "Iris-virginica"


def deltaO(fact, s):
    ans = []
    for i in range(0, len(s)):
        ans.append(2*(s[i]-fact[i+5])*(1-s[i])*s[i])
    return ans

def deltaH(out, dO, sigH):
    ans= []
    for j in range(0,len(sigH)):
        delta = 0
        for i in range(0,len(dO)):
            delta+=(out[i][j]*dO[i])
        delta*=(1-sigH[j])*sigH[j]
        ans.append(delta)
    return ans


def NewWeight(delta,x,w):
    ans = []
    for j in range(0, len(delta)):
        listw = []
        for i in range(0, len(w[j])):
            dw=x[i]*delta[j]
            weight= w[j][i]-(a*dw)
            listw.append(weight)
        ans.append(listw)
    return ans

def NewBias(delta,b):
    ans = []
    for i in range(0, len(b)):
        ans.append(b[i]-(a*delta[i]))
    return ans

# defining Feedforward===============================================

def BP(i,dataset,wh,bh,wo,bo):
    Hh = fH(dataset[i],wh,bh)
    sigh = sigmoid(Hh)
    Ho = fH(sigh,wo,bo)
    sigo = sigmoid(Ho)
    deltao = deltaO(dataset[i], sigo);
    deltah = deltaH(wo, deltao, sigh);
    pre = predic(sigo)
    err = error(dataset[i], sigo)
    newWO = NewWeight(deltao, sigh, wo)
    newWH = NewWeight(deltah, dataset[i],wh)
    newBO = NewBias(deltao,bo)
    newBH = NewBias(deltah,bh)
    output = [pre,err,newWO,newWH,newBO,newBH]
    return output
    

def BPSoftmax(i,dataset,wh,bh,wo,bo):
    Hh = fH(dataset[i],wh,bh)
    softh = softmax(Hh)
    Ho = fH(softh,wo,bo)
    softo = softmax(Ho)
    deltao = deltaO(dataset[i], softo);
    deltah = deltaH(wo, deltao, softh);
    pre = predic(softo)
    err = error(dataset[i], softh)
    newWO = NewWeight(deltao, softh, wo)
    newWH = NewWeight(deltah, dataset[i],wh)
    newBO = NewBias(deltao,bo)
    newBH = NewBias(deltah,bh)
    output = [pre,err,newWO,newWH,newBO,newBH]
    return output

def MLBP(dataset,wh,bh,wo,bo):
    AvgErrTrain = []
    AvgErrValid = []
    for j in range(0, 100):
        dataTrain = []
        dataValid = []
        if j%5 == 0:
            for i in range(0, len(dataset)):
                if (0 <= i <10) or (50<= i <60) or (100<= i <110):
                    dataValid.append(dataset[i])
                else:
                    dataTrain.append(dataset[i])
        elif j%5 == 1:
            for i in range(0, len(dataset)):
                if (10 <= i <20) or (60<= i <70) or (110<= i <120):
                    dataValid.append(dataset[i])
                else:
                    dataTrain.append(dataset[i])
        elif j%5 == 2:
            for i in range(0, len(dataset)):
                if (20 <= i <30) or (70<= i <80) or (120<= i <130):
                    dataValid.append(dataset[i])
                else:
                    dataTrain.append(dataset[i])
        elif j%5 == 3:
            for i in range(0, len(dataset)):
                if (30 <= i <40) or (80<= i <90) or (130<= i <140):
                    dataValid.append(dataset[i])
                else:
                    dataTrain.append(dataset[i])
        elif j%5 == 4:
            for i in range(0, len(dataset)):
                if (40 <= i <50) or (90<= i <100) or (140<= i <150):
                    dataValid.append(dataset[i])
                else:
                    dataTrain.append(dataset[i])
        #train
        ErrSeto = []
        ErrVers = []
        ErrVirg = []
        AvgErr = []
        for i in range(0, len(dataTrain)):
            output = BP(i,dataTrain,wh,bh,wo,bo)
            ErrSeto.append(output[1][0])
            ErrVers.append(output[1][1])
            ErrVirg.append(output[1][2])
            wo = output[2]
            wh = output[3]
            bo = output[4]
            bh = output[5]
            AvgErr.append((ErrSeto[i]+ErrVers[i]+ErrVirg[i])/3)
        Terr = 0
        for i in range(0, len(AvgErr)):
            Terr = Terr + AvgErr[i]
        AvgErrTrain.append(Terr / len(AvgErr))
        
        #valid
        predicV = []
        ErrSetoV = []
        ErrVersV = []
        ErrVirgV = []
        AvgErrV = []
        for i in range(0, len(dataValid)):
            outputV = BP(i,dataValid, wh, bh, wo, bo)
            ErrSetoV.append(outputV[1][0])
            ErrVersV.append(outputV[1][1])
            ErrVirgV.append(outputV[1][2])
            AvgErrV.append((ErrSetoV[i]+ErrVersV[i]+ErrVirgV[i])/3)
            if j == 99:
                predicV.append(outputV[0])
        TerrV = 0
        for i in range(0, len(AvgErrV)):
            TerrV = TerrV + AvgErrV[i]
        AvgErrValid.append(TerrV / len(AvgErrV))
        
    #Accuracy checking    
    CountTrue = 0
    for i in range(0, len(predicV)):
        if predicV[i] == dataValid[i][4]:
            CountTrue = CountTrue + 1
    accuracy = (CountTrue/len(predicV))*100
    
    #plot
    plt.plot(AvgErrTrain, label = "train")
    plt.plot(AvgErrValid, label = "validation")
    plt.legend()
    plt.ylabel('Error')
    plt.xlabel('Epoch')
    plt.title('Sigmoid Error Graph')
    plt.show()
    print("Accuracy: {}%".format(accuracy))
    
def MLBPSoftmax(dataset, wh, bh, wo, bo):    
    AvgErrTrain = []
    AvgErrValid = []
    for j in range(0, 100):
        dataTrain = []
        dataValid = []
        if j%5 == 0:
            for i in range(0, len(dataset)):
                if (0 <= i <10) or (50<= i <60) or (100<= i <110):
                    dataValid.append(dataset[i])
                else:
                    dataTrain.append(dataset[i])
        elif j%5 == 1:
            for i in range(0, len(dataset)):
                if (10 <= i <20) or (60<= i <70) or (110<= i <120):
                    dataValid.append(dataset[i])
                else:
                    dataTrain.append(dataset[i])
        elif j%5 == 2:
            for i in range(0, len(dataset)):
                if (20 <= i <30) or (70<= i <80) or (120<= i <130):
                    dataValid.append(dataset[i])
                else:
                    dataTrain.append(dataset[i])
        elif j%5 == 3:
            for i in range(0, len(dataset)):
                if (30 <= i <40) or (80<= i <90) or (130<= i <140):
                    dataValid.append(dataset[i])
                else:
                    dataTrain.append(dataset[i])
        elif j%5 == 4:
            for i in range(0, len(dataset)):
                if (40 <= i <50) or (90<= i <100) or (140<= i <150):
                    dataValid.append(dataset[i])
                else:
                    dataTrain.append(dataset[i])
        
        #Train
        ErrSeto = []
        ErrVers = []
        ErrVirg = []
        AvgErr = []
        for i in range(0, len(dataTrain)):
            output = BPSoftmax(i,dataTrain,wh,bh,wo,bo)
            ErrSeto.append(output[1][0])
            ErrVers.append(output[1][1])
            ErrVirg.append(output[1][2])
            wo = output[2]
            wh = output[3]
            bo = output[4]
            bh = output[5]
            AvgErr.append((ErrSeto[i]+ErrVers[i]+ErrVirg[i])/3)
        Terr = 0
        for i in range(0, len(AvgErr)):
            Terr=Terr+AvgErr[i]
        AvgErrTrain.append(Terr / len(AvgErr))
        
        #Valid
        predicV = []
        ErrSetoV = []
        ErrVersV = []
        ErrVirgV = []
        AvgErrV = []
        for i in range(0, len(dataValid)):
            outputV = BPSoftmax(i, dataValid,wh,bh,wo,bo)
            ErrSetoV.append(outputV[1][0])
            ErrVersV.append(outputV[1][1])
            ErrVirgV.append(outputV[1][2])
            AvgErrV.append((ErrSetoV[i] + ErrVersV[i] + ErrVirgV[i])/3)
            if j == 99:
                predicV.append(outputV[0])        
        TerrV = 0
        for i in range(0, len(AvgErrV)):
            TerrV = TerrV + AvgErrV[i]
        AvgErrValid.append(TerrV / len(AvgErrV))
        
    #Accuracy Checking
    CountTrue = 0
    for i in range(0, len(predicV)):
        if predicV[i] == dataValid[i][4]:
            CountTrue = CountTrue + 1
    accuracy = (CountTrue/len(predicV))*100
    
    #plot
    plt.plot(AvgErrTrain, label = "train")
    plt.plot(AvgErrValid, label = "validation")
    plt.legend()
    plt.ylabel('Error')
    plt.xlabel('Epoch')
    plt.title('Softmax Error Graph')
    plt.show()
    print("accuracy: {}%".format(accuracy))
        
#=============================================================================
a=0.1
wh =[[0.21,0.32,0.23,0.21],
     [0.12,0.13,0.23,0.11],
     [0.14,0.31,0.33,0.42],
     ]
bh = [0.12,0.33,0.11]

wo =[[0.32,0.12,0.13],
     [0.31,0.11,0.23],
     [0.42,0.13,0.31],
     ]
bo = [0.12,0.23,0.12]
MLBP(dataset,wh,bh,wo,bo)
MLBPSoftmax(dataset,wh,bh,wo,bo)
