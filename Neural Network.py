# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 15:45:59 2018

@author: Axel
"""

import pylab as pl
import numpy as np
import csv
import random
import copy

#Importing data from CSV file
def read_lines():
    with open('iris22.csv') as data:
        reader = csv.reader(data)
        for row in reader:
            yield [ float(i) for i in row ]
#list of Iris Dataset. x5 is class
x = list(read_lines())

#Input Epoch and Alpha value
#n= int(input('enter epoch: '))
#a= float(input('enter alpha: '))


#===========random theta
def theta():
    return random.uniform(0,1)
def tb():
    return random.uniform(0,1)

#====================function for h(x,theta,b)
def h(tq,i,b):
    n=len(tq)
    ans=0.0
    for j in range (n):
        ans+=tq[j] * x[i][j]
    ans+= b
#    ans=tq[0]*x[i][0]+tq[1]*x[i][1]+tq[2]*x[i][2]+tq[3]*x[i][3]+b
    return ans

def hbp(tq,i,b,data):
    n=len(tq)
    ans=0.0
    for j in range (n):
        ans+=tq[j] * data[i][j]
    ans+= b
#    ans=tq[0]*x[i][0]+tq[1]*x[i][1]+tq[2]*x[i][2]+tq[3]*x[i][3]+b
    return ans

#=====================function for Sigmoid(h)
def sigmoid(ha):
    return 1/(1+np.exp(-ha))

#====================loss Function
def error(e,s):
    return (s-x[e][4])**2    

#Delta Function for Theta 1(q0), Theta 2(q1), Theta 3(q2), Theta 4(q3), and bias
def deltaq(d,j,s):
    return 2*(s-x[d][4])*(1.0-s)*s*x[d][j]

def deltab(d,s):
    return 2*(s-x[d][4])*(1.0-s)*s*1

#New Value function for Theta 1(q0), Theta 2(q1), Theta 3(q2), Theta 4(q3), and bias
def newq(tq,a,dt,n):
    return tq[n]-(a*dt[n])

def newb(b,a,db):
    return b-(a*db)

#Split Iris dataset into 2 datasets
def sp(x):
    sp.iris1 = sum([x[split:split+50] for split in range(0, len(x),len(x))],[])
    sp.iris2 = sum([x[split:split+50] for split in range(50, len(x), 50)],[])
    return 0

def split(arr, size):
     arrs = []
     while len(arr) > size:
         pice = arr[:size]
         arrs.append(pice)
         arr   = arr[size:]
     arrs.append(arr)
     return arrs
FFerr=[]
BPerr=[]
#============================================================================
#======================= Feed Forward k-fold Only ===========================
def MLFF(ww1,ww2,wwb1,wwb2,data):
    for m in range (len(dataset)):
        if (m==0):
            for i in range (0,len(data[0])):
            #hiddenlayer
                hidden=[]
                for j in range (len(ww1)):
                    tq=ww1[j]
                    dt=copy.deepcopy(tq)
                    b=wwb1
                    ha=hbp(tq,i,b,data[0])
                    s=sigmoid(ha)
                    hidden.append(s)
                    for j in range (len(tq)):
                        dt[j]=deltaq(i,j,s)
                    db=deltab(i,s)
                    for j in range (len(tq)):
                        tq[j]=newq(tq,a,dt,j);
                    b=newb(b,a,db)
                    
            #output
                y=[]
                for j in range (len(ww2)):
                    tq=ww2[j]
                    dt=copy.deepcopy(tq)
                    b=wwb2
                    ha=hbp(tq,i,b,data[0])
                    s=sigmoid(ha)
                    y.append(sigmoid(ha))
                    e=error(i,s)
                    MLFF.te+=e
                    for j in range (len(tq)):
                        dt[j]=deltaq(i,j,s)
                    db=deltab(i,s)
                    for j in range (len(tq)):
                        tq[j]=newq(tq,a,dt,j);
                    b=newb(b,a,db)
                
        else:
            MLFF.a=m
            for i in range (0,len(data[m])):
            #hiddenlayer
                hidden=[]
                for j in range (len(ww1)):
                    tq=ww1[j]
                    b=wwb1
                    ha=hbp(tq,i,b,data[0])
                    hidden.append(sigmoid(ha))
                    
            #output
                y=[]
                for j in range (len(w2)):
                    tq=ww2[j]
                    b=wwb2
                    ha=hbp(tq,i,b,data[0])
                    s=sigmoid(ha)
                    y.append(sigmoid(ha))
                    e=error(i,s)
                    MLFF.te+=e


def layerFF(dataset,weight,weightb):
    collect=[]
    for neuron in range (len(weight)):
        collect.append(MLFF(dataset,weight,weightb,neuron))
    return collect

#============================================================================
#====================== Back Propagation k-fold Only ========================
#============================================================================
def MLBP(w1,w2,wb1,wb2,data):
    for m in range (len(dataset)):
        if (m==0):
            for i in range (0,len(data[0])):
            #hiddenlayer
                hidden=[]
                for j in range (len(w1)):
                    tq=w1[j]
                    b=wb1
                    ha=hbp(tq,i,b,data[0])
                    hidden.append(sigmoid(ha))
                    
            #output
                y=[]
                for j in range (len(w2)):
                    tq=w2[j]
                    b=wb2
                    ha=hbp(tq,i,b,data[0])
                    s=sigmoid(ha)
                    y.append(sigmoid(ha))
                    e=error(i,s)
                    MLBP.te+=e
                
            #theta
                #weight2
                thetahidden=[]
                for row in range (len(w2)):
                    rows=[];temp=w2[row];
                    for column in range(len(temp)):
                        rows.append(2*(y[row]-data[0][i][4])*y[row]*(1-y[row])*hidden[column])
                    thetahidden.append(rows)
                thetabias2=2*(y[row]-data[0][i][4])*y[row]*(1-y[row])*1
                #weight1
                thetainput=[]
                for row in range (len(w1)):
                    rows=[];temp=w1[row];
                    for column in range(len(temp)):
                        rows.append(2*(thetabias2*temp[0]+thetabias2*temp[1]+thetabias2*temp[2])*hidden[row]*(1-hidden[row])*data[0][i][column])
                    thetainput.append(rows)
                thetabias1=2*(thetabias2*temp[0]+thetabias2*temp[1]+thetabias2*temp[2])*hidden[row]*(1-hidden[row])*1
                MLBP.a=thetahidden;MLBP.b=thetainput
            #update
                for row in range (len(w2)):
                    temp=w2
                    for column in range(len(temp)):
                        w2[row][column]=w2[row][column]-a*thetahidden[column][row]
                for row in range (len(w1)):
                    temp=w1
                    for column in range(len(temp)):
                        w1[row][column]=w1[row][column]-a*thetainput[column][row]
                wb1=wb1-a*thetabias1
                wb2=wb2-a*thetabias2
                
        else:
            for i in range (0,len(data[m])):
            #hiddenlayer
                hidden=[]
                for j in range (len(w1)):
                    tq=w1[j]
                    b=wb1
                    ha=hbp(tq,i,b,data[0])
                    hidden.append(sigmoid(ha))
                    
            #output
                y=[]
                for j in range (len(w2)):
                    tq=w2[j]
                    b=wb2
                    ha=hbp(tq,i,b,data[0])
                    s=sigmoid(ha)
                    y.append(sigmoid(ha))
                    e=error(i,s)
                    MLBP.te+=e
            
#==============STEP===================
epoch=100
a=0.1
################################ FeedForward #################################
#dataTEST=[valid[0]]
dataset=split(x,30)
w1=[
   [theta(), theta(), theta(), theta()],
   [theta(), theta(), theta(), theta()],
   [theta(), theta(), theta(), theta()]
   ]
wb1=tb()
ww1=copy.deepcopy(w1)
wwb1=copy.deepcopy(wb1)

w2=[
   [theta(), theta(), theta()]
   ]
wb2=tb()
ww2=copy.deepcopy(w2)
wwb2=copy.deepcopy(wb2)

temp=[]
for n in range (epoch):
    MLFF.te=0
    MLFF(ww1, ww2, wwb1, wwb2, dataset)
    FFerr.append(MLFF.te/epoch)
    for c in range(len(dataset)-1):
        temp=dataset[c];dataset[c]=dataset[c+1];dataset[c+1]=temp
pl.plot(FFerr)

#############################BackPropagation##################################
for n in range (epoch):
    MLBP.te=0
    MLBP(w1,w2,wb1,wb2,dataset)
    BPerr.append(MLBP.te/epoch)
    for c in range(len(dataset)-1):
        temp=dataset[c];dataset[c]=dataset[c+1];dataset[c+1]=temp
pl.plot(BPerr,'-r')
pl.legend(['FeedForward', 'Backpropagation'], loc='upper right')