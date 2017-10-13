# -*- coding: utf-8 -*-
# 函数形为y=kx+b ,leastsq
import numpy as np

import matplotlib.pyplot as plt
from scipy.optimize import leastsq

def readData(fileName):
    xcord=[]
    ycord=[]
    fr=open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split(',')
        xcord.append(float(lineArr[0]))
        ycord.append(float(lineArr[1]))        
    
    return xcord,ycord
    
X_raw,Y_raw=readData("data/lsm_data.csv")

#y=k*x + b
def func(params,x):
    k,b=params
    return k*x+b

def error(params,x,y,s):
    print(s)
    return func(params,x)-y

init_params = [1,1];
###主函数从此开始###

X=np.array(X_raw)
Y=np.array(Y_raw)

s="The number of iteration" #试验最小二乘法函数leastsq得调用几次error函数才能找到使得均方误差之和最小的k、b
Para=leastsq(error,init_params,args=(X,Y,s)) #把error函数中除了p以外的参数打包到args中
k,b=Para[0]
print("k="+str(k)+ ";b="+str(b))


plt.scatter(X_raw,Y_raw,s=30,c='red',marker='s')
 
x=np.linspace(100,240,20)
y=k*x+b
plt.plot(x,y,'b--')
plt.show()