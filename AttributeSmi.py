import numpy as np
import math
import dtw as DTW
from numpy import array, zeros, argmin, inf, equal, ndim
import ReadMatrix as RW

def fbs(a, b):
    return math.fabs(a - b)

def DTWandMaxDist(feature_a,feature_b,fbs):
    r, c = len(feature_a), len(feature_b)
    D0 = zeros((r + 1, c + 1))
    D0[0, 1:] = inf
    D0[1:, 0] = inf
    D1 = D0[1:, 1:]

    # 相似度，标准化即可
    temp0 = normalizing(feature_a)
    temp1 = normalizing(feature_b)

    D1[-1, -1], C, D1, path = DTW.dtw(temp0, temp1, fbs)
    a = path[0]
    b = path[1]
    dist = []
    for i in range(0, len(a)):
        dist.append(math.fabs(a[i] - b[i]))
    maxdis = max(dist)
    # print("maxdis", maxdis)
    return D1[-1, -1],maxdis

def normalizing(alist):
    MinValue0 = min(alist)
    MaxValue0 = max(alist)
    normalize = []
    for i in range(0, len(alist)):
        y0 = (alist[i] - MinValue0) / (MaxValue0 - MinValue0)
        normalize.append(y0)
    return normalize

def PAX(hang,lie,data,PAX_num):
    PAX_data=[]
    sum=0

    for i in range(0, len(data)):
        temp=[]
        for j  in range(0,len(data[i])):
            if j%PAX_num==PAX_num-1:
                sum = sum + data[i][j]
                temp.append(sum/PAX_num)
                sum=0
            else:
                sum = sum+data[i][j]
        PAX_data.append(temp)

    MatrixPAX = np.zeros((hang,lie))

    maxDisL = []
    Maxt_maxDisL = np.zeros((hang, lie))

    for i in range(0, len(PAX_data)):
        simi = []
        for j in range(0, len(PAX_data)):
            dtw01, maxdis01=DTWandMaxDist(PAX_data[i], PAX_data[j], fbs)
            Maxt_maxDisL[i][j] = maxdis01*PAX_num*1
            simi.append(dtw01)
        simisum = 0
        for h in range(0, len(simi)):
            simisum = simisum + simi[h]
        for h in range(0, len(simi)):
            MatrixPAX[i][h] = (simisum-simi[h])/simisum

    print("MatrixPAX",MatrixPAX)
    print("Maxt_maxDisL", Maxt_maxDisL)
    return MatrixPAX,Maxt_maxDisL

if __name__ == '__main__':
    # MMM = RW.RMatrix()
    # MMM = np.random.rand(5,1500)
    data1 = [[7,7,7,7,7,7,8,8,8,84,8,88,1,1,84,88,4,48,1,88],[7,7,7,7,7,7,8,8,8,84,8,88,1,1,84,88,4,48,1,88],[7,888,88,55,7,7,8,8,8,84,8,88,1,1,84,88,4,48,1,88],[7,55,88,7,7,6,8,8,8,45,8,33,1,15,84,88,4,48,17,56]]
    # print(MMM)
    PAX(hang=len(data1), lie=len(data1), data=data1, PAX_num=2)