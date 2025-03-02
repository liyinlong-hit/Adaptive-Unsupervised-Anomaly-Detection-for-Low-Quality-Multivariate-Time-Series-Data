import numpy as np
from sklearn.preprocessing import MinMaxScaler
import ReadMatrix as RW

def GetScore_Matrix(Result_Matrix,threshold):
    Score=[]
    threshold
    #行
    row = len(Result_Matrix)
    # 列
    Len = len(Result_Matrix[0])
    # print("Result_Matrix",Result_Matrix)
    # print(row)
    for i  in range(0,Len):
        Sum=0
        for j in range(0,row):
            Sum = Sum+Result_Matrix[j][i]
        Score.append(Sum)

    for i in range(0,len(Score)):
        if Score[i]<threshold:
            Score[i]=0
        else:
            Score[i] = 1
    # print("Score",Score)
    return Score


if __name__ == '__main__':
    AA = RW.RMatrix()
    threshold=6
    GetScore_Matrix(AA,threshold)