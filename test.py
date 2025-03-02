import torch
from dataloader import StockDataset
from torch.utils.data import DataLoader
from model import *
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import GetF1 as GF
import GetSMGData2 as SMD

if not os.path.exists("result_picture"):
    os.makedirs("result_picture")

if not os.path.exists("best_model"):
    os.makedirs("best_model")


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", help="which model", type=str, default="ECA")
    args = parser.parse_args()
    return args

def test():
    args = parse_args()

    test_data = StockDataset('dataset/data.csv', 5, is_test=True)
    print_step = 10

    test_loader = DataLoader(test_data, batch_size=256, shuffle=False, num_workers=2)

    model_dict = {
        "Base": CNNLSTMModel,
        "SE": CNNLSTMModel_SE,
        "ECA": CNNLSTMModel_ECA,
        "CBAM": CNNLSTMModel_CBAM,
        "HW": CNNLSTMModel_HW
    }

    model = model_dict[args.model]()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    params = torch.load(f"best_model/{args.model}_best.pth")
    model.load_state_dict(params)

    eval_loss = 0.0
    with torch.no_grad():
        y_gt = []
        y_pred = []
        for data, label in test_loader:
            y_gt += label.numpy().squeeze(axis=1).tolist()
            out = model(data)
            loss = criterion(out, label)
            eval_loss += loss.item()
            y_pred += out.numpy().squeeze(axis=1).tolist()
        print(len(y_gt), len(y_pred))

    y_gt = np.array(y_gt)
    y_gt = y_gt[:, np.newaxis]
    y_pred = np.array(y_pred)
    y_pred = y_pred[:, np.newaxis]
    print("***********",y_pred)
    # 计算评价指标
    r2 = r2_score(y_gt, y_pred)
    rmse = np.sqrt(mean_squared_error(y_gt, y_pred))
    mae = mean_absolute_error(y_gt, y_pred)
    mape = mean_absolute_percentage_error(y_gt, y_pred) * 100  # 转换为百分比

    # # 绘制图像
    # plt.figure(figsize=(12, 6))  # 设置图形的大小
    # draw = pd.concat([pd.DataFrame(y_gt), pd.DataFrame(y_pred)], axis=1)
    # draw.columns = ['Real', 'Predicted']
    # plt.plot(draw.iloc[200:500]['Real'], label='Real', color='blue', linewidth=2.0)  # 真实数据
    # plt.plot(draw.iloc[200:500]['Predicted'], label='Predicted', color='red', linestyle='--', linewidth=2.0)  # 预测数据
    # plt.legend(loc='upper right', fontsize=12, shadow=True, fancybox=True)  # 图例设置
    # plt.title(f"{args.model} Test Data Comparison", fontsize=20, fontweight='bold')  # 标题设置
    # plt.xlabel('Time', fontsize=15)  # X轴标签
    # plt.ylabel('Value', fontsize=15)  # Y轴标签
    # plt.grid(True)  # 显示网格
    # plt.savefig(f"result_picture/{args.model}_fic.jpg")  # 保存图像

    print("{}'s eval loss is {}".format(args.model, eval_loss / len(test_loader)))
    # 打印结果
    print(f"{args.model}'s eval loss is {eval_loss / len(test_loader):.4f}")
    print(f"R-squared (R2): {r2:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Mean Absolute Percentage Error (MAPE): {mape:.4f}%")


    print("GS_F1")
    ALL_F1=[]
    # for i in range(0,len(MMM)+1):
    threshold =1
    print("++++++++++++++++++++++++++++++",1)
    # Score = GS.GetScore_Matrix(MMM,threshold)
    AllFeature, all_is_anomaly = SMD.GetDATA()
    Score = all_is_anomaly
    amnourmal = all_is_anomaly
    GF.calc_point2point(np.array(Score), np.array(amnourmal))
    predict = GF.adjust_predicts(np.array(Score), np.array(amnourmal),threshold=10,pred=np.array(Score))
    f1, precision, recall=GF.calc_point2point(np.array(predict), np.array(amnourmal))
    ALL_F1.append(f1)
    print("最大的F1",max(ALL_F1))

if __name__ in '__main__':
    test()
