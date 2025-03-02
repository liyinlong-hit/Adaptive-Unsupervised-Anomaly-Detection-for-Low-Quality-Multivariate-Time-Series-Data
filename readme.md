# CNN LSTM Attention 股价预测

新增了xgboost预测的代码：![点这里](./20221205.ipynb)

- CNN + LSTM
- CNN + LSTM + ECA(attention)
- CNN + LSTM + SE(attention)
- CNN + LSTM + HW(attention)
- CNN + LSTM + CBAM(attention)

| 模型  | R-squared |RMSE  | MAE | MAPE    |
|-------|-----------|------|-------|---------|
| CNN + LSTM        | 0.9243    |0.0105 |0.0062| 0.8685% |
| CNN + LSTM + ECA  | 0.9176    |0.0109 |0.0064| 0.8929% |
| CNN + LSTM + SE   | 0.9334    |0.0098 |0.0057| 0.8070% |
| CNN + LSTM + HW   | 0.7359    |0.0196 |0.0109| 1.4956% |
| CNN + LSTM + CBAM | 0.7938    |0.0173 |0.0063| 0.8742% |

# train
```python
python train.py -m Base
python train.py -m ECA
python train.py -m SE
python train.py -m HW
python train.py -m CBAM
```
模型会存在best_model路径下

# test
```python
python test.py -m Base
python test.py -m ECA
python test.py -m SE
python test.py -m HW
python test.py -m CBAM
```
预测结果会存在result_picture下

# 预测结果

## CNN + LSTM

![CNN + LSTM](./result_picture/Base_fic.jpg)
## CNN + LSTM + ECA
![CNN + LSTM + ECA](./result_picture/ECA_fic.jpg)
## CNN + LSTM + SE
![CNN + LSTM + SE](./result_picture/SE_fic.jpg)
## CNN + LSTM + HW
![CNN + LSTM + HW](./result_picture/HW_fic.jpg)
## CNN + LSTM + CBAM

![CNN + LSTM + CBAM](./result_picture/CBAM_fic.jpg)

