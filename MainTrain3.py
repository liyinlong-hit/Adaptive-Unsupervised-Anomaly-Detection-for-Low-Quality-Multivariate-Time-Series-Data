import Misplacement2 as MisPlace2
import pickle
import GetSMGData2 as SMD
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import argparse
import time
import numpy as np
import torch.nn.init as init
import math
import torch.optim as optim
from sklearn.metrics import accuracy_score

from sklearn.metrics import precision_score, recall_score, f1_score
# 潜在高斯矩阵
class LatentGaussianMixture(nn.Module):
    def __init__(self, args):
        super(LatentGaussianMixture, self).__init__()
        self.args = args


        self.mu_c = nn.Parameter(torch.rand(args.cluster_num, args.rnn_dim))

        self.log_sigma_sq_c = nn.Parameter(torch.zeros(args.cluster_num, args.rnn_dim), requires_grad=False)

        self.fc_mu_z = nn.Linear(args.rnn_dim, args.rnn_dim)
        self.fc_sigma_z = nn.Linear(args.rnn_dim, args.rnn_dim)

        # 初始化权重和偏置
        init.normal_(self.fc_mu_z.weight, std=0.02)
        init.constant_(self.fc_mu_z.bias, 0.0)
        init.normal_(self.fc_sigma_z.weight, std=0.02)
        init.constant_(self.fc_sigma_z.bias, 0.0)

    def post_sample(self, embedded_state, return_loss=False):
        args = self.args

        mu_z = self.fc_mu_z(embedded_state)
        log_sigma_sq_z = self.fc_sigma_z(embedded_state)

        eps_z = torch.randn_like(log_sigma_sq_z, dtype=torch.float32)
        latent_z = mu_z + torch.sqrt(torch.exp(log_sigma_sq_z)) * eps_z

        stack_z = latent_z.repeat(1, args.cluster_num, 1)
        # print("stack_z.shape",stack_z.shape)
        stack_mu_c = self.mu_c.repeat(1, 1)
        # print("stack_mu_c.shape",stack_mu_c.shape)
        stack_mu_z = mu_z.repeat(1, args.cluster_num, 1)
        stack_log_sigma_sq_c = self.log_sigma_sq_c.repeat(1, 1)
        stack_log_sigma_sq_z = log_sigma_sq_z.repeat(1, args.cluster_num, 1)

        pi_post_logits = - torch.sum((stack_z - stack_mu_c) ** 2 / torch.exp(stack_log_sigma_sq_c), dim=-1)
        pi_post = F.softmax(pi_post_logits, dim=-1) + 1e-10

        if not return_loss:
            return latent_z
        else:
            a = stack_log_sigma_sq_c + torch.exp(stack_log_sigma_sq_z) / torch.exp(stack_log_sigma_sq_c)
            b = (stack_mu_z - stack_mu_c) ** 2 / torch.exp(stack_log_sigma_sq_c)
            # print(a.shape)
            # print(b.shape)
            c=a+b
            d = 0.5 * torch.mean(1 + log_sigma_sq_z, dim=-1)
            # print("c.shape",c.shape)
            # print("pi_post.shape", pi_post.shape)
            # 增加 pi_post 的维度，使其形状变为 [1, 77, 1]
            pi_post_expanded = pi_post.unsqueeze(-1)
            # 扩展 pi_post_expanded 以匹配 c 的形状
            pi_post_expanded = pi_post_expanded.expand(-1, -1, c.size(-1))
            # print("d",d)
            batch_gaussian_loss = 0.5 * torch.sum(c*pi_post_expanded)-d

            batch_uniform_loss = torch.mean(torch.mean(pi_post, dim=0) * torch.log(torch.mean(pi_post, dim=0)))

            return latent_z, batch_gaussian_loss,batch_uniform_loss

    def prior_sample(self):
        pass

class MisPlace_Model(nn.Module):
    def __init__(self, args):
        super(MisPlace_Model, self).__init__()
        self.args = args

        input_size = 38
        encoding_dim =  args.x_latent_size

        # 编码器
        self.encoder = nn.RNN(input_size, encoding_dim, batch_first=True)
        # 解码器
        self.decoder = nn.RNN(encoding_dim, input_size, batch_first=True)
        # 获得高斯分布潜在空间
        # 潜在空间是能够获取了，但是现在应该调整为每一行的潜在空间，不是整个矩阵的
        self.latent_space = LatentGaussianMixture(args)

        # out_size = args.map_size[0] * args.map_size[1]
        # # 定义输出层
        # self.out_linear = nn.Linear(args.rnn_size, out_size)

    def forward(self, inputs):

        # 编码器的输出是一个元组，包含所有时间步的输出和最后一个隐藏状态
        encoder_outputs, encoder_hidden = self.encoder(inputs)

        # 获得高斯分布潜在空间
        latent_z, batch_gaussian_loss,batch_uniform_loss = self.latent_space.post_sample(encoder_hidden, return_loss=True)
        # print("Z",z)
        # print("GSlatent_losses",batch_gaussian_loss,batch_uniform_loss)

        # 解码器的输入是编码器的最后一个隐藏状态，需要先将其unsqueeze(1)以匹配RNN的期望输入形状
        decoder_input = encoder_hidden.squeeze(0).unsqueeze(1)

        # 解码器的输出是所有时间步的输出
        decoder_outputs, _ = self.decoder(decoder_input)

        return decoder_outputs,batch_gaussian_loss,batch_uniform_loss,latent_z

def MisPlace_train():
    model = MisPlace_Model(args)
    # sampler = DataGenerator(args)  # 假设DataGenerator已经被改写为PyTorch兼容的版本
    AllFeature, all_is_anomaly = SMD.GetDATA()
    df1 = pickle.load(open('machine-2-1_test.pkl', 'rb'))
    df1[:, -1] = all_is_anomaly

    sampler = df1

    all_val_loss = []

    # 定义损失函数和优化器
    criterion = nn.MSELoss()


    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    device = torch.device("cpu")

    for epoch in range(args.num_epochs):
        all_loss = []
        allbatch=int(len(sampler) / args.batch_size)
        print(allbatch)

        matrix = sampler
        # 定义要取出的子矩阵的大小,batch_size就是我们窗口大小
        submatrix_size = (args.batch_size, 38)

        # 计算可以取出多少个完整的子矩阵
        num_full_submatrices = matrix.shape[0] // submatrix_size[0]


        final_Z=[]
        # 循环依次取出每个子矩阵
        for i in range(num_full_submatrices):
            all_inputs = []
            # 计算子矩阵的起始和结束索引
            start_index = i * submatrix_size[0]
            end_index = start_index + submatrix_size[0]

            # 从矩阵中取出子矩阵
            submatrix = matrix[start_index:end_index, :]

            # 打印或处理取出的子矩阵
            # print(f"Submatrix {i + 1}:")
            # print(submatrix)
            # 将数据移动到正确的设备上
            inputs = submatrix
            # inputs = inputs.to(device)
            all_inputs.append(inputs)

            tensor_data = torch.tensor(all_inputs)
            # 清除梯度
            optimizer.zero_grad()

            # 前向传播
            outputs,batch_gaussian_loss,batch_uniform_loss,latent_z = model(tensor_data)

            # 计算损失
            loss = criterion(outputs, tensor_data)
            loss = 0.95*loss+0.05*batch_gaussian_loss
            # print("loss",loss)
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            all_val_loss.append(loss)
            final_Z = latent_z
    print("潜在空间Z训练结束，all_Z_loss",all_val_loss)
    # print("final_Z",final_Z)
    return final_Z

def calc_point2point(predict, actual):
    """
    calculate f1 score by predict and actual.

    Args:
        predict (np.ndarray): the predict label
        actual (np.ndarray): np.ndarray
    """
    TP = np.sum(predict * actual)
    TN = np.sum((1 - predict) * (1 - actual))
    FP = np.sum(predict * (1 - actual))
    FN = np.sum((1 - predict) * actual)

    if TP==0:
        precision = TP / (TP + FP + 0.00001)
        recall = TP / (TP + FN + 0.00001)
        f1 = 2 * precision * recall / (precision + recall + 0.00001)
    else:
        precision = TP / (TP + FP)
        recall = TP / (TP + FN )
        f1 = 2 * precision * recall / (precision + recall )
    FPR = FP / (FP + TN + 0.00001)
    print("f1, precision, recall",f1, precision, recall)
    # print("TP, TN, FP, FN", TP, TN, FP, FN)
    # return f1, precision, recall, FPR, TP, TN, FP, FN
    # return f1, precision, recall

def detection_adjustment(gt,pred):
    # detection adjustment
    anomaly_state = False
    for i in range(len(gt)):
        gt = gt[0]
        pred = pred[0]
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            # 从i位置向前调整找到0位置
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            # 向后找
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1

    pred = np.array(pred)
    gt = np.array(gt)
    # print("pred: ", pred)
    return pred,gt

def test_img(args):
    # testing
    test_loss = 0
    correct = 0
    AllFeature, all_is_anomaly = SMD.GetDATA()
    df1 = pickle.load(open('machine-2-1_test.pkl', 'rb'))
    df1[:, -1] = all_is_anomaly

    AllLabel = all_is_anomaly

    sampler = df1

    matrix = sampler
    # 定义要取出的子矩阵的大小,batch_size就是我们窗口大小
    submatrix_size = (args.batch_size, 38)

    # 计算可以取出多少个完整的子矩阵
    num_full_submatrices = matrix.shape[0] // submatrix_size[0]

    # 循环依次取出每个子矩阵
    all_inputs = []
    all_label=[]
    for i in range(num_full_submatrices):
        # 计算子矩阵的起始和结束索引
        start_index = i * submatrix_size[0]
        end_index = start_index + submatrix_size[0]

        # 从矩阵中取出子矩阵
        submatrix = matrix[start_index:end_index, :]
        sublabel = all_is_anomaly[start_index:end_index]
        # 将列表转换为张量
        tensor_from_list = torch.tensor(sublabel)
        # 使用 unsqueeze 在第一个维度添加一个维度
        sublabel = tensor_from_list.unsqueeze(0)

        inputs = submatrix
        all_inputs.append(inputs)
        all_label.append(sublabel)


    test_attention_layer = torch.load('attention_layer.pth')
    test_multi_scale_conv = torch.load('multi_scale_conv.pth')
    test_fully_connected_decoder = torch.load('fully_connected_decoder.pth')
    test_fully_connected_decoder2 = torch.load('fully_connected_decoder2.pth')
    print("模型加载")

    # for j in range(len(all_inputs)):
    for j in range(62, 70):
        print("测试j::", j)
        a_inputs = [all_inputs[j]]
        sublabel = all_label[j]
        tensor_data = torch.tensor(a_inputs)
        optimizer.zero_grad()
        # 通过SelfAttention层
        Att_X = test_attention_layer(tensor_data)
        # 使用 torch.unsqueeze 在第二维插入一个维度
        inputAA_reshaped = torch.unsqueeze(Att_X, 1)
        inputAA_reshaped = inputAA_reshaped.repeat(1, 3, 1, 1)

        output_tensor = test_multi_scale_conv(inputAA_reshaped)

        encoded_tensor = output_tensor

        # 展平张量
        flattened_tensor = encoded_tensor.view(1, -1)  # 使用.view()方法展平张量

        # 前向传播解码
        decoded_output = test_fully_connected_decoder(flattened_tensor)

        ZandZ_decoded_output = test_fully_connected_decoder2(decoded_output)

        # 输出和标签对齐
        sublabel = sublabel.float()
        # 计算准确率
        # x = torch.sigmoid(ZandZ_decoded_output)
        # # 使用阈值0.9将输出映射为0或1
        x = (ZandZ_decoded_output >= 0.9).float()  # 或者使用x.round()，但round会将0.9四舍五入为1
        print("x",x)
        # x = ZandZ_decoded_output

        # 重写
        # detection_adjustment(sublabel.detach().numpy(), x.detach().numpy())
        # 使用torch.eq比较两个张量的对应元素是否相等
        equal_elements = torch.eq(sublabel, x)
        # 计算相等元素的个数
        num_equal = torch.sum(equal_elements).item()
        print("num_equal",num_equal)
        accuracy = num_equal/args.batch_size
        print("return accuracy", accuracy)
        # calc_point2point(sublabel.detach().numpy(), x.detach().numpy())
        pred,gt1 = detection_adjustment(sublabel.tolist(), x.tolist())
        print("sublabel",sublabel)
        print("x",x)
        print("pred",pred)
        # calc_point2point(pred, gt1)

        # 计算精确度
        precision = precision_score(gt1, pred)
        print(f"Precision: {precision}")

        # 计算召回率
        recall = recall_score(gt1, pred)
        print(f"Recall: {recall}")

        # 计算F1分数
        f1 = f1_score(gt1, pred)
        print(f"F1 Score: {f1}")


    return accuracy





class SelfAttention(nn.Module):
    def __init__(self, num_attention_heads, input_size, hidden_size, attention_probs_dropout_prob):
        super(SelfAttention, self).__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_attention_heads))
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = hidden_size
        self.query = nn.Linear(input_size, self.all_head_size)
        self.key = nn.Linear(input_size, self.all_head_size)
        self.value = nn.Linear(input_size, self.all_head_size)
        self.attn_dropout = nn.Dropout(attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, input_tensor):
        # QKV都是输入向量input_tensor
        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        return context_layer

class FullyConnectedDecoder(nn.Module):
    def __init__(self, in_features, out_in_features):
        super(FullyConnectedDecoder, self).__init__()
        # 假设输入特征图已经被展平为一维向量
        self.fc1 = nn.Linear(in_features, 128)  # 第一个全连接层
        self.fc2 = nn.Linear(128, 64)  # 第二个全连接层
        self.fc3 = nn.Linear(64, out_in_features)  # 最后一个全连接层

    def forward(self, x):
        # 假设x是编码器的输出，形状为(batch_size, in_features)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # 将输出重新映射回二维空间，形状为(batch_size, out_height, out_width)
        # x = x.view(-1, out_height, out_width)
        return x

class FullyConnectedDecoder2(nn.Module):
    def __init__(self, in_features, out_in_features):
        super(FullyConnectedDecoder2, self).__init__()
        # 假设输入特征图已经被展平为一维向量
        self.fc1 = nn.Linear(in_features, 128)  # 第一个全连接层
        self.fc2 = nn.Linear(128, 64)  # 第二个全连接层
        self.fc3 = nn.Linear(64, out_in_features)  # 最后一个全连接层

    def forward(self, x):
        # 假设x是编码器的输出，形状为(batch_size, in_features)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = torch.sigmoid(x)
        # 将输出重新映射回二维空间，形状为(batch_size, out_height, out_width)
        # x = x.view(-1, out_height, out_width)
        return x
#
class MultiScaleConvolution(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MultiScaleConvolution, self).__init__()
        # 不同尺度的卷积层
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=7, padding=3)

    def forward(self, x):
        # 应用不同尺度的卷积
        output1 = F.relu(self.conv1(x))
        output2 = F.relu(self.conv2(x))
        output3 = F.relu(self.conv3(x))

        # 将不同尺度的特征图进行融合
        # 方法一：简单的堆叠（concatenate）
        # output = torch.cat([output1, output2, output3], 1)

        # 方法二：加权求和（weighted sum）
        output = (output1 + output2 + output3)/3
        # print(output)

        return output

if __name__ == '__main__':
    # 错位部分++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # region
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_filename', type=str, default="../data/processed_beijing{}.csv",
                        help='data file')
    parser.add_argument('--map_size', type=tuple, default=(130, 130),
                        help='size of map')
    parser.add_argument('--model_type', type=str, default="",
                        help='choose a model')

    parser.add_argument('--mem_num', type=int, default=5,
                        help='size of sd memory')

    parser.add_argument('--neg_size', type=int, default=64,
                        help='size of negative sampling')
    parser.add_argument('--num_epochs', type=int, default=1,
                        help='number of epochs')
    parser.add_argument('--grad_clip', type=float, default=10.,
                        help='clip gradients at this value')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--decay_rate', type=float, default=1.,
                        help='decay of learning rate')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='minibatch size')

    parser.add_argument('--model_id', type=str, default="",
                        help='model id')
    parser.add_argument('--partial_ratio', type=float, default=1.0,
                        help='partial trajectory evaluation')
    parser.add_argument('--eval', type=bool, default=False,
                        help='partial trajectory evaluation')
    parser.add_argument('--pt', type=bool, default=False,
                        help='partial trajectory evaluation')

    parser.add_argument('--gpu_id', type=str, default="0")

    # 目前的代码只有x_latent_size，rnn_dim相同维度才能运行，好像潜在空间越小，收敛越好
    parser.add_argument('--x_latent_size', type=int, default=128,
                        help='size of input embedding')
    parser.add_argument('--cluster_num', type=int, default=30,
                        help='cluster_num')
    parser.add_argument('--rnn_dim', type=int, default=128,
                        help='rnn_dim')
    # endregion
    args = parser.parse_args()


    AllFeature, all_is_anomaly = SMD.GetDATA()
    df1 = pickle.load(open('machine-2-1_test.pkl', 'rb'))
    df1[:, -1] = all_is_anomaly

    MisPlace_latent_z = MisPlace_train()

    AllFeature, all_is_anomaly = SMD.GetDATA()
    df1 = pickle.load(open('machine-2-1_test.pkl', 'rb'))
    df1[:, -1] = all_is_anomaly

    sampler = df1
    AllLabel = all_is_anomaly
    # 所以loss
    all_val_loss = []
    # 定义模型
    # 实例化SelfAttention
    num_attention_heads = 1
    input_size = 38
    hidden_size = 128
    attention_probs_dropout_prob = 0.1

    attention_layer = SelfAttention(num_attention_heads, input_size, hidden_size, attention_probs_dropout_prob)

    # 假设输入图像是三通道的
    # 实例化MultiScaleConvolution
    in_channels = 3
    out_channels = 64  # 输出通道数
    # 实例化多尺度卷积模块
    multi_scale_conv = MultiScaleConvolution(in_channels, out_channels)
    # 实例化全连接层解码器
    # 假设编码器的输出特征图已经被展平
    in_features1 = out_channels*hidden_size*hidden_size
    out_features1 = 128
    fully_connected_decoder = FullyConnectedDecoder(in_features1, out_features1)

    # 实例化全连接层解码器2
    in_features2 = out_features1
    out_features2 = 128
    fully_connected_decoder2 = FullyConnectedDecoder2(in_features2, out_features2)


    all_loss = []
    allbatch = int(len(sampler) / args.batch_size)
    print(allbatch)

    matrix = sampler
    # 定义要取出的子矩阵的大小,batch_size就是我们窗口大小
    submatrix_size = (args.batch_size, 38)

    # 计算可以取出多少个完整的子矩阵
    num_full_submatrices = matrix.shape[0] // submatrix_size[0]

    final_Z = MisPlace_train()
    all_inputs = []
    # 循环依次取出每个子矩阵
    for i in range(num_full_submatrices):
        # print("i:",i)
        # 计算子矩阵的起始和结束索引
        start_index = i * submatrix_size[0]
        end_index = start_index + submatrix_size[0]

        # 从矩阵中取出子矩阵
        submatrix = matrix[start_index:end_index, :]
        sublabel = all_is_anomaly[start_index:end_index]
        # 将列表转换为张量
        tensor_from_list = torch.tensor(sublabel)
        # 使用 unsqueeze 在第一个维度添加一个维度
        sublabel = tensor_from_list.unsqueeze(0)
        # 打印或处理取出的子矩阵
        # 将数据移动到正确的设备上
        inputs = submatrix

        all_inputs.append(inputs)



    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(list(attention_layer.parameters()) + list(multi_scale_conv.parameters())
                           + list(fully_connected_decoder.parameters())+ list(fully_connected_decoder2.parameters()), lr=0.001)

    # 训练模型
    epochs = 1
    for epoch in range(epochs):
        # for j in range(len(all_inputs)):
        for j in range(60,70):
            a_inputs = [all_inputs[j]]
            tensor_data = torch.tensor(a_inputs)
            optimizer.zero_grad()
            # 通过SelfAttention层
            Att_X = attention_layer(tensor_data)
            # 使用 torch.unsqueeze 在第二维插入一个维度
            inputAA_reshaped = torch.unsqueeze(Att_X, 1)
            inputAA_reshaped = inputAA_reshaped.repeat(1, 3, 1, 1)
            # flattened_output = output.view(1, -1)
            # decoded_output = fully_connected_decoder(flattened_output)
            output_tensor = multi_scale_conv(inputAA_reshaped)

            encoded_tensor = output_tensor

            # 展平张量
            flattened_tensor = encoded_tensor.view(1, -1)  # 使用.view()方法展平张量

            # 前向传播解码
            decoded_output = fully_connected_decoder(flattened_tensor)

            final_Z = final_Z.squeeze().unsqueeze(0)
            decoded_output = decoded_output + final_Z

            ZandZ_decoded_output = fully_connected_decoder2(decoded_output)

            # 输出和标签对齐
            sublabel = sublabel.float()

            # 防止梯度小时，创造一个初始梯度
            predict = ZandZ_decoded_output.detach().numpy()
            # predict = predict[0]
            actual = sublabel.detach().numpy()
            # actual = actual[0]

            FP = np.sum(predict * (1 - actual))
            FN = np.sum((1 - predict) * actual)
            Ftensor = torch.tensor(0.01*(FP+FN), requires_grad=True)

            faketensor = torch.tensor(0.0000120, requires_grad=True)
            loss = criterion(ZandZ_decoded_output, sublabel)+faketensor+Ftensor
            loss.backward(retain_graph=True)
            optimizer.step()
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')
            torch.save(multi_scale_conv, 'multi_scale_conv.pth')
            torch.save(attention_layer, 'attention_layer.pth')
            torch.save(fully_connected_decoder, 'fully_connected_decoder.pth')
            torch.save(fully_connected_decoder2, 'fully_connected_decoder2.pth')
            test_img(args)
        # model = torch.load('multi_scale_conv.pth')
        # print("模型加载")