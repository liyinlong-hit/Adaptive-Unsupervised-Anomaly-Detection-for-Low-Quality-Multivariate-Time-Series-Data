import torch
from torch.utils.data import Dataset
import pandas as pd
from sklearn import preprocessing
import numpy as np
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
        z = mu_z + torch.sqrt(torch.exp(log_sigma_sq_z)) * eps_z

        stack_z = z.repeat(1, args.cluster_num, 1)
        print("stack_z.shape",stack_z.shape)
        stack_mu_c = self.mu_c.repeat(1, 1)
        print("stack_mu_c.shape",stack_mu_c.shape)
        stack_mu_z = mu_z.repeat(1, args.cluster_num, 1)
        stack_log_sigma_sq_c = self.log_sigma_sq_c.repeat(1, 1)
        stack_log_sigma_sq_z = log_sigma_sq_z.repeat(1, args.cluster_num, 1)

        pi_post_logits = - torch.sum((stack_z - stack_mu_c) ** 2 / torch.exp(stack_log_sigma_sq_c), dim=-1)
        pi_post = F.softmax(pi_post_logits, dim=-1) + 1e-10

        if not return_loss:
            return z
        else:
            a = stack_log_sigma_sq_c + torch.exp(stack_log_sigma_sq_z) / torch.exp(stack_log_sigma_sq_c)
            b = (stack_mu_z - stack_mu_c) ** 2 / torch.exp(stack_log_sigma_sq_c)
            # b = b.repeat(1, 1)
            print(a.shape)
            print(b.shape)
            c=a+b
            d = 0.5 * torch.mean(1 + log_sigma_sq_z, dim=-1)
            print("c.shape",c.shape)
            # f = torch.squeeze(c)
            # print("f.shape",f.shape)
            print("pi_post.shape", pi_post.shape)
            # print("d",d)
            batch_gaussian_loss = 0.5 * torch.sum(pi_post * (c))

            batch_uniform_loss = torch.mean(torch.mean(pi_post, dim=0) * torch.log(torch.mean(pi_post, dim=0)))
            return z, [batch_gaussian_loss, batch_uniform_loss]

    def prior_sample(self):
        pass
# 潜在高斯矩阵


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args

        input_size = 38
        encoding_dim =  args.x_latent_size

        # 编码器
        self.encoder = nn.RNN(input_size, encoding_dim, batch_first=True)
        # 解码器
        self.decoder = nn.RNN(encoding_dim, input_size, batch_first=True)

        self.latent_space = LatentGaussianMixture(args)

        # 定义聚类相关的参数
        self.mu_c = nn.Parameter(torch.randn(args.mem_num, args.rnn_size))
        self.log_sigma_sq_c = nn.Parameter(torch.zeros(args.mem_num, args.rnn_size), requires_grad=False)
        self.log_pi_prior = nn.Parameter(torch.zeros(args.mem_num), requires_grad=False)
        self.pi_prior = F.softmax(self.log_pi_prior, dim=-1)

        # 定义生成mu_z和sigma_z的线性层,rnn_size=256
        self.mu_z_linear = nn.Linear(args.rnn_size, args.rnn_size)
        self.sigma_z_linear = nn.Linear(args.rnn_size, args.rnn_size)

        # Clusters
        self.mu_c = nn.Parameter(
            torch.rand(args.mem_num, args.rnn_size) * 2 - 1)  # Random uniform init between -1 and 1
        self.log_sigma_sq_c = nn.Parameter(torch.zeros(args.mem_num, args.rnn_size))  # Initialized to 0
        self.log_pi_prior = nn.Parameter(torch.zeros(args.mem_num))  # Initialized to 0
        self.pi_prior = F.softmax(self.log_pi_prior, dim=0)
        #
        # # Placeholders in PyTorch are replaced with functions that create tensors
        # self.init_mu_c = lambda x: self.mu_c.data.copy_(x)
        # self.init_sigma_c = lambda x: self.log_sigma_sq_c.data.copy_(x)
        # self.init_pi = lambda x: self.log_pi_prior.data.copy_(x)
        #
        # # Latent
        # self.mu_z_w = nn.Parameter(torch.randn(args.rnn_size, args.rnn_size) * 0.02)
        # self.mu_z_b = nn.Parameter(torch.zeros(args.rnn_size))
        # self.sigma_z_w = nn.Parameter(torch.randn(args.rnn_size, args.rnn_size) * 0.02)
        # self.sigma_z_b = nn.Parameter(torch.zeros(args.rnn_size))






        out_size = args.map_size[0] * args.map_size[1]
        # 定义输出层
        self.out_linear = nn.Linear(args.rnn_size, out_size)

    def forward(self, inputs):

        # 编码器的输出是一个元组，包含所有时间步的输出和最后一个隐藏状态
        encoder_outputs, encoder_hidden = self.encoder(inputs)

        z, latent_losses = self.latent_space.post_sample(encoder_hidden, return_loss=True)


        # # 计算mu_z和sigma_z
        # mu_z = self.mu_z_linear(encoder_hidden.squeeze(0))
        # log_sigma_sq_z = self.sigma_z_linear(encoder_hidden.squeeze(0))
        #
        # # 采样z
        # eps_z = torch.randn_like(log_sigma_sq_z)
        # z = mu_z + torch.sqrt(torch.exp(log_sigma_sq_z)) * eps_z
        #
        # # 注意力机制
        # att_logits = - torch.sum((z.unsqueeze(1) - self.mu_c) ** 2 / torch.exp(self.log_sigma_sq_c), dim=-1)
        # att = F.softmax(att_logits, dim=-1) + 1e-10


        # 解码器的输入是编码器的最后一个隐藏状态，需要先将其unsqueeze(1)以匹配RNN的期望输入形状
        decoder_input = encoder_hidden.squeeze(0).unsqueeze(1)

        # 解码器的输出是所有时间步的输出
        decoder_outputs, _ = self.decoder(decoder_input)

        # Latent
        # mu_z = torch.matmul(encoder_hidden, self.mu_z_w) + self.mu_z_b
        # log_sigma_sq_z = torch.matmul(encoder_hidden, self.sigma_z_w) + self.sigma_z_b
        #
        # eps_z = torch.randn_like(log_sigma_sq_z, device=encoder_hidden.device)
        # z = mu_z + torch.sqrt(torch.exp(log_sigma_sq_z)) * eps_z
        # print("ZZZ：",z)


        return decoder_outputs


def train():
    model = Model(args)
    # sampler = DataGenerator(args)  # 假设DataGenerator已经被改写为PyTorch兼容的版本
    AllFeature, all_is_anomaly = SMD.GetDATA()
    df1 = pickle.load(open('machine-2-1_test.pkl', 'rb'))
    df1[:, -1] = all_is_anomaly

    sampler = df1

    all_val_loss = []
    start = time.time()

    # 定义损失函数和优化器
    criterion = nn.MSELoss()

    # 加载预训练模型
    # model_name = f"./models/{args.model_type}_{args.x_latent_size}_{args.rnn_size}/{args.model_type}_pretrain"
    # model.restore(model_name)  # 假设restore函数已经被改写为PyTorch兼容的版本

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    device = torch.device("cpu")

    for epoch in range(args.num_epochs):
        all_loss = []
        allbatch=int(len(sampler) / args.batch_size)
        print(allbatch)

        matrix = sampler
        # 定义要取出的子矩阵的大小
        submatrix_size = (args.batch_size, 38)

        # 计算可以取出多少个完整的子矩阵
        num_full_submatrices = matrix.shape[0] // submatrix_size[0]


        all_inputs=[]
        # 循环依次取出每个子矩阵
        for i in range(num_full_submatrices):
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
        outputs = model(tensor_data)

        # 计算损失
        loss = criterion(outputs, tensor_data)
        print("loss",loss)
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def compute_loss(model, sampler):
    all_loss = []
    device = torch.device("cpu")

    # 假设sampler.iterate_all_data是一个生成器，它产生批量数据
    for batch_data in sampler.iterate_all_data(args.batch_size, partial_ratio=args.partial_ratio):
        # 将数据移动到正确的设备上
        inputs, mask, seq_length = [data.to(device) for data in batch_data]

        # 清除梯度
        model.zero_grad()

        # 前向传播
        loss = model(inputs, mask, seq_length)

        # 反向传播
        loss.backward()

        # 将损失值转换为标量并累积
        all_loss.append(loss.item())

    # 计算平均损失
    return np.mean(all_loss)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--data_filename', type=str, default="../data/processed_beijing{}.csv",
                        help='data file')
    parser.add_argument('--map_size', type=tuple, default=(130, 130),
                        help='size of map')
    # parser.add_argument('--data_filename', type=str, default="../data/processed_porto{}.csv",
    #                     help='data file')
    # parser.add_argument('--map_size', type=tuple, default=(51, 158),
    #                     help='size of map')
    parser.add_argument('--model_type', type=str, default="",
                        help='choose a model')

    parser.add_argument('--mem_num', type=int, default=5,
                        help='size of sd memory')

    parser.add_argument('--neg_size', type=int, default=64,
                        help='size of negative sampling')
    parser.add_argument('--num_epochs', type=int, default=10,
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

    parser.add_argument('--rnn_size', type=int, default=30,
                        help='size of RNN hidden state')
    parser.add_argument('--x_latent_size', type=int, default=30,
                        help='size of input embedding')
    parser.add_argument('--cluster_num', type=int, default=30,
                        help='cluster_num')
    parser.add_argument('--rnn_dim', type=int, default=30,
                        help='rnn_dim')


    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    AllFeature, all_is_anomaly = SMD.GetDATA()
    df1 = pickle.load(open('machine-2-1_test.pkl', 'rb'))
    df1[:, -1] = all_is_anomaly
    # mu_c = [0.5, 0.6]  # 示例值
    # model = Model(mu_c)
    # print(model)
    train()

