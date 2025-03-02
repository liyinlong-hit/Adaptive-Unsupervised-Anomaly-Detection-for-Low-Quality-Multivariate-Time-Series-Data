import torch
import torch.nn as nn
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# 这段代码定义了一个自注意力机制，其中num_attention_heads是注意力头的数量，
# input_size是输入的特征维度，hidden_size是隐藏层的大小，
# attention_probs_dropout_prob是注意力概率的dropout概率。代码中首先通过三个线性层计算Q、K、V，然后计算注意力分数，
# 应用softmax函数，最后通过dropout层并计算加权的值
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

        # context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        # new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        # context_layer = context_layer.view(*new_context_layer_shape)

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
        output = output1 + output2 + output3
        print(output)

        return output


# 实例化SelfAttention
num_attention_heads = 1
input_size = 512
hidden_size = 512
attention_probs_dropout_prob = 0.1

attention_layer = SelfAttention(num_attention_heads, input_size, hidden_size, attention_probs_dropout_prob)

# 创建一个随机输入张量，假设batch_size=32，seq_len=10
batch_size = 32
seq_len = 10
input_tensor = torch.randn(batch_size, seq_len, input_size)

# 通过SelfAttention层
context_layer = attention_layer(input_tensor)

# 打印输出张量的形状
print(f"context_layer Output shape: {context_layer.shape}")
# print("context_layer",context_layer)  # 应该输出torch.Size([32, 10, 512])

# 假设输入图像是三通道的
in_channels = 3
out_channels = 64  # 输出通道数

# 实例化多尺度卷积模块
multi_scale_conv = MultiScaleConvolution(in_channels, out_channels)

# 创建一个随机输入张量，模拟一个批次大小为1的图像
input_tensor = torch.randn(1, in_channels, 224, 224)

# 前向传播
output_tensor = multi_scale_conv(input_tensor)

print(f"Input shape: {input_tensor.shape}")
print(f"Output shape: {output_tensor.shape}")


encoded_tensor = output_tensor
# 获取张量的所有维度
dims = encoded_tensor.shape
# 计算所有维度的乘积
dim_product = torch.prod(torch.tensor(dims))
# 打印结果
# print(dim_product.item())  # 使用 .item() 将结果转换为 Python 的标量类型

# 假设编码器的输出特征图已经被展平
in_features = dim_product
out_in_features=128

# 实例化全连接层解码器
fully_connected_decoder = FullyConnectedDecoder(in_features, out_in_features)

# 展平张量
flattened_tensor = encoded_tensor.view(1, -1)  # 使用.view()方法展平张量
# 打印展平后的张量形状
print(flattened_tensor.shape)

# 前向传播
decoded_output = fully_connected_decoder(flattened_tensor)

print(f"Decoded Output shape: {decoded_output.shape}")
print(decoded_output)


