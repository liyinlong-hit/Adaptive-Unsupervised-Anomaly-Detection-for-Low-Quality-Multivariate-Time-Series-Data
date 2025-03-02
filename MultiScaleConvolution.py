import torch
import torch.nn as nn
import torch.nn.functional as F


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




























# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
# class MultiScaleUpsampling(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(MultiScaleUpsampling, self).__init__()
#         # 不同尺度的转置卷积层用于上采样
#         self.conv_trans1 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, padding=1, output_padding=1)
#         self.conv_trans2 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=5, padding=2, output_padding=2)
#         self.conv_trans3 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=7, padding=3, output_padding=3)
#
#     def forward(self, x):
#         # 应用不同尺度的转置卷积进行上采样
#         output1 = F.relu(self.conv_trans1(x))
#         output2 = F.relu(self.conv_trans2(x))
#         output3 = F.relu(self.conv_trans3(x))
#
#         # 将不同尺度的特征图进行融合
#         # 方法一：简单的堆叠（concatenate）
#         # output = torch.cat([output1, output2, output3], 1)
#
#         # 方法二：加权求和（weighted sum）
#         output = output1 + output2 + output3
#
#         return output
#
# # 假设编码器的输出通道数与解码器的输入通道数相同
# in_channels = 64  # 编码器的输出通道数
# out_channels = 3  # 最终输出的通道数，例如RGB图像
#
# # 实例化多尺度上采样模块
# multi_scale_upsample = MultiScaleUpsampling(in_channels, out_channels)
#
# # 使用编码器的输出作为解码器的输入
# output_tensor = multi_scale_conv(input_tensor)
#
# # 前向传播
# upsampled_output = multi_scale_upsample(output_tensor)
#
# print(f"Upsampled Output shape: {upsampled_output.shape}")