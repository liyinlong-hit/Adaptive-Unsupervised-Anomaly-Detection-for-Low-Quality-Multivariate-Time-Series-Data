#

if __name__ == '__main__':

    precision = 25.68

    recall = 47.65


    f1 = 2 * precision * recall / (precision + recall)
    print("F1",f1)



# import torch
# import torch.nn as nn
# import torch.optim as optim
#
#
# # print("AllLabel",AllLabel[j*128:(j+1)*128-1])
# # indexes_of_ones = [index for index, value in enumerate(AllLabel) if value == 1]
# # print(indexes_of_ones)
#
# # 定义自编码器模型
# class Autoencoder(nn.Module):
#     def __init__(self, input_size, encoding_dim, sequence_length):
#         super(Autoencoder, self).__init__()
#         # 编码器
#         self.encoder = nn.RNN(input_size, encoding_dim, batch_first=True)
#         # 解码器
#         self.decoder = nn.RNN(encoding_dim, input_size, batch_first=True)
#
#     def forward(self, x):
#         # 编码器的输出是一个元组，包含所有时间步的输出和最后一个隐藏状态
#         encoder_outputs, encoder_hidden = self.encoder(x)
#
#         # 解码器的输入是编码器的最后一个隐藏状态，需要先将其unsqueeze(1)以匹配RNN的期望输入形状
#         decoder_input = encoder_hidden.squeeze(0).unsqueeze(1)
#
#         # 解码器的输出是所有时间步的输出
#         decoder_outputs, _ = self.decoder(decoder_input)
#
#         # 由于解码器的输出是所有时间步的，我们需要将其reshape以匹配原始输入的形状
#         # decoder_outputs = decoder_outputs.contiguous().view(-1, sequence_length, input_size)
#
#         return decoder_outputs
#
#
# # 假设输入数据的特征维度是784（例如，28x28像素的图像）
# input_size = 784
# # 编码后的维度
# encoding_dim = 128
# # 序列长度，例如，28x28像素的图像可以看作是28个时间步的序列
# sequence_length = 28
#
# # 创建自编码器模型实例
# autoencoder = Autoencoder(input_size, encoding_dim, sequence_length)
#
# # 定义损失函数和优化器
# criterion = nn.MSELoss()
# optimizer = optim.Adam(autoencoder.parameters(), lr=1e-3)
#
# # 假设我们有一些数据，这里我们使用随机数据作为示例
# # 在实际应用中，data_loader应该是一个PyTorch DataLoader实例
# # 这里我们需要将数据reshape成(batch_size, sequence_length, input_size)的形状
# data_loader = torch.randn(100, sequence_length, input_size)  # 100个样本，每个样本28个时间步，每个时间步784个特征
#
# # 训练自编码器
# num_epochs = 5
# for epoch in range(num_epochs):
#     for data in data_loader:
#         # 将输入数据传递给自编码器
#         outputs = autoencoder(data_loader)
#
#         # 计算损失
#         loss = criterion(outputs, data)
#
#         # 反向传播和优化
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#     print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
#
# # 测试自编码器
# # 假设test_data是我们要测试的数据
# test_data = torch.randn(1, sequence_length, input_size)
# reconstructed = autoencoder(test_data)
# print("Original: ", test_data)
# print("Reconstructed: ", reconstructed)
#
# import torch
# import torch.nn as nn
# import math
#
# # 这段代码定义了一个自注意力机制，其中num_attention_heads是注意力头的数量，
# # input_size是输入的特征维度，hidden_size是隐藏层的大小，
# # attention_probs_dropout_prob是注意力概率的dropout概率。代码中首先通过三个线性层计算Q、K、V，然后计算注意力分数，
# # 应用softmax函数，最后通过dropout层并计算加权的值
# class SelfAttention(nn.Module):
#     def __init__(self, num_attention_heads, input_size, hidden_size, attention_probs_dropout_prob):
#         super(SelfAttention, self).__init__()
#         if hidden_size % num_attention_heads != 0:
#             raise ValueError(
#                 "The hidden size (%d) is not a multiple of the number of attention "
#                 "heads (%d)" % (hidden_size, num_attention_heads))
#         self.num_attention_heads = num_attention_heads
#         self.attention_head_size = int(hidden_size / num_attention_heads)
#         self.all_head_size = hidden_size
#         self.query = nn.Linear(input_size, self.all_head_size)
#         self.key = nn.Linear(input_size, self.all_head_size)
#         self.value = nn.Linear(input_size, self.all_head_size)
#         self.attn_dropout = nn.Dropout(attention_probs_dropout_prob)
#
#     def transpose_for_scores(self, x):
#         new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
#         x = x.view(*new_x_shape)
#         return x.permute(0, 2, 1, 3)
#
#     def forward(self, input_tensor):
#         mixed_query_layer = self.query(input_tensor)
#         mixed_key_layer = self.key(input_tensor)
#         mixed_value_layer = self.value(input_tensor)
#
#         query_layer = self.transpose_for_scores(mixed_query_layer)
#         key_layer = self.transpose_for_scores(mixed_key_layer)
#         value_layer = self.transpose_for_scores(mixed_value_layer)
#
#         attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
#         attention_scores = attention_scores / math.sqrt(self.attention_head_size)
#
#         attention_probs = nn.Softmax(dim=-1)(attention_scores)
#         attention_probs = self.attn_dropout(attention_probs)
#
#         context_layer = torch.matmul(attention_probs, value_layer)
#
#         context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
#         new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
#         context_layer = context_layer.view(*new_context_layer_shape)
#
#         return context_layer