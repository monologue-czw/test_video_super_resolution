import torch

input_tensor = torch.randn(1, 320, 64, 64)
flattened_tensor = input_tensor.flatten(start_dim=2)  # 形状为 (1, 320, 4096)
linear_layer = torch.nn.Linear(4096, 768)
projected_tensor = linear_layer(flattened_tensor)  # 形状为 (1, 320, 768)
multihead_attn = torch.nn.MultiheadAttention(embed_dim=768, num_heads=8)
attn_output, attn_output_weights = multihead_attn(projected_tensor, projected_tensor, projected_tensor)  # 形状为 (1, 320, 768)
# 使用平均池化
pooling_layer = torch.nn.AvgPool1d(kernel_size=int(320 / 77))  # kernel_size 大约为 4
pooled_tensor = pooling_layer(attn_output.permute(0, 2, 1)).permute(0, 2, 1)  # 形状为 (1, 77, 768)

# 或直接截断
truncated_tensor = attn_output[:, :77, :]  # 形状为 (1, 77, 768)