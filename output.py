import os
from io import BytesIO
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
from PIL import Image
from datasets import load_dataset
from diffusers import DDIMScheduler, DiffusionPipeline
from numpy import mean
from torch.utils.data import Dataset,DataLoader
from transformers import pipeline
import json
import torch
import torch.nn as nn
import torch.utils.checkpoint
from models_video import UNetVideoModel, AutoencoderKLVideo,UNet2DConditionModel,UNet3DConditionModel
import decord
import requests
from models_video.pipeline_upscale_a_video import VideoUpscalePipeline

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


class all_model(nn.Module):
    def __init__(self):
        super(all_model, self).__init__()

        unet_config_path = "C:\\Users\\34936\\Desktop\\Upscale-A-Video-master\\pretrained_models\\upscale_a_video\\unet\\unet_video_config.json"
        pretrained_model_unet = "C:\\Users\\34936\\Desktop\\Upscale-A-Video-master\\pretrained_models\\upscale_a_video\\unet\\unet_video.bin"
        self.unet = UNetVideoModel.from_config(unet_config_path)
        self.unet.load_state_dict(torch.load(pretrained_model_unet, map_location="cpu"), strict=False)
        self.unet = self.unet.float()
        self.unet.eval()

        for name, param in self.unet_parameters():
            if name.startswith("conv_change"):  # 只训练conv_change层
                param.requires_grad = True
            else:
                param.requires_grad = False

        # 直接实例化VAE模型并加载预训练权重
        vae_config_path = "C:\\Users\\34936\\Desktop\\Upscale-A-Video-master\\configs\\vae_video_config.json"
        pretrained_model_vae = "C:\\Users\\34936\\Desktop\\Upscale-A-Video-master\\pretrained_models\\upscale_a_video\\vae\\vae_video.bin"
        self.vae = AutoencoderKLVideo.from_config(vae_config_path)
        self.vae.load_state_dict(torch.load(pretrained_model_vae, map_location="cpu"), strict=True)
        self.vae = self.vae.float()
        self.vae.eval()

        self.condition_2D = UNet2DConditionModel(block_out_channels=(256, 512, 512, 1024))
        self.condition_3D = UNet3DConditionModel()

        self.conv_in = nn.Conv3d(in_channels=3, out_channels=4, kernel_size=1)
        self.conv_out = nn.Conv3d(in_channels=4, out_channels=3, kernel_size=1)

    def unet_parameters(self):
        return self.unet.named_parameters()

    def forward(self,
                input,
                timestep,
                low_res,
                condition_img,
                ):
        encoder_output = self.vae.encode(input, return_dict=False)[0]

        # 从encoder_output中获取潜在空间的样本或模式
        z = encoder_output.sample() if not encoder_output.deterministic else encoder_output.mean
        print("after encoder", z.shape)

        # 条件输入
        output_2D, condition_2D = self.condition_2D(sample=low_res, timestep=timestep)
        low_res_3D = self.conv_in(low_res)
        output_3D, condition_3D = self.condition_3D(sample=low_res_3D, timestep=timestep)
        conditions = [condition_2D, condition_3D]
        output_3D = self.conv_out(output_3D)
        print("condition", conditions[0][0].shape, conditions[1][8].shape)

        unet_output = self.unet(sample=z, timestep=timestep, low_res=low_res,
                                encoder_hidden_states=conditions)
        print("after unet", unet_output.sample.shape)

        decoder_input = unet_output.sample
        final_output = self.vae.decode(decoder_input, img=condition_img, return_dict=False)[0]

        return final_output,output_2D,output_3D


class UDM10Dataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.video_folders = sorted(os.listdir(root_dir))  # 获取视频文件夹列表

    def __len__(self):
        return len(self.video_folders)

    def __getitem__(self, idx):
        video_folder = os.path.join(self.root_dir, f'{idx:03d}')  # 使用 f-string 格式化文件夹名称
        frames = []
        for i in range(31):  # 每个视频有 31 帧
            frame_path = os.path.join(video_folder, f'{i:04d}.png')  # 假设图片命名格式为 0000.png, 0001.png, ...
            frame = Image.open(frame_path).convert('RGB')
            if self.transform:
                frame = self.transform(frame)
            frames.append(frame)
        video = torch.stack(frames)  # 将帧堆叠成 (T, C, H, W) 格式
        video = video.permute(1, 0, 2, 3)  # 转换为 (C, T, H, W) 格式
        return video

# 调用 all_model 模型
model = all_model()
model.to(device)

trainable_params = [
    param for name, param in model.named_parameters()  # 使用 named_parameters() 获取参数名称
    if param.requires_grad and (name.startswith("unet.conv_change")  # 对参数名称使用 startswith
                                or name.startswith("UNet2DConditionModel")
                                or name.startswith("UNet3DConditionModel")
                                or name.startswith("conv_in")
                                or name.startswith("conv_out")
                                )
]

optimizer = torch.optim.Adam(trainable_params, lr=1e-4)

noised_scheduler = DDIMScheduler.from_config("C:\\Users\\34936\\Desktop\\Upscale-A-Video-master\\pretrained_models\\upscale_a_video\\scheduler\\scheduler_config.json")


batch_size = 1  # 或其他您想要的大小
epoch =2


# 定义图像变换
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # 归一化
])

# 创建数据集和数据加载器
dataset_GT = UDM10Dataset(root_dir="C:\\Users\\34936\\Desktop\\UDM10\\data\\UDM10\\GT", transform=transform)  # 替换为您的 UDM10 数据集路径
dataloader_GT = DataLoader(dataset_GT, batch_size=batch_size, shuffle=True)  # 设置批次大小和 shuffle

dataset_BI = UDM10Dataset(root_dir="C:\\Users\\34936\\Desktop\\UDM10\\data\\UDM10\\BIx4", transform=transform)  # 替换为您的 UDM10 数据集路径
dataloader_BI = DataLoader(dataset_BI, batch_size=batch_size, shuffle=True)  # 设置批次大小和 shuffle

condition_img_tensor = None

for i in range(epoch):
    for batch_GT, batch_BI in zip(dataloader_GT, dataloader_BI):
        GT_videos = batch_GT  # 获取 GT 视频数据
        BI_videos = batch_BI  # 获取 BI 视频数据

        GT_videos=  GT_videos.to(device)
        BI_videos = BI_videos.to(device)


        timestep = noised_scheduler.timesteps[0].to(device)  # 获取第一个时间步长

        # 生成噪声
        noise = torch.randn_like(GT_videos).to(device)

        # 将噪声添加到数据中
        noisy_videos = noised_scheduler.add_noise(GT_videos, noise, timestep)

        # 使用模型去噪
        pred_noise, output2D, output3D = model(noisy_videos, timestep, BI_videos, condition_img_tensor)

        # 计算损失
        loss = mean((pred_noise - noise) ** 2) + mean((output2D - BI_videos) ** 2) + mean(
            (output3D - BI_videos) ** 2)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        print(loss)














'''
# 示例输入帧和条件图像
input = torch.randn(1, 3, 5, 64, 64)  # 示例输入B,C,T,H,W,就是HR
timestep = torch.tensor([5])
low_res = torch.randn(1,3,5,16,16)   #4倍下采样后的数据
encoder_hidden_states=torch.randn(1,77,1024)
condition_img = torch.randn(1,3,5,16,16)  # 示例条件图像B,C,H,W

# 进行前向传播
output = model(input,timestep,low_res,condition_img)

# 打印输出的形状
print(output.shape)


# 模型转换为半精度并设置为评估模式
pipeline.unet = pipeline.unet.float()
pipeline.unet.eval()
input_frames = torch.randn(1, 4, 5, 64, 64)  # 示例输入B,C,T,H,W
timestep = torch.tensor([5])
low_res = torch.randn(1,3,5,64,64)
encoder_hidden_states=torch.randn(1,77,1024)
# 使用模型进行推理
with torch.no_grad():
    upscaled_frames = pipeline.unet(sample=input_frames,timestep=timestep,low_res=low_res,encoder_hidden_states=encoder_hidden_states)

# 处理输出
print(upscaled_frames.sample.shape)

'''