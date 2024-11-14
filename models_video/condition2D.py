from diffusers import UNet2DModel
from diffusers.models.unet_2d import UNet2DOutput
from torch import nn
from typing import Union, Optional, Tuple
import torch


class reshapetocondition(nn.Module):       #B*T,C,H,W-->B,C*T,H*W
    def __init__(self,C,H,W):
        super(reshapetocondition,self).__init__()
        self.C = C
        self.H = H
        self.W = W
        self.conv = nn.Conv2d(in_channels=self.C,out_channels=77,kernel_size=1,padding=0,stride=1)
        self.linear = nn.Linear(in_features=self.H*self.W,out_features=1024)

    def forward(self,input):
        B = input.size(0)
        out = self.conv(input)
        out = out.view(B,77,-1)
        out = self.linear(out)

        return out


class UNet2DConditionModel(UNet2DModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.condition = []      #这里是5，77，1024--》1，77，1024（B*T）
        self.condition_output = []

        self.reshape1 = reshapetocondition(256, 32, 32)  # 指的是输入的T,有bug，要手动输入
        self.reshape2 = reshapetocondition(512, 16, 16)
        self.reshape3 = reshapetocondition(512, 8, 8)
        self.reshape4 = reshapetocondition(1024, 8, 8)
        self.reshape5 = reshapetocondition(1024, 16, 16)
        self.reshape6 = reshapetocondition(512, 32, 32)
        self.reshape7 = reshapetocondition(512, 64, 64)
        self.reshape8 = reshapetocondition(256, 64, 64)
        self.reshape_layers = [self.reshape1, self.reshape2, self.reshape3, self.reshape4, self.reshape4,
                               self.reshape5, self.reshape6, self.reshape7, self.reshape8]


    def reshape(self,input):
        B,T,C = input.size()
        input = input.view(B,C,T)
        linear = nn.Linear(in_features=T,out_features=77)
        out = linear(input)
        out = out.permute(0,2,1)
        return out

    def forward(
            self,
            sample: torch.Tensor,
            timestep: Union[torch.Tensor, float, int],
            class_labels: Optional[torch.Tensor] = None,
            return_dict: bool = True,
    ) -> Union[UNet2DOutput, Tuple]:

        B,C,T,H,W = sample.size()
        sample = sample.view(B*T,C,H,W)

        #print(sample.shape)

        # 0. center input if necessary
        if self.config.center_input_sample:
            sample = 2 * sample - 1.0

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps * torch.ones(sample.shape[0], dtype=timesteps.dtype, device=timesteps.device)

        t_emb = self.time_proj(timesteps)

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=self.dtype)
        emb = self.time_embedding(t_emb)

        if self.class_embedding is not None:
            if class_labels is None:
                raise ValueError("class_labels should be provided when doing class conditioning")

            if self.config.class_embed_type == "timestep":
                class_labels = self.time_proj(class_labels)

            class_emb = self.class_embedding(class_labels).to(dtype=self.dtype)
            emb = emb + class_emb
        elif self.class_embedding is None and class_labels is not None:
            raise ValueError("class_embedding needs to be initialized in order to use class conditioning")

        # 2. pre-process
        skip_sample = sample
        sample = self.conv_in(sample)

        # 3. down
        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if hasattr(downsample_block, "skip_conv"):
                sample, res_samples, skip_sample = downsample_block(
                    hidden_states=sample, temb=emb, skip_sample=skip_sample
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)

            down_block_res_samples += res_samples


            self.condition.append(sample)
            #print("Shape after downsample block:", sample.shape)

        # 4. mid
        sample = self.mid_block(sample, emb)

        self.condition.append(sample)
        #print("Shape after mid block:", sample.shape)

        # 5. up
        skip_sample = None
        for upsample_block in self.up_blocks:
            res_samples = down_block_res_samples[-len(upsample_block.resnets):]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

            if hasattr(upsample_block, "skip_conv"):
                sample, skip_sample = upsample_block(sample, res_samples, emb, skip_sample)
            else:
                sample = upsample_block(sample, res_samples, emb)

            self.condition.append(sample)
            #print("Shape after upsample block:", sample.shape)

        # 6. post-process
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)


        if skip_sample is not None:
            sample += skip_sample

        if self.config.time_embedding_type == "fourier":
            timesteps = timesteps.reshape((sample.shape[0], *([1] * len(sample.shape[1:]))))
            sample = sample / timesteps

        if not return_dict:
            return (sample,)

        for i, tensor in enumerate(self.condition):
            reshaped_tensor = self.reshape_layers[i](tensor)  # 应用对应的 reshapetocondition 实例
            reshaped_tensor = reshaped_tensor.view(B, 77*T, 1024)
            reshaped_tensor = self.reshape(reshaped_tensor)
            self.condition_output.append(reshaped_tensor)

        return UNet2DOutput(sample=sample),self.condition_output




'''
input = torch.randn(1,3,5,64,64)
timestep = torch.tensor([100])

mymodel = UNet2DConditionModel(block_out_channels=(256, 512, 512, 1024))
output,condition = mymodel(sample=input,timestep = torch.tensor([100]))
print(output.sample.shape)
print(condition[0].shape,condition[1].shape,condition[2].shape,condition[3].shape,condition[4].shape,
      condition[5].shape,condition[6].shape,condition[7].shape,condition[8].shape)
'''
