import torch
import functools
from torch.utils.data import Dataset, DataLoader
from torch import nn
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pytorch_ssim
import math

torch.cuda.empty_cache()
print(torch.cuda.is_available())
import argparse


#define save function.
def saveRawFile(t, volume, letter):
    #(1)convert tensor to numpy ndarray.
    volume_np = volume.detach().numpy()

    #(2)save volume_np as .raw file.
    fileName = "%svolume_np_%.4d_%s.raw" % (dataSavePath, (fileStartVal + t * fileIncrement) / constVal, letter)
    volume_np.astype('float32').tofile(fileName)
    print('volume_np has been saved.\n')



def saveRawFilex(t, volume):
    fileName = '%sv9_6_%.2d.raw' % (dataSavePath, (fileStartVal + t * fileIncrement) / constVal)
    volume = volume.view(dim_highResVolumes[0], dim_highResVolumes[1], dim_highResVolumes[2])
    #copy tensor from gpu to cpu.
    volume = volume.cpu()
    #convert tensor to numpy ndarray.
    volume = volume.detach().numpy()
    volume.astype('float32').tofile(fileName)



#modified RandomCrop3D class (refer to: https://discuss.pytorch.org/t/efficient-way-to-crop-3d-image-in-pytorch/78421), which:
#with one call, crop 3 input volumes at the same position;
#with different calls, randomly crop volumes at different positions.
class MyRandomCrop3D3(object):
    def __init__(self, volume_sz, cropVolume_sz):
        c, d, h, w = volume_sz
        assert (d, h, w) >= cropVolume_sz
        self.volume_sz = tuple((d, h, w))
        self.cropVolume_sz = tuple(cropVolume_sz)


    def __call__(self, volume_t1, volume_t2, volume_t3):
        slice_dhw = [self._get_slice(i, k) for i, k in zip(self.volume_sz, self.cropVolume_sz)]
        return self._crop(volume_t1, volume_t2, volume_t3, *slice_dhw)


    @staticmethod
    def _get_slice(volume_sz, cropVolume_sz):
        try:
            lower_bound = torch.randint(volume_sz - cropVolume_sz, (1,)).item()
            return lower_bound, lower_bound + cropVolume_sz
        except:
            return (None, None)


    @staticmethod
    def _crop(volume_t1, volume_t2, volume_t3, slice_d, slice_h, slice_w):
        #print(f"slice_d[0]:slice_d[1], slice_h[0]:slice_h[1], slice_w[0]:slice_w[1]: {slice_d[0], slice_d[1], slice_h[0], slice_h[1], slice_w[0], slice_w[1]}")
        return volume_t1[:, slice_d[0]:slice_d[1], slice_h[0]:slice_h[1], slice_w[0]:slice_w[1]], \
               volume_t2[:, slice_d[0]:slice_d[1], slice_h[0]:slice_h[1], slice_w[0]:slice_w[1]], \
               volume_t3[:, slice_d[0]:slice_d[1], slice_h[0]:slice_h[1], slice_w[0]:slice_w[1]]



#custom dataset: return 3 consecutive input lowResVolumes and 3 correspondingly output highResVolumes.
class VolumesDataset3_test(Dataset):
    def __init__(self, dataSourcePath, totalTimesteps, nTestTimesteps,
                 dim, dim_highResVolumes, dim_lowResVolumes, dsamScaleFactor,
                 fileStartVal=35, fileIncrement=1, constVal=1, float32DataType=np.float32,
                 transform=None, dataInt=2):
        self.dataSourcePath = dataSourcePath
        self.totalTimesteps = totalTimesteps    #totalTimesteps=48.
        self.nTestTimesteps = nTestTimesteps    #nTestTimesteps=14.
        self.fileStartVal = fileStartVal
        self.fileIncrement = fileIncrement
        self.constVal = constVal
        self.dsamScaleFactor = dsamScaleFactor
        self.float32DataType = float32DataType
        self.transform = transform
        self.dim = dim
        self.dim_highResVolumes = dim_highResVolumes
        self.dim_lowResVolumes = dim_lowResVolumes
        self.dataInt = dataInt
        #test.
        #print(f"highResVolumesDim.shape: {self.highResVolumesDim[0], self.highResVolumesDim[1], self.highResVolumesDim[2]}")
        #print(f"lowResVolumesDim.shape: {self.lowResVolumesDim[0], self.lowResVolumesDim[1], self.lowResVolumesDim[2]}")
        # test.


    def __len__(self):
        return self.nTestTimesteps  #nTestTimesteps=14.


    #given a timestep t2, return one sample including:
    #3 consecutive input lowResVolumes_t1/t2/t3, and
    #3 correspondingly output highResVolumes_t1/t2/t3.
    def __getitem__(self, t2): #t2:[0, 13]=>[34, 47].
        #at a timestep t2:
        #(1)conver t2 range from [0, 13] to [34, 47].
        t2 = t2 + (self.totalTimesteps - self.nTestTimesteps)   #t2: [34, 47].

        if t2 < (self.totalTimesteps-self.nTestTimesteps) or t2 >= self.totalTimesteps:
            print(f't2: {t2} is outside the normal range.\n')
            return

        #(2)given t2 (t2!=34 and t2!=47), generate t1/t3.
        if t2 == (self.totalTimesteps-self.nTestTimesteps): #t2==34.
            t2 = self.totalTimesteps-self.nTestTimesteps+1  #35.
        elif t2 == (self.totalTimesteps-1): #t2==47.
            t2 = self.totalTimesteps - 2    #46.

        t1, t3 = t2-1, t2+1


        #(3)at t1/t2/t3, read original data volume_t1/t2/t3.
        #(3.1)read original volume_t1.
        if self.dataInt == 2:
            fileName = "%snormInten_%.2d.raw" % (
                self.dataSourcePath, (self.fileStartVal + t1 * self.fileIncrement) / self.constVal)
        elif self.dataInt == 10:
            fileName = "%snormInten_%.2f.raw" % (
                self.dataSourcePath, (self.fileStartVal + t1 * self.fileIncrement) / self.constVal)
        else:
            fileName = "%snormInten_%.4d.raw" % (
                self.dataSourcePath, (self.fileStartVal + t1 * self.fileIncrement) / self.constVal)

        volume_t1 = np.fromfile(fileName, dtype=self.float32DataType)
        # convert numpy ndarray to tensor.
        volume_t1 = torch.from_numpy(volume_t1)
        # reshape.
        volume_t1 = volume_t1.view([1, self.dim[0], self.dim[1], self.dim[2]])  # [channels, depth, height, width].

        # (2.2)read original volume_t2.
        if self.dataInt == 2:
            fileName = "%snormInten_%.2d.raw" % (
                self.dataSourcePath, (self.fileStartVal + t2 * self.fileIncrement) / self.constVal)
        elif self.dataInt == 10:
            fileName = "%snormInten_%.2f.raw" % (
                self.dataSourcePath, (self.fileStartVal + t2 * self.fileIncrement) / self.constVal)
        else:
            fileName = "%snormInten_%.4d.raw" % (
                self.dataSourcePath, (self.fileStartVal + t2 * self.fileIncrement) / self.constVal)
        volume_t2 = np.fromfile(fileName, dtype=self.float32DataType)
        # convert numpy ndarray to tensor.
        volume_t2 = torch.from_numpy(volume_t2)
        # reshape.
        volume_t2 = volume_t2.view([1, self.dim[0], self.dim[1], self.dim[2]])  # [channels, depth, height, width].

        # (2.3)read original volume_t3.
        if self.dataInt == 2:
            fileName = "%snormInten_%.2d.raw" % (
                self.dataSourcePath, (self.fileStartVal + t3 * self.fileIncrement) / self.constVal)
        elif self.dataInt == 10:
            fileName = "%snormInten_%.2f.raw" % (
                self.dataSourcePath, (self.fileStartVal + t3 * self.fileIncrement) / self.constVal)
        else:
            fileName = "%snormInten_%.4d.raw" % (
                self.dataSourcePath, (self.fileStartVal + t3 * self.fileIncrement) / self.constVal)
        volume_t3 = np.fromfile(fileName, dtype=self.float32DataType)

        #convert numpy ndarray to tensor.
        volume_t3 = torch.from_numpy(volume_t3)
        #reshape.
        volume_t3 = volume_t3.view([1, self.dim[0], self.dim[1], self.dim[2]])   #[channels, depth, height, width].


        #(4)crop original data to get crop data.
        if self.transform:
            tempHResVolume_t1, tempHResVolume_t2, tempHResVolume_t3 = self.transform(volume_t1, volume_t2, volume_t3)   #transform: [channels, depth, height, width].
            #test.
            #print(f"tempHResVolume_t3.shape: {tempHResVolume_t3.shape}")
            #saveRawFile(t1, tempHResVolume_t1, '1')
            #saveRawFile(t2, tempHResVolume_t2, '1')
            #saveRawFile(t3, tempHResVolume_t3, '1')
            #print(f"tempHResVolume_t3.view.shape: {tempHResVolume_t3.view([self.highResVolumesDim[0], self.highResVolumesDim[1], self.highResVolumesDim[2]]).shape}")
            #test: correct (during test, when 6 raw files are saved, the 1st 4 raw files are not cropped at the same position.
            #This is because the timesteps between the 1st 4 raw files and 2nd 4 raw files are partially overlap, and thus the 2nd raw files rewrite the 1st raw files.
            #If no overlap, then there should be 8 raw files in total).


        #(5)declare highResVolumes ([channels, depth, height, width]), and store crop data into it.
        highResVolumes = torch.zeros([3, self.dim_highResVolumes[0], self.dim_highResVolumes[1], self.dim_highResVolumes[2]])
        highResVolumes[0, :, :, :] = tempHResVolume_t1
        highResVolumes[1, :, :, :] = tempHResVolume_t2
        highResVolumes[2, :, :, :] = tempHResVolume_t3
        #test.
        #print(f"highResVolumes.shape: {highResVolumes.shape}")
        #saveRawFile(t1, highResVolumes[0, :, :, :], '2')
        #saveRawFile(t2, highResVolumes[1, :, :, :], '2')
        #saveRawFile(t3, highResVolumes[2, :, :, :], '2')
        #test: correct.


        #(6)downsample crop data to be lowResVolumes, and store them into lowResVolumes.
        #(6.1)reshape tempHResVolume_t1/t2/t3 to [mini-batch, channels, [optional depth], [optional height], width].
        tempHResVolume_t1t1 = tempHResVolume_t1.view([1, tempHResVolume_t1.shape[0], tempHResVolume_t1.shape[1], tempHResVolume_t1.shape[2], tempHResVolume_t1.shape[3]])  # [mini-batch, channels, [optional depth], [optional height], width].
        tempHResVolume_t2t2 = tempHResVolume_t2.view([1, tempHResVolume_t2.shape[0], tempHResVolume_t2.shape[1], tempHResVolume_t2.shape[2], tempHResVolume_t2.shape[3]])  # [mini-batch, channels, [optional depth], [optional height], width].
        tempHResVolume_t3t3 = tempHResVolume_t3.view([1, tempHResVolume_t3.shape[0], tempHResVolume_t3.shape[1], tempHResVolume_t3.shape[2], tempHResVolume_t3.shape[3]])  # [mini-batch, channels, [optional depth], [optional height], width].
        #test.
        #print(f"tempHResVolume_tt.shape: {tempHResVolume_tt.shape}, tempHResVolume_t1t1.shape: {tempHResVolume_t1t1.shape}, tempHResVolume_t2t2.shape: {tempHResVolume_t2t2.shape}, tempHResVolume_t3t3.shape: {tempHResVolume_t3t3.shape}")
        #test: correct.


        #(6.2)downsample tempHResVolume_t11/t22/t33.
        lowResVolumes = torch.zeros([3, self.dim_lowResVolumes[0], self.dim_lowResVolumes[1], self.dim_lowResVolumes[2]])
        lowResVolumes[0, :, :, :] = nn.functional.interpolate(tempHResVolume_t1t1, scale_factor=self.dsamScaleFactor, mode='trilinear', align_corners=True)
        lowResVolumes[1, :, :, :] = nn.functional.interpolate(tempHResVolume_t2t2, scale_factor=self.dsamScaleFactor, mode='trilinear', align_corners=True)
        lowResVolumes[2, :, :, :] = nn.functional.interpolate(tempHResVolume_t3t3, scale_factor=self.dsamScaleFactor, mode='trilinear', align_corners=True)
        #test.
        #print(f"lowResVolumes.shape: {lowResVolumes.shape}")
        #print(f"highResVolumes.shape: {highResVolumes.shape}")
        #saveRawFile(t1, lowResVolumes[0, :, :, :], '3')
        #saveRawFile(t2, lowResVolumes[1, :, :, :], '3')
        #saveRawFile(t3, lowResVolumes[2, :, :, :], '3')
        #saveRawFile(t1, highResVolumes[0, :, :, :], '2')
        #saveRawFile(t2, highResVolumes[1, :, :, :], '2')
        #saveRawFile(t3, highResVolumes[2, :, :, :], '2')
        #test.


        return lowResVolumes, highResVolumes, t1, t2, t3
        #correct.



#add on 2022.4.10.
#define crop function, which crops a volume ([batch size, channels, depth, height, width]) center to
#the cropVolume_shape ([batch size, channels, depth, height, width]).
def crop(volume, cropVolume_shape):
    middle_depth = volume.shape[2] // 2
    middle_height = volume.shape[3] // 2
    middle_width = volume.shape[4] // 2

    #compute cropped depth.
    start_depth = middle_depth - cropVolume_shape[2] // 2
    end_depth = start_depth + cropVolume_shape[2]

    #compute cropped height.
    start_height = middle_height - cropVolume_shape[3] // 2
    end_height = start_height + cropVolume_shape[3]

    #compute cropped width.
    start_width = middle_width - cropVolume_shape[4] // 2
    end_width = start_width + cropVolume_shape[4]

    cropVolume = volume[:, :, start_depth:end_depth, start_height:end_height, start_width:end_width]
    return cropVolume



#define ContractBlock class.

# class ResidualBlock(nn.Module):
#     def __init__(self, in_features):
#         super(ResidualBlock, self).__init__()
#
#         conv_block = [
#             nn.utils.spectral_norm(nn.Conv3d(in_features, in_features, 3, stride=1, padding=1, bias=False)),
#             nn.InstanceNorm3d(in_features, 0.8),
#             nn.ReLU(inplace=True),
#             nn.utils.spectral_norm(nn.Conv3d(in_features, in_features, 3, stride=1, padding=1, bias=False)),
#             nn.InstanceNorm3d(in_features, 0.8),
#         ]
#
#         self.conv_block = nn.Sequential(*conv_block)
#
#     def forward(self, x):
#         return x + self.conv_block(x)

# class GeneratorResNet(nn.Module):
#     def __init__(self, in_channel=3, out_channel=3, inner_channel=64, res_block=9):
#         super(GeneratorResNet, self).__init__()
#         self.in_channel = in_channel
#         self.out_channel = out_channel
#         self.inner_channel = inner_channel
#         # 初始化卷积层 k=7,s=1,p=3 通道3->64
#         # 这里也可以改成1卷积
#         # v5_3 修改为两个conv0
#         #self.conv_first = nn.Sequential(
#         #    nn.Conv3d(self.in_channel, self.inner_channel, 7, stride=1, padding=3, bias=False),
#         #    nn.InstanceNorm3d(self.inner_channel, affine=True, track_running_stats=True))
#         self.conv_first = nn.Sequential(
#             nn.Conv3d(self.in_channel, self.inner_channel // 2, 1, stride=1, padding=0, bias=False),
#             nn.Conv3d(self.inner_channel // 2, self.inner_channel, 3, stride=1, padding=1, bias=False),
#             nn.InstanceNorm3d(self.inner_channel, affine=True, track_running_stats=True),
#             nn.ReLU(inplace=True),
#             nn.Conv3d(self.inner_channel, self.inner_channel * 2, 3, stride=1, padding=1, bias=False),
#             nn.InstanceNorm3d(self.inner_channel * 2, affine=True, track_running_stats=True),
#         )
#
#         # 降采样
#
#         self.downsampling = nn.Sequential(
#             nn.Conv3d(self.inner_channel * 2, self.inner_channel * 4, 4, stride=2, padding=1, bias=False),
#             nn.InstanceNorm3d(self.inner_channel * 4, affine=True, track_running_stats=True),
#             nn.ReLU(inplace=True),
#         )
#         resblock1 = []
#         # 残差块 此时通道数256
#         for _ in range(res_block):
#             resblock1 += [ResidualBlock(self.inner_channel * 4)]
#         self.res1 = nn.Sequential(
#             *resblock1,
#             nn.Conv3d(self.inner_channel * 4, self.inner_channel * 4, 3, stride=1, padding=1, bias=False),
#         )
#         # 上采样 此时通道数256
#         #######################
#         # 插入resblock
#         resblockb = []
#         for _ in range(res_block):
#             resblockb += [ResidualBlock(self.inner_channel * 2)]
#         self.resb= nn.Sequential(
#             *resblockb,
#             nn.Conv3d(self.inner_channel * 2, self.inner_channel * 2, 3, stride=1, padding=1, bias=False),
#         )
#         self.upsamplingb = nn.Sequential(
#             nn.ConvTranspose3d(self.inner_channel * 2, self.inner_channel, 4, stride=2, padding=1, bias=False),
#             nn.InstanceNorm3d(self.inner_channel, affine=True, track_running_stats=True),
#             nn.ReLU(inplace=True),
#             nn.ConvTranspose3d(self.inner_channel, self.inner_channel // 2, 4, stride=2, padding=1, bias=False),
#             nn.InstanceNorm3d(self.inner_channel // 2, affine=True, track_running_stats=True),
#             nn.ReLU(inplace=True),
#             #nn.Conv3d(self.inner_channel // 2, self.inner_channel, 3, stride=1, padding=1, bias=False),
#             #nn.InstanceNorm3d(self.inner_channel, affine=True, track_running_stats=True),
#             #nn.ReLU(inplace=True),
#             #nn.Conv3d(self.inner_channel * 2, self.inner_channel * 8, 3, 1, 1),
#             #PixelShuffle3d(2),
#             #nn.Conv3d(self.inner_channel, self.inner_channel * 4, 3, 1, 1),
#             #PixelShuffle3d(2)
#         )
#         ######################
#         self.upsampling1 = nn.Sequential(
#             nn.ConvTranspose3d(self.inner_channel * 4, self.inner_channel * 2, 4, stride=2, padding=1, bias=False),
#             nn.InstanceNorm3d(self.inner_channel * 2, affine=True, track_running_stats=True),
#             nn.ReLU(inplace=True),
#             #nn.Conv3d(self.inner_channel * 4, self.inner_channel * 16, 3, 1, 1),
#             #PixelShuffle3d(2)
#         )
#         # 残差块 此时通道数128
#
#         resblock2 = []
#         for _ in range(res_block):
#             resblock2 += [ResidualBlock(self.inner_channel * 4)]
#
#         self.res2 = nn.Sequential(
#             *resblock2,
#             nn.Conv3d(self.inner_channel * 4, self.inner_channel * 4, 3, stride=1, padding=1, bias=False),
#         )
#         # 上采样 此时通道数128
#
#         self.upsampling2 = nn.Sequential(
#             nn.ConvTranspose3d(self.inner_channel * 4, self.inner_channel * 2, 4, stride=2, padding=1, bias=False),
#             nn.InstanceNorm3d(self.inner_channel * 2, affine=True, track_running_stats=True),
#             nn.ReLU(inplace=True),
#             nn.ConvTranspose3d(self.inner_channel * 2, self.inner_channel, 4, stride=2, padding=1, bias=False),
#             nn.InstanceNorm3d(self.inner_channel, affine=True, track_running_stats=True),
#             nn.ReLU(inplace=True),
#             #nn.Conv3d(self.inner_channel * 4, self.inner_channel * 16, 3, 1, 1),
#             #PixelShuffle3d(2),
#             #nn.Conv3d(self.inner_channel * 2, self.inner_channel * 8, 3, 1, 1),
#             #PixelShuffle3d(2),
#         )
#         self.output = nn.Sequential(
#             nn.Conv3d(self.inner_channel * 3 // 2, self.inner_channel, 3, stride=1, padding=1),
#             nn.InstanceNorm3d(self.inner_channel, affine=True, track_running_stats=True),
#             nn.ReLU(inplace=True),
#             nn.Conv3d(self.inner_channel, self.inner_channel // 2, 3, stride=1, padding=1),
#             nn.InstanceNorm3d(self.inner_channel // 2, affine=True, track_running_stats=True),
#             nn.ReLU(inplace=True),
#             nn.Conv3d(self.inner_channel // 2, self.out_channel, 1, stride=1, padding=0, bias=False),
#             nn.Tanh()
#         )
#         # 输出层 此时通道数16 out_channel
#
#     def forward(self, x):
#         x = self.conv_first(x)
#         x1 = x  # 跳跃连接
#         x = self.downsampling(x)
#
#         x = self.res1(x)
#         x = self.upsampling1(x)
#         x = torch.cat([x1, x], dim=1)
#
#         x = self.res2(x)
#         x = self.upsampling2(x)
#
#         xb = self.resb(x1)
#         xb = self.upsamplingb(xb) # 已包括卷积
#         x = torch.cat([xb, x], dim=1)
#         output = self.output(x)
#         return output
#test.
#disc = Discriminator_PatchGAN(6)
#print(disc)
#test.

class Self_Attn(nn.Module):
    def __init__(self, in_dim):
        super(Self_Attn, self).__init__()
        # 卷积层
        self.query_conv = nn.Conv3d(in_channels=in_dim, out_channels=in_dim//2, kernel_size=1)
        self.key_conv = nn.Conv3d(in_channels=in_dim, out_channels=in_dim//2, kernel_size=1)
        self.value_conv = nn.Conv3d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        # 初始 gamma
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """

            input :
                x : input feature maps( B * C * W * H)

        """


        m_batchsize, C, width, height, depth = x.size()

        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height*depth).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height*depth)
        energy = torch.bmm(proj_query, proj_key)

        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height*depth)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height, depth)

        out = self.gamma*out + x
        return out

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        # 修改为SN层
        conv_block = [
            nn.utils.spectral_norm(nn.Conv3d(in_features, in_features, 3, stride=1, padding=1, bias=False)),
            nn.BatchNorm3d(in_features),
            #nn.PReLU(),
            nn.ReLU(inplace=True),
            nn.utils.spectral_norm(nn.Conv3d(in_features, in_features, 3, stride=1, padding=1, bias=False)),
            nn.BatchNorm3d(in_features),
        ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)


class FeatureExatract(nn.Module):
    def __init__(self, in_features, out_features, dense=False):
        super(FeatureExatract, self).__init__()
        self.dense = dense
        self.out_features = out_features
        self.conv_block_up = nn.Sequential(
            # 四个卷积层
            nn.utils.spectral_norm(nn.Conv3d(in_features, out_features, kernel_size=3, stride=1, padding=1)),
            nn.BatchNorm3d(out_features),
            nn.ReLU(),

            nn.utils.spectral_norm(nn.Conv3d(out_features, out_features, kernel_size=3, stride=1, padding=1)),
            nn.BatchNorm3d(out_features),
            nn.ReLU(),

            nn.utils.spectral_norm(nn.Conv3d(out_features, out_features, kernel_size=3, stride=1, padding=1)),
            nn.BatchNorm3d(out_features),
        )
        self.conv_block_down = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv3d(in_features, out_features, kernel_size=3, stride=1, padding=1)),
        )

    def forward(self, x):
        # return self.conv_block_up(x) + self.conv_block_down(x) + x
        # print(self.conv_block_up(x).shape)
        # print(x.shape)
        if self.dense:
            x1 = self.conv_block_up(x)
            x2 = self.conv_block_down(x)
            # dense 链接后， 通道数为应该输出的尺寸大小加上输入的尺寸大小
            # torch.cat([x1 + x2, x], dim=1)
            return x1 + x2 + x
        else:
            return self.conv_block_down(x) + self.conv_block_up(x)



class GeneratorResNet(nn.Module):
    def __init__(self, in_channel=3, out_channel=3, inner_channel=64, res_block=4):
        super(GeneratorResNet, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.inner_channel = inner_channel
        # 初始化卷积层 k=7,s=1,p=3 通道3->64
        # 这里也可以改成1卷积
        # v5_3 修改为两个conv0
        #self.conv_first = nn.Sequential(
        #    nn.Conv3d(self.in_channel, self.inner_channel, 7, stride=1, padding=3, bias=False),
        #    nn.InstanceNorm3d(self.inner_channel, affine=True, track_running_stats=True))

        self.conv0 = nn.Sequential(
            nn.Conv3d(self.in_channel, self.inner_channel, 1)
        )

        self.conv_first = nn.Sequential(
            #nn.Conv3d(self.in_channel, self.inner_channel // 2, 1, stride=1, padding=0, bias=False),
            nn.Conv3d(self.in_channel, self.inner_channel, 9, stride=1, padding=4, bias=False),
            nn.BatchNorm3d(self.inner_channel),
            nn.PReLU(),
            nn.Conv3d(self.inner_channel, self.inner_channel * 2, 5, stride=1, padding=2, bias=False),
            nn.BatchNorm3d(self.inner_channel * 2),
        )

        self.downsampling = nn.Sequential(
            nn.Conv3d(self.inner_channel * 2, self.inner_channel * 4, 4, stride=2, padding=1),
            nn.BatchNorm3d(self.inner_channel * 4),
            nn.PReLU(),
        )

        self.self_attention = Self_Attn(self.inner_channel * 4)

        self.deconv1 = nn.Sequential(
            nn.utils.spectral_norm(
                nn.ConvTranspose3d(self.inner_channel * 4, self.inner_channel * 2, 4, stride=2, padding=1)),
            nn.BatchNorm3d(self.inner_channel * 2),
            nn.PReLU(),
            # nn.Conv3d(self.inner_channel * 4, self.inner_channel * 16, 3, 1, 1),
            # PixelShuffle3d(2)
        )
        # 解卷积
        # 2FE -> 1FE



        resblock1 = []
        # 残差块 此时通道数256
        for _ in range(res_block):
            resblock1 += [ResidualBlock(self.inner_channel * 2)]
        self.res1 = nn.Sequential(
            *resblock1,
            nn.Conv3d(self.inner_channel * 2, self.inner_channel * 2, 3, stride=1, padding=1),
        )

        # 1FE -> 2FE


        # 上采样 此时通道数256
        #######################
        # 插入resblock
        resblockb = []
        for _ in range(res_block):
            resblockb += [ResidualBlock(self.inner_channel * 2)]
        self.resb= nn.Sequential(
            *resblockb,
            nn.Conv3d(self.inner_channel * 2, self.inner_channel * 2, 3, stride=1, padding=1, bias=False),
        )
        self.deconvb = nn.Sequential(
            nn.utils.spectral_norm(nn.ConvTranspose3d(self.inner_channel * 2, self.inner_channel, 4, stride=2, padding=1)),
            nn.InstanceNorm3d(self.inner_channel, affine=True, track_running_stats=True),
            nn.PReLU(),
            #nn.Conv3d(self.inner_channel // 2, self.inner_channel, 3, stride=1, padding=1, bias=False),
            #nn.InstanceNorm3d(self.inner_channel, affine=True, track_running_stats=True),
            #nn.ReLU(inplace=True),
            #nn.Conv3d(self.inner_channel * 2, self.inner_channel * 8, 3, 1, 1),
            #PixelShuffle3d(2),
            #nn.Conv3d(self.inner_channel, self.inner_channel * 4, 3, 1, 1),
            #PixelShuffle3d(2)
        )
        ######################

        # 残差块 此时通道数128

        resblock2 = []
        for _ in range(res_block):
            resblock2 += [ResidualBlock(self.inner_channel * 4)]

        self.res2 = nn.Sequential(
            *resblock2,
            nn.Conv3d(self.inner_channel * 4, self.inner_channel * 4, 3, stride=1, padding=1, bias=False),
        )
        # 上采样 此时通道数128

        self.deconv2 = nn.Sequential(
            nn.utils.spectral_norm(nn.ConvTranspose3d(self.inner_channel * 4, self.inner_channel * 2, 4, stride=2, padding=1)),
            nn.BatchNorm3d(self.inner_channel * 2),
            nn.PReLU(),
            FeatureExatract(self.inner_channel * 2, self.inner_channel * 2, dense=True),
            nn.PReLU(),
            #nn.Conv3d(self.inner_channel * 4, self.inner_channel * 16, 3, 1, 1),
            #PixelShuffle3d(2),
            #nn.Conv3d(self.inner_channel * 2, self.inner_channel * 8, 3, 1, 1),
            #PixelShuffle3d(2),
        )

        self.deconv3 = nn.Sequential(
            nn.utils.spectral_norm(
                nn.ConvTranspose3d(self.inner_channel * 2, self.inner_channel, 4, stride=2, padding=1)),
            #nn.utils.spectral_norm(nn.Conv3d(self.inner_channel * 2, self.inner_channel * 1, 3, 1, 1)),
            #nn.Upsample(scale_factor=2.5, mode='trilinear', align_corners=True),
            # nn.Conv3d(self.inner_channel * 2, self.inner_channel * 8, 3, 1, 1),
            # PixelShuffle3d(2),
            nn.InstanceNorm3d(self.inner_channel, affine=True, track_running_stats=True),
            nn.PReLU(),
        )

        resblock3 = []
        for _ in range(res_block):
            resblock3 += [ResidualBlock(self.inner_channel * 2)]

        self.res3 = nn.Sequential(
            *resblock3,
            nn.Conv3d(self.inner_channel * 2, self.inner_channel * 2, 3, stride=1, padding=1, bias=False),
        )

        self.upsampling_3 = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='trilinear', align_corners=True),
            #nn.PReLU(),
        )

        self.output = nn.Sequential(
            #nn.ConvTranspose3d(self.inner_channel * 2, self.out_channel, 4, stride=2, padding=1),
            #nn.Tanh(),
            FeatureExatract(self.inner_channel, self.inner_channel // 2),
            # nn.Conv3d(self.inner_channel, self.inner_channel // 2, 3, stride=1, padding=1),
            # nn.InstanceNorm3d(self.inner_channel // 2, affine=True, track_running_stats=True),
            nn.PReLU(),
            # FeatureExatract(self.inner_channel // 2, self.inner_channel // 4),
            # nn.PReLU(),
            # nn.Conv3d(self.inner_channel, self.inner_channel // 2, 3, stride=1, padding=1),
            # nn.InstanceNorm3d(self.inner_channel // 2, affine=True, track_running_stats=True),
            # nn.ReLU(inplace=True),
            nn.Conv3d(self.inner_channel // 2, self.out_channel, 1, stride=1, padding=0, bias=False),
            nn.Tanh(),
        )
        # 输出层 此时通道数16 out_channel

    def forward(self, x):
        x_up = self.conv0(x)
        x = self.conv_first(x)
        x1 = x  # 跳跃连接

        x = self.downsampling(x)
        x = self.self_attention(x)
        x = self.deconv1(x)
        x1 = self.res1(x1)
        x = torch.cat([x1, x], dim=1)

        x = self.deconv2(x)
        x = self.deconv3(x)

        x_up = self.upsampling_3(x_up)
        #x = torch.cat([x_up, x], dim=1)
        x = x + x_up

        output = self.output(x)
        return output

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=100, help="number of epochs of training")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=100, help="interval between saving image samples")
parser.add_argument("--checkpoint_interval", type=int, default=5000, help="batch interval between model checkpoints")
parser.add_argument("--residual_blocks", type=int, default=2, help="number of residual blocks in the generator")
parser.add_argument("--warmup_batches", type=int, default=20, help="number of batches with pixel-wise loss only")
parser.add_argument("--lambda_adv", type=float, default=5e-3, help="adversarial loss weight")
parser.add_argument("--lambda_pixel", type=float, default=1e-2, help="pixel-wise loss weight")
opt = parser.parse_args()
#0. declare variables.

server = 'dell'
dataset_name = "viscous"
if server == 'dell':
    if dataset_name == "square_cylinder":
        dataSourcePath = 'SC/test/'
        dataSavePath = 'SC/saved/'
    elif dataset_name == "p_square_cylinder":
        dataSourcePath = 'product_ssrtvd/SquareCylinder/'
        dataSavePath = 'product_ssrtvd/SquareCylinder/saved/'
    elif dataset_name == "ionization":
        dataSourcePath = 'ionization_ab_H/test/'
        dataSavePath = 'ionization_ab_H/saved/v9_6_pre/'
    elif dataset_name == "hurricane_wind":
        dataSourcePath = 'hurricane_wind/test/'
        dataSavePath = 'hurricane_wind/saved/v9_6_pre/'
    elif dataset_name == "hurricane_qsnow":
        dataSourcePath = 'hur_qsnow_048/test/'
        dataSavePath = 'hur_qsnow_048/saved/v9_6_predict/'
    elif dataset_name == "p_hurricane_qsnow":
        dataSourcePath = 'product_ssrtvd/hurricane_QSNOW/'
        dataSavePath = 'product_ssrtvd/hurricane_QSNOW/saved/pre_file/'
    elif dataset_name == "p_hurricane_wind":
        dataSourcePath = 'product_ssrtvd/hurricane_wind/'
        dataSavePath = 'product_ssrtvd/hurricane_wind/saved/pre_file/'
    elif dataset_name == "viscous":
        dataSourcePath = "viscous096/test/"
        dataSavePath = "viscous096/saved/v9_4_pre/"
    elif dataset_name == "halfcylinder6400":
        dataSourcePath = "halfcylinder3d-Re6400_am/"
        dataSavePath = "halfcylinder3d-Re6400_am/saved/v9_6_predict/"
    print(server)
    print(dataset_name)
else:
    dataSourcePath = '../../../../../../Bigdata/maji/data/SquareCylinder/test/'
    dataSavePath = '../../../../../../Bigdata/maji/data/SquareCylinder/ex-y/predict/'
    print("maji")
    print("SquareCylinder")
totalTimesteps = 102
fileStartVal = 1
fileIncrement = 1
constVal = 1
cropScaleFactor = (1, 1, 1)   #[depth, height, width].
dsamScaleFactor = (0.25, 0.25, 0.25)    #[depth, height, width].
dim = (48, 240, 240)     #[depth, height, width].


if dataset_name == "hurricane_wind":
    dim = (48, 240, 240)
    fileStartVal = 1
    fileIncrement = 1
    filter_size = (3, 9, 9)
    dataInt = 2
    totalTimesteps = 48
elif dataset_name == "hurricane_qsnow":
    dim = (48, 240, 240)
    fileStartVal = 1
    fileIncrement = 1
    filter_size = (3, 9, 9)
    dataInt = 2
    totalTimesteps = 48
elif dataset_name == "p_hurricane_qsnow" or dataset_name == "p_hurricane_wind":
    dim = (50, 250, 250)
    fileStartVal = 1
    fileIncrement = 1
    filter_size = (3, 9, 9)
    dataInt = 2
    totalTimesteps = 48
    #cropScaleFactor = (0.48, 0.48, 0.48)  # [depth, height, width].
    dsamScaleFactor = (0.25, 0.25, 0.25)  # [depth, height, width].
elif dataset_name == "square_cylinder" or dataset_name == "p_square_cylinder":
    dim = (48, 64, 192)
    fileStartVal = 8
    fileIncrement = 40
    filter_size = (3, 4, 8)
    dataInt = 4
    totalTimesteps = 102
elif dataset_name == "ionization":
    dim = (96, 96, 144)
    fileStartVal = 0
    fileIncrement = 1
    filter_size = (5, 5, 6)
    dataInt = 4
    totalTimesteps = 100
elif dataset_name == "viscous":
    dim = (96, 96, 96)
    fileStartVal = 0
    fileIncrement = 1
    filter_size = (3, 3, 3)
    dataInt = 4
    totalTimesteps = 121
    #cropScaleFactor = (0.96, 0.96, 0.96)  # [depth, height, width].
    dsamScaleFactor = (0.25, 0.25, 0.25)  # [depth, height, width].
    print(dataset_name)
    print(dim)
elif dataset_name == "halfcylinder6400":
    dim = (40, 120, 320)
    fileStartVal = 0
    fileIncrement = 0.1
    filter_size = (1, 2, 6)
    dataInt = 10
    totalTimesteps = 151
    print(dataset_name)
    print(dim)
    batchSize = 1
    cropScaleFactor = (1, 1, 1)  # [depth, height, width].


trainsetRatio = 0.7 #according to deep learning specialization, if you have a small collection of data, then use 70%/30% for train/test.
nTrainTimesteps = round(totalTimesteps * trainsetRatio)   #nTrainTimesteps=34.
nTestTimesteps = totalTimesteps - nTrainTimesteps   #nTestTimesteps=14.
dim_highResVolumes = (int(dim[0] * cropScaleFactor[0]),
                                 int(dim[1] * cropScaleFactor[1]),
                                 int(dim[2] * cropScaleFactor[2]))
dim_lowResVolumes = (int(dim_highResVolumes[0] * dsamScaleFactor[0]),
                                 int(dim_highResVolumes[1] * dsamScaleFactor[1]),
                                 int(dim_highResVolumes[2] * dsamScaleFactor[2]))
float32DataType = np.float32
myRandCrop3D = MyRandomCrop3D3(volume_sz=(1, dim[0], dim[1], dim[2]),
            cropVolume_sz=dim_highResVolumes)
batchSize = 1   #must be 1.

#lr = 0.0002
lr = 0.0002
beta1 = 0.5
beta2 = 0.999
device = torch.device('cuda:1')
lowResVolumes_channels = 3
highResVolumes_channels = 3
final_train_step = 2800
mseCriterion = nn.MSELoss()



#1. instantiate the test dataset, and load it into dataloader.
testDataset = VolumesDataset3_test(
    dataSourcePath=dataSourcePath, totalTimesteps=totalTimesteps, nTestTimesteps=nTestTimesteps,
    dim=dim, dim_highResVolumes=dim_highResVolumes, dim_lowResVolumes=dim_lowResVolumes,
    dsamScaleFactor=dsamScaleFactor,
    fileStartVal=fileStartVal, fileIncrement=fileIncrement, constVal=constVal, float32DataType=float32DataType,
    transform=myRandCrop3D, dataInt=dataInt
)
#test.
#print(f"trainDataset len: {trainDataset.__len__()}")
#trainDataset.__getitem__(15)
#test.


testDataloader = DataLoader(testDataset, batch_size=batchSize, shuffle=False)
#correct.


res_layer = 16


#2. initialize Generator, its optimizers, and load their state_dict.
#print(1)
gen = GeneratorResNet(3, 3, 64, res_layer).to(device)
gen_opt = torch.optim.Adam(gen.parameters(), lr=lr, betas=(beta1, beta2))
#print(2)
#load Generator and its optimizer's state_dict.
#fileName_state_dict = "../../../../../../Bigdata/maji/data/SquareCylinder/ex-y/srg_v5_1_5040.pth"
version = "v9_4"
if dataset_name == "hurricane_wind":
    fileName_state_dict = f"ionization_ab_H/saved/{version}/srg_{version}_14000.pth"
elif dataset_name == "square_cylinder":
    fileName_state_dict = f"SC/saved/{version}/srg_{version}_14000.pth"
elif dataset_name == "hurricane_wind":
    fileName_state_dict = f"hurricane_wind/saved/{version}/srg_{version}_14000.pth"
elif dataset_name == "hurricane_qsnow":
    fileName_state_dict = f"hur_qsnow_048/saved/{version}/srg_{version}_14000.pth"
def predictGAN(generator):
    #Dataloader returns the batches.

    gen = generator
    ssim = 0
    curStep = 0
    total_psnr = 0
    for real_lowResVolumes, real_highResVolumes, t1, t2, t3 in tqdm(testDataloader):    #real_lowResVolumes: [1, 3, 12, 16, 48].
        #get current batch size.
        curBatchSize = len(real_lowResVolumes)
        #copy real_lowResVolumes to gpu.
        real_lowResVolumes = real_lowResVolumes.to(device)
        real_highResVolumes = real_highResVolumes.to(device)

        curStep += 1
        #print(real_highResVolumes.shape)
        #(3.1)predict by using Generator. highResVolumes_predict: [batch size, channels, depth, height, width].
        #print(f'real_lowVolumes.shape: {real_lowResVolumes.shape}')
        highResVolumes_predict = gen(real_lowResVolumes)
        #print(real_highResVolumes.shape)
        #print(highResVolumes_predict.shape)


        ############## 计算SSIM
        ssim_loss = pytorch_ssim.SSIM3D(window_size=11)
        ll = ssim_loss(highResVolumes_predict, real_highResVolumes).item()
        ssim += ll
        #print(f"ssim: {ll}")
        ##############

        ############## 计算PSNR
        maxvalue = 1
        mse = mseCriterion(highResVolumes_predict, real_highResVolumes)
        mse = mse.cpu()
        mse = mse.detach().numpy()
        psnr = 10 * np.log10((maxvalue ** 2) / mse)
        print(f"ssim: {ll} | psnr: {psnr}")
        total_psnr += psnr
        ##############
        #(3.2)save highResVolumes_predict. highResVolumes_predict: [batch size, channels, depth, height, width].
        #t2: [72, 100].
        saveRawFilex(t2, highResVolumes_predict[0, 1, :, :, :])
        if t2 == (nTrainTimesteps + 1): #t2==35.
            saveRawFilex(t1, highResVolumes_predict[0, 0, :, :, :])
        elif t2 == (totalTimesteps - 2):    #t2==46.
            saveRawFilex(t3, highResVolumes_predict[0, 2, :, :, :])
    print(ssim / curStep)
    print(total_psnr / curStep)
    mean_ssim = ssim / curStep
    mean_psnr = total_psnr / curStep
    return mean_ssim, mean_psnr

#print(fileName_state_dict)
number_of_model = 1
matrix = [[0 for i in range(2)] for i in range(number_of_model)]
#print(matrix)

for i in range(number_of_model):



    #file_number = (10 + 1) * 600 + 7200
    #file_number = (1 + i) * 100
    file_number = 3000
    # fileName_state_dict = f"../../ionization_ab_H/saved/{version}/srg_{version}_{file_number}.pth"
    if dataset_name == "ionization":
        #fileName_state_dict = f"ionization_ab_H/saved/{version}/srg_{version}_{file_number}.pth"
        fileName_state_dict = f"ionization_ab_H/saved/srg_{version}_{file_number}.pth"
    elif dataset_name == "square_cylinder":
        fileName_state_dict = f"SC/saved/{version}/srg_{version}_{file_number}.pth"
    elif dataset_name == "p_square_cylinder":
        fileName_state_dict = f"product_ssrtvd/SquareCylinder/saved/{version}/srg_{version}_{file_number}.pth"
    elif dataset_name == "hurricane_wind":
        #fileName_state_dict = f"hurricane_wind/saved/{version}/srg_{version}_{file_number}.pth"
        fileName_state_dict = f"hurricane_wind/saved/srg_{version}_{file_number}.pth"
    elif dataset_name == "hurricane_qsnow":
        #fileName_state_dict = f"hur_qsnow_048/saved/srg_{version}_{file_number}.pth"
        fileName_state_dict = f"hurricane_wind/saved/srg_{version}_{file_number}.pth"
        #fileName_state_dict = f"hur_qsnow_048/saved/{version}/srg_{version}_{file_number}.pth"
    elif dataset_name == "p_hurricane_qsnow":
        fileName_state_dict = f"product_ssrtvd/hurricane_QSNOW/saved/{version}/srg_{version}_{file_number}.pth"
    elif dataset_name == "p_hurricane_wind":
        fileName_state_dict = f"product_ssrtvd/hurricane_wind/saved/{version}/srg_{version}_{file_number}.pth"
    elif dataset_name == "viscous":
        #fileName_state_dict = f"viscous096/saved/{version}/srg_{version}_{file_number}.pth"
        fileName_state_dict = f"viscous096/saved/srg_{version}_{file_number}.pth"
    elif dataset_name == "halfcylinder6400":
        #fileName_state_dict = f"halfcylinder3d-Re6400_am//saved/{version}/srg_{version}_{file_number}.pth"
        fileName_state_dict = f"halfcylinder3d-Re6400_am/saved/srg_{version}_{file_number}.pth"
    print(fileName_state_dict)
    loaded_state = torch.load(fileName_state_dict)
    gen.load_state_dict(loaded_state["gen"])
    gen_opt.load_state_dict(loaded_state["gen_opt"])

    # you must call model.eval() to set dropout and batch normalization layers
    # to evaluation mode before running inference. Refer to: https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_models_for_inference.html.
    gen.eval()
    # correct: 2022.4.11.

    ssim, psnr = predictGAN(gen)
    matrix[i] = ssim, psnr

print(f"version: {version}")
print(f"dataset: {dataset_name}")
time_step_show = list(range(number_of_model))
ssim_show = list(range(number_of_model))
psnr_show = list(range(number_of_model))
for i in range(number_of_model):
    file_number = (1 + i) * 100
    print(f"file {file_number}: ssim: {matrix[i][0]} | psnr: {matrix[i][1]}")

    #print(f"psnr: {matrix[i][1]}")
    time_step_show[i] = file_number
    ssim_show[i] = matrix[i][0]
    psnr_show[i] = matrix[i][1]
def show_table():
    plt.plot(time_step_show, ssim_show, 'b*--', alpha=0.5, linewidth=1, label='acc')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel(f'{version}_ssim')
    plt.show()

    plt.plot(time_step_show, psnr_show, 'b*--', alpha=0.5, linewidth=1, label='acc')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel(f'{version}_psnr')
    plt.show()

#show_table()

# 16800 0.8086 18.207
#### 更新之后 目标0.93 22.78
# 15120 0.8579 22.26
# 16800 0.8764 22.58

# v_2_2
# 14000 0.8916668653488159
# 21.817729353539217
# 15400 0.901126229763031
# 22.036264436936857
# 16800 0.8914212147394817
# 20.55684411055986
#

# srg_v7_2 SN
# 7000  0.8610 20.16
# 14000 0.8959 21.77
# 15400 0.9002 21.72
# 16800 0.8921 21.62


# srg_v5_4 hurricane 目标0.9482
# 12600 0.9577
