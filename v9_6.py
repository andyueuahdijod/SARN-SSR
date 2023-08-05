import functools
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import torch.nn.functional as F
import math
import pytorch_ssim
from torch.autograd import Variable
import argparse
torch.cuda.current_device()
torch.cuda._initialized = True
torch.cuda.empty_cache()
print(torch.cuda.is_available())


#  继承自王者v8_4_29
#  旨在超越王者抵达剑之勇者
def saveRawFile(t, volume, letter):
    # (1)convert tensor to numpy ndarray.
    volume_np = volume.detach().numpy()

    # (2)save volume_np as .raw file.
    fileName = "%svolume_np_%.2_%s.raw" % (dataSavePath, (fileStartVal + t * fileIncrement) / constVal, letter)
    volume_np.astype('float32').tofile(fileName)
    print('volume_np has been saved.\n')


def saveRawFile2(curStep, t, volume):
    fileName = '%sSR_%s_%d.raw' % (dataSavePath, t, curStep)
    volume = volume.view(dim_highResVolumes[0], dim_highResVolumes[1], dim_highResVolumes[2])
    # copy tensor from gpu to cpu.
    volume = volume.cpu()
    # convert tensor to numpy ndarray.
    volume = volume.detach().numpy()
    volume.astype('float32').tofile(fileName)

class PixelShuffle3d(nn.Module):
    '''
    This class is a 3d version of pixelshuffle.
    '''
    def __init__(self, scale):
        '''
        :param scale: upsample scale
        '''
        super().__init__()
        self.scale = scale

    def forward(self, input):
        batch_size, channels, in_depth, in_height, in_width = input.size()
        nOut = channels // self.scale ** 3

        out_depth = in_depth * self.scale
        out_height = in_height * self.scale
        out_width = in_width * self.scale

        input_view = input.contiguous().view(batch_size, nOut, self.scale, self.scale, self.scale, in_depth, in_height, in_width)

        output = input_view.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()

        return output.view(batch_size, nOut, out_depth, out_height, out_width)
# modified RandomCrop3D class (refer to: https://discuss.pytorch.org/t/efficient-way-to-crop-3d-image-in-pytorch/78421), which:
# with one call, crop 3 input volumes at the same position;
# with different calls, randomly crop volumes at different positions.
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
        # print(f"slice_d[0]:slice_d[1], slice_h[0]:slice_h[1], slice_w[0]:slice_w[1]: {slice_d[0], slice_d[1], slice_h[0], slice_h[1], slice_w[0], slice_w[1]}")
        return volume_t1[:, slice_d[0]:slice_d[1], slice_h[0]:slice_h[1], slice_w[0]:slice_w[1]], \
               volume_t2[:, slice_d[0]:slice_d[1], slice_h[0]:slice_h[1], slice_w[0]:slice_w[1]], \
               volume_t3[:, slice_d[0]:slice_d[1], slice_h[0]:slice_h[1], slice_w[0]:slice_w[1]]


# custom dataset: return 3 consecutive input lowResVolumes and 3 correspondingly output highResVolumes.
class VolumesDataset3(Dataset):
    def __init__(self, dataSourcePath, nTimesteps, dim, dim_highResVolumes, dim_lowResVolumes, dsamScaleFactor,
                 fileStartVal=8, fileIncrement=40, constVal=1, float32DataType=np.float32,
                 transform=None, dataInt=4):
        self.dataSourcePath = dataSourcePath
        self.nTimesteps = nTimesteps
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
        # test.
        # print(f"highResVolumesDim.shape: {self.highResVolumesDim[0], self.highResVolumesDim[1], self.highResVolumesDim[2]}")
        # print(f"lowResVolumesDim.shape: {self.lowResVolumesDim[0], self.lowResVolumesDim[1], self.lowResVolumesDim[2]}")
        # test.

    def __len__(self):
        return self.nTimesteps  # =34.

    # given a timestep t2, return one sample including:
    # 3 consecutive input lowResVolumes_t1/t2/t3, and
    # 3 correspondingly output highResVolumes_t1/t2/t3.
    def __getitem__(self, t2):  # t2:[0, 33].
        # at a timestep t2:
        if t2 < 0 or t2 >= self.nTimesteps:
            print(f't2: {t2} is outside the normal range.\n')
            return

        # (1)given t2 (t2!=0 and t2!=33), generate t1/t3.
        if t2 == 0:
            t2 = 1
        elif t2 == (self.nTimesteps - 1):
            t2 = self.nTimesteps - 2

        t1, t3 = t2 - 1, t2 + 1

        # (2)at t1/t2/t3, read original data volume_t1/t2/t3.
        # (2.1)read original volume_t1.

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
        # convert numpy ndarray to tensor.
        volume_t3 = torch.from_numpy(volume_t3)
        # reshape.
        volume_t3 = volume_t3.view([1, self.dim[0], self.dim[1], self.dim[2]])  # [channels, depth, height, width].

        # (3)crop original data to get crop data.
        if self.transform:
            tempHResVolume_t1, tempHResVolume_t2, tempHResVolume_t3 = self.transform(volume_t1, volume_t2,
                                                                                     volume_t3)  # transform: [channels, depth, height, width].
            # test.
            # print(f"tempHResVolume_t3.shape: {tempHResVolume_t3.shape}")
            # saveRawFile(t1, tempHResVolume_t1, '1')
            # saveRawFile(t2, tempHResVolume_t2, '1')
            # saveRawFile(t3, tempHResVolume_t3, '1')
            # print(f"tempHResVolume_t3.view.shape: {tempHResVolume_t3.view([self.highResVolumesDim[0], self.highResVolumesDim[1], self.highResVolumesDim[2]]).shape}")
            # test: correct (during test, when 6 raw files are saved, the 1st 4 raw files are not cropped at the same position.
            # This is because the timesteps between the 1st 4 raw files and 2nd 4 raw files are partially overlap, and thus the 2nd raw files rewrite the 1st raw files.
            # If no overlap, then there should be 8 raw files in total).

        # (4)declare highResVolumes ([channels, depth, height, width]), and store crop data into it.
        highResVolumes = torch.zeros(
            [3, self.dim_highResVolumes[0], self.dim_highResVolumes[1], self.dim_highResVolumes[2]])
        highResVolumes[0, :, :, :] = tempHResVolume_t1
        highResVolumes[1, :, :, :] = tempHResVolume_t2
        highResVolumes[2, :, :, :] = tempHResVolume_t3
        # test.
        # print(f"highResVolumes.shape: {highResVolumes.shape}")
        # saveRawFile(t1, highResVolumes[0, :, :, :], '2')
        # saveRawFile(t2, highResVolumes[1, :, :, :], '2')
        # saveRawFile(t3, highResVolumes[2, :, :, :], '2')
        # test: correct.

        # (5)downsample crop data to be lowResVolumes, and store them into lowResVolumes.
        # (5.1)reshape tempHResVolume_t1/t2/t3 to [mini-batch, channels, [optional depth], [optional height], width].
        tempHResVolume_t1t1 = tempHResVolume_t1.view(
            [1, tempHResVolume_t1.shape[0], tempHResVolume_t1.shape[1], tempHResVolume_t1.shape[2],
             tempHResVolume_t1.shape[3]])  # [mini-batch, channels, [optional depth], [optional height], width].
        tempHResVolume_t2t2 = tempHResVolume_t2.view(
            [1, tempHResVolume_t2.shape[0], tempHResVolume_t2.shape[1], tempHResVolume_t2.shape[2],
             tempHResVolume_t2.shape[3]])  # [mini-batch, channels, [optional depth], [optional height], width].
        tempHResVolume_t3t3 = tempHResVolume_t3.view(
            [1, tempHResVolume_t3.shape[0], tempHResVolume_t3.shape[1], tempHResVolume_t3.shape[2],
             tempHResVolume_t3.shape[3]])  # [mini-batch, channels, [optional depth], [optional height], width].
        # test.
        # print(f"tempHResVolume_tt.shape: {tempHResVolume_tt.shape}, tempHResVolume_t1t1.shape: {tempHResVolume_t1t1.shape}, tempHResVolume_t2t2.shape: {tempHResVolume_t2t2.shape}, tempHResVolume_t3t3.shape: {tempHResVolume_t3t3.shape}")
        # test: correct.

        # (5.2)downsample tempHResVolume_t11/t22/t33.
        lowResVolumes = torch.zeros(
            [3, self.dim_lowResVolumes[0], self.dim_lowResVolumes[1], self.dim_lowResVolumes[2]])
        lowResVolumes[0, :, :, :] = nn.functional.interpolate(tempHResVolume_t1t1, scale_factor=self.dsamScaleFactor,
                                                              mode='trilinear', align_corners=True)
        lowResVolumes[1, :, :, :] = nn.functional.interpolate(tempHResVolume_t2t2, scale_factor=self.dsamScaleFactor,
                                                              mode='trilinear', align_corners=True)
        lowResVolumes[2, :, :, :] = nn.functional.interpolate(tempHResVolume_t3t3, scale_factor=self.dsamScaleFactor,
                                                              mode='trilinear', align_corners=True)
        # test.
        # print(f"lowResVolumes.shape: {lowResVolumes.shape}")
        # print(f"highResVolumes.shape: {highResVolumes.shape}")
        # saveRawFile(t1, lowResVolumes[0, :, :, :], '3')
        # saveRawFile(t2, lowResVolumes[1, :, :, :], '3')
        # saveRawFile(t3, lowResVolumes[2, :, :, :], '3')
        # saveRawFile(t1, highResVolumes[0, :, :, :], '2')
        # saveRawFile(t2, highResVolumes[1, :, :, :], '2')
        # saveRawFile(t3, highResVolumes[2, :, :, :], '2')
        # test.

        return lowResVolumes, highResVolumes, t1, t2, t3
        # correct.


# add on 2022.4.10.
# define crop function, which crops a volume ([batch size, channels, depth, height, width]) center to
# the cropVolume_shape ([batch size, channels, depth, height, width]).
def crop(volume, cropVolume_shape):
    middle_depth = volume.shape[2] // 2
    middle_height = volume.shape[3] // 2
    middle_width = volume.shape[4] // 2

    # compute cropped depth.
    start_depth = middle_depth - cropVolume_shape[2] // 2
    end_depth = start_depth + cropVolume_shape[2]

    # compute cropped height.
    start_height = middle_height - cropVolume_shape[3] // 2
    end_height = start_height + cropVolume_shape[3]

    # compute cropped width.
    start_width = middle_width - cropVolume_shape[4] // 2
    end_width = start_width + cropVolume_shape[4]

    cropVolume = volume[:, :, start_depth:end_depth, start_height:end_height, start_width:end_width]
    return cropVolume



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


class GeneratorSA(nn.Module):
    """
    GeneratorSA
    input:
        z: lat
    """
    def __init__(self, in_channel=3, out_channel=3, inner_channel=64,  attn=True):
        super().__init__()
        self.attn = attn
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.inner_channel = inner_channel
        # layer 1 turn 100 dims -> 512 dims, size 1 -> 3
        self.l1 = nn.Sequential(
            nn.utils.spectral_norm(nn.ConvTranspose3d(self.in_channel, self.inner_channel, 3)),
        )


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


# class DiscriminatorSpatial(nn.Module):
#     def makeEachLayer(self, input_channel, output_channel, final_layer=False):
#         if not final_layer:
#             return nn.Sequential(
#                 nn.utils.spectral_norm(nn.Conv3d(input_channel, output_channel, kernel_size=4, stride=2, padding=1)),
#                 nn.LeakyReLU(0.2, inplace=True),
#             )
#         else:
#             return nn.Sequential(
#                 nn.Conv3d(input_channel, output_channel, kernel_size=self.filter_size)
#             )
#
#     def __init__(self, input_channel=1, output_channel=1, filter_size=(3, 3, 8)):
#         super(DiscriminatorSpatial, self).__init__()
#         self.filter_size = filter_size
#         hidden_channels = 32
#
#         self.upfeature = FeatureMapBlock(3, hidden_channels)
#         self.contract1 = ContractingBlock(hidden_channels, use_n=False, kernel_size=(4, 4, 4), activation='lrelu')
#         self.contract2 = ContractingBlock(hidden_channels * 2, kernel_size=(4, 4, 4), activation='lrelu')
#         self.contract3 = ContractingBlock(hidden_channels * 4, kernel_size=(4, 4, 4), activation='lrelu')
#         self.sa1 = Self_Attn(hidden_channels * 8)
#         self.downfeature = FeatureMapBlock(hidden_channels * 8, 1)
#
#     def forward(self, x):
#         self.x0 = x0 = self.upfeature(x)
#         self.x1 = x1 = self.contract1(x0)
#         self.x2 = x2 = self.contract2(x1)
#         self.x3 = x3 = self.contract3(x2)
#         self.x4 = x4 = self.sa1(x3)
#         self.xn = xn = self.downfeature(x4)
#
#         return xn
class DiscriminatorSpatial(nn.Module):
    def makeEachLayer(self, input_channel, output_channel, final_layer=False):
        if not final_layer:
            return nn.Sequential(
                nn.utils.spectral_norm(nn.Conv3d(input_channel, output_channel, kernel_size=4, stride=2, padding=1)),
                nn.LeakyReLU(0.2, inplace=True),
            )
        else:
            return nn.Sequential(
                nn.Conv3d(input_channel, output_channel, kernel_size=self.filter_size)
            )

    def __init__(self, input_channel=3, output_channel=1, filter_size=(3, 8, 8)):
        super(DiscriminatorSpatial, self).__init__()
        self.filter_size = filter_size
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.disc = nn.Sequential(
            # 该卷积的效果是每执行一层，每个维度的尺寸减半
            self.makeEachLayer(self.input_channel, 64),
            self.makeEachLayer(64, 128),
            self.makeEachLayer(128, 256),
            self.makeEachLayer(256, 512),
            Self_Attn(512),
            self.makeEachLayer(512, self.output_channel, final_layer=True),
            # 为了获得1*1*1的输出，计算得出Kernel_size
            #self.makeEachLayer(512, 1, final_layer=True),
        )

    def forward(self, highResVolume):
        x = highResVolume
        self.intFeat0 = x = self.disc[0](x)
        self.intFeat1 = x = self.disc[1](x)
        self.intFeat2 = x = self.disc[2](x)
        self.intFeat3 = x = self.disc[3](x)
        self.intFeat4 = x = self.disc[4](x)
        dis_pred = self.disc(highResVolume)
        return dis_pred.view(len(dis_pred), -1)

class DiscriminatorTemporal(nn.Module):
    def makeEachLayer(self, input_channel, output_channel):
        return nn.Sequential(
            nn.utils.spectral_norm(nn.Conv3d(input_channel, output_channel, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def __init__(self, input_channel=3, output_channel=1, filter_size=(3, 8, 8)):
        super(DiscriminatorTemporal, self).__init__()
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.disc = nn.Sequential(
            self.makeEachLayer(self.input_channel, 64),
            self.makeEachLayer(64, 128),
            self.makeEachLayer(128, 256),
            self.makeEachLayer(256, 512),
            nn.Conv3d(512, self.output_channel, kernel_size=1)
        )



    def forward(self, highResVolume):
        x = highResVolume
        self.intFeat0 = x = self.disc[0](x)
        self.intFeat1 = x = self.disc[1](x)
        self.intFeat2 = x = self.disc[2](x)
        self.intFeat3 = x = self.disc[3](x)

        dis_pred = self.disc(highResVolume)
        return dis_pred.view(len(dis_pred), -1)

def torch_wasserstein_loss(tensor_a,tensor_b):
    #Compute the first Wasserstein distance between two 1D distributions.
    return(torch_cdf_loss(tensor_a,tensor_b,p=1))

def torch_energy_loss(tensor_a,tensor_b):
    # Compute the energy distance between two 1D distributions.
    return((2**0.5)*torch_cdf_loss(tensor_a,tensor_b,p=2))

def torch_cdf_loss(tensor_a,tensor_b,p=1):
    # last-dimension is weight distribution
    # p is the norm of the distance, p=1 --> First Wasserstein Distance
    # to get a positive weight with our normalized distribution
    # we recommend combining this loss with other difference-based losses like L1

    # normalize distribution, add 1e-14 to divisor to avoid 0/0
    tensor_a = tensor_a / (torch.sum(tensor_a, dim=-1, keepdim=True) + 1e-14)
    tensor_b = tensor_b / (torch.sum(tensor_b, dim=-1, keepdim=True) + 1e-14)
    # make cdf with cumsum
    cdf_tensor_a = torch.cumsum(tensor_a,dim=-1)
    cdf_tensor_b = torch.cumsum(tensor_b,dim=-1)

    # choose different formulas for different norm situations
    if p == 1:
        cdf_distance = torch.sum(torch.abs((cdf_tensor_a-cdf_tensor_b)),dim=-1)
    elif p == 2:
        cdf_distance = torch.sqrt(torch.sum(torch.pow((cdf_tensor_a-cdf_tensor_b),2),dim=-1))
    else:
        cdf_distance = torch.pow(torch.sum(torch.pow(torch.abs(cdf_tensor_a-cdf_tensor_b),p),dim=-1),1/p)

    cdf_loss = cdf_distance.mean()
    return cdf_loss
def weights_init(m):
    if isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
        nn.init.normal_(m.weight, mean, std)
    if isinstance(m, nn.BatchNorm3d):
        nn.init.normal_(m.weight, mean, std)
        nn.init.constant_(m.bias, 0)

def get_index(ssim, psnr):

    if ssim <= 0.9:
        index_s = 5 / (1 - ssim)
    elif ssim <= 0.95:
        index_s = 2.5 / (1 - ssim) + 25
    elif ssim <= 0.99:
        index_s = 1 / (1 - ssim) + 55
    else:
        index_s = 170

    if psnr <= 30:
        index_p = 3 * psnr
    elif psnr <= 35:
        index_p = 3 * 30 + 10 * (psnr - 30)
    elif psnr <= 40:
        index_p = 3 * 30 + 10 * 5 + 20 * (psnr - 35)
    elif psnr > 40:
        index_p = 3 * 30 + 10 * 5 + 20 * 5 + 50 * (psnr - 40)
    return index_s + index_p
version = "v9_6"
# given cur_step, save its model.
def saveModel(cur_step):
    if save_model:
        visfileName = "%s%s/srg_%s.pt" % (dataSavePath, version, version)
        torch.save({'gen': gen}, visfileName)
        fileName = "%s%s/srg_%s_%d.pth" % (dataSavePath, version, version, cur_step)
        torch.save({'gen': gen.state_dict(),
                    'gen_opt': gen_opt.state_dict(),
                    # 'discS': discS.state_dict(),
                    # 'discS_opt': discS_opt.state_dict(),
                    # 'discT': discT.state_dict(),
                    # 'discT_opt': discT_opt.state_dict()
                    }, fileName)  # , _use_new_zipfile_serialization=False)


# 0. declare variables.

# 修改地址
device = torch.device('cuda:0')  # device = 'cuda'
server = 'dell'
dataset_name = "hurricane_wind"
if server == 'dell':
    if dataset_name == "square_cylinder":
        dataSourcePath = 'SC/train/'
        dataSavePath = 'SC/saved/'
    elif dataset_name == "p_square_cylinder":
        dataSourcePath = 'product_ssrtvd/SquareCylinder/'
        dataSavePath = 'product_ssrtvd/SquareCylinder/saved/'
    elif dataset_name == "ionization":
        dataSourcePath = 'ionization_ab_H/train/'
        dataSavePath = 'ionization_ab_H/saved/'
    elif dataset_name == "p_ionization":
        dataSourcePath = 'product_ssrtvd/ionization_ab_H/'
        dataSavePath = 'product_ssrtvd/ionization_ab_H/saved/'
    elif dataset_name == "hurricane_wind":
        dataSourcePath = 'hurricane_wind/train/'
        dataSavePath = 'hurricane_wind/saved/'

    elif dataset_name == "hurricane_qsnow":
        dataSourcePath = 'hur_qsnow_048/train/'
        dataSavePath = 'hur_qsnow_048/saved/'
    elif dataset_name == "p_hurricane_qsnow":
        dataSourcePath = 'product_ssrtvd/hurricane_QSNOW/'
        dataSavePath = 'product_ssrtvd/hurricane_QSNOW/saved/'
    elif dataset_name == "viscous":
        dataSourcePath = "viscous096/train/"
        dataSavePath = "viscous096/saved/"
    elif dataset_name == "halfcylinder6400":
        dataSourcePath = "halfcylinder3d-Re6400_am/"
        dataSavePath = "halfcylinder3d-Re6400_am/saved/"
    print(server)
    print(dataSourcePath)


fileStartVal = 1
fileIncrement = 1
constVal = 1
# 修改采样大小
cropScaleFactor = (0.5, 0.5, 0.5)  # [depth, height, width].
dsamScaleFactor = (0.25, 0.25, 0.25)  # [depth, height, width].
# 修改尺寸大小
dim = (48, 240, 240)  # [depth, height, width].

if dataset_name == "hurricane_wind":
    dim = (48, 240, 240)
    filter_size = (1, 7, 7)
    dataInt = 2
    totalTimesteps = 48
    print(dataset_name)
    print(dim)
    batchSize = 5
elif dataset_name == "hurricane_qsnow":
    dim = (48, 240, 240)
    filter_size = (1, 7, 7)
    dataInt = 2
    totalTimesteps = 48
    print(dataset_name)
    print(dim)
    batchSize = 5
elif dataset_name == "p_hurricane_qsnow":
    dim = (50, 250, 250)
    filter_size = (1, 7, 7)
    dataInt = 2
    totalTimesteps = 48
    print(dataset_name)
    print(dim)
    batchSize = 5
    cropScaleFactor = (0.48, 0.48, 0.48)  # [depth, height, width].
elif dataset_name == "halfcylinder6400":
    dim = (40, 120, 320)
    fileStartVal = 0
    fileIncrement = 0.1
    filter_size = (1, 2, 6)
    dataInt = 10
    totalTimesteps = 151
    print(dataset_name)
    print(dim)
    batchSize = 5
    cropScaleFactor = (0.8, 0.4, 0.4)  # [depth, height, width].

elif dataset_name == "square_cylinder" or dataset_name == "p_square_cylinder":
    dim = (48, 64, 192)
    fileStartVal = 8
    fileIncrement = 40
    filter_size = (1, 2, 6)
    dataInt = 4
    totalTimesteps = 102
    print(dataset_name)
    print(dim)
    batchSize = 5
elif dataset_name == "ionization":
    dim = (96, 96, 144)
    fileStartVal = 0
    fileIncrement = 1
    filter_size = (3, 3, 4)
    dataInt = 4
    totalTimesteps = 100
    print(dataset_name)
    print(dim)
    batchSize = 5
elif dataset_name == "p_ionization":
    dim = (100, 100, 150)
    fileStartVal = 0
    fileIncrement = 1
    filter_size = (3, 3, 4)
    dataInt = 4
    totalTimesteps = 100
    print(dataset_name)
    print(dim)
    batchSize = 5
    cropScaleFactor = (0.5, 0.5, 0.5)  # [depth, height, width].
    dsamScaleFactor = (0.2, 0.2, 0.2)  # [depth, height, width].
elif dataset_name == "viscous":
    dim = (96, 96, 96)
    fileStartVal = 0
    fileIncrement = 1
    filter_size = (3, 3, 3)
    dataInt = 4
    totalTimesteps = 121
    cropScaleFactor = (0.5, 0.5, 0.5)  # [depth, height, width].
    dsamScaleFactor = (0.25, 0.25, 0.25)  # [depth, height, width].
    print(dataset_name)
    print(dim)
    batchSize = 5

trainsetRatio = 0.7  # according to deep learning specialization, if you have a small collection of data, then use 70%/30% for train/test.
nTrainTimesteps = round(totalTimesteps * trainsetRatio)  # nTrainTimesteps=34.


dim_highResVolumes = (int(dim[0] * cropScaleFactor[0]),
                      int(dim[1] * cropScaleFactor[1]),
                      int(dim[2] * cropScaleFactor[2]))
dim_lowResVolumes = (int(dim_highResVolumes[0] * dsamScaleFactor[0]),
                     int(dim_highResVolumes[1] * dsamScaleFactor[1]),
                     int(dim_highResVolumes[2] * dsamScaleFactor[2]))
float32DataType = np.float32
myRandCrop3D = MyRandomCrop3D3(volume_sz=(1, dim[0], dim[1], dim[2]),
                               cropVolume_sz=dim_highResVolumes)
# 7.6 修改b 5->4


BCE_criterion = nn.BCEWithLogitsLoss()

mseCriterion = nn.MSELoss()
smoothL1Loss = nn.SmoothL1Loss()
# lambda1 = 0.001 #according to paper.
# lambda2 = 1 #according to paper.
# lambda3 = 0.05  #according to paper.
lambda1 = 0.001 # adv
lambda2 = 1  # content
lambda3 = 0.05    # fea
# add on 2022.4.1.
# 7.5 0.0002 -> 0.00002
lr = 0.0002
beta1 = 0.5
beta2 = 0.999

mean = 0.0
std = 0.02
# 修改filter size

# 修改 epoch
# n_epochs = 400
n_epochs = 4000


lowResVolumes_channels = 3
highResVolumes_channels = 3
save_model = True

trainDataset = VolumesDataset3(
    dataSourcePath=dataSourcePath, nTimesteps=nTrainTimesteps,
    dim=dim, dim_highResVolumes=dim_highResVolumes, dim_lowResVolumes=dim_lowResVolumes,
    dsamScaleFactor=dsamScaleFactor,
    fileStartVal=fileStartVal, fileIncrement=fileIncrement, constVal=constVal, float32DataType=float32DataType,
    transform=myRandCrop3D, dataInt=dataInt
)
# test.
# print(f"trainDataset len: {trainDataset.__len__()}")
# trainDataset.__getitem__(15)
# test.


trainDataloader = DataLoader(trainDataset, batch_size=batchSize, shuffle=True)
# correct.


# nb = 23 => nb = 16
# 2. initialize Generator, Discriminator, their optimizers, and their weights.
gen = GeneratorResNet(3, 3, 64, 16).to(device)
gen_opt = torch.optim.Adam(gen.parameters(), lr=lr, betas=(beta1, beta2))
scheduler_gen = torch.optim.lr_scheduler.StepLR(gen_opt, step_size=2000, gamma=0.99)

discS = DiscriminatorSpatial(3, 1, filter_size).to(device)
discS_opt = torch.optim.Adam(discS.parameters(), lr=lr, betas=(beta1, beta2))
scheduler_dis = torch.optim.lr_scheduler.StepLR(discS_opt, step_size=2000, gamma=0.99)
# Feel free to change pretrained=False, if you's like to train the model from scratch.
# 修改名称用于标识
pretrained = False  # True
if pretrained:
    loaded_state = torch.load("ionization_ab_H/crop=0.5/u_4x/srgan_v0202_train_1120.pth")
    gen.load_state_dict(loaded_state["gen"])
    gen_opt.load_state_dict(loaded_state["gen_opt"])
    discS.load_state_dict(loaded_state["discS"])
    discS_opt.load_state_dict(loaded_state["discS_opt"])

    print("Loaded! v0202 1120")
else:
    gen = gen.apply(weights_init)
    discS = discS.apply(weights_init)


# 3. training.
def trainGAN():

    mean_discT_loss = 0
    display_step = 140  # 显示损失步骤
    display_step2 = 600  # 保存模型步骤
    display_epoch = 5
    display_epoch_2 = 100
    # 判断梯度消失
    endloop = False
    for epoch in tqdm(range(n_epochs)):
        epoch_psnr = 0
        epoch_ssim = 0
        curStep = 0
        mean_gen_loss = 0
        mean_discS_loss = 0
        mean_adv_loss = 0
        mean_content_loss = 0
        mean_feature_loss = 0
        # Dataloader returns the batches.
        if endloop:
            break
        # real_lowResVolumes: [5, 3, 6, 8, 24], real_highResVolumes: [5, 3, 24, 32, 96].
        for real_lowResVolumes, real_highResVolumes, t1, t2, t3 in trainDataloader:  # tqdm(trainDataloader):

            real_lowResVolumes = real_lowResVolumes.to(device)
            real_highResVolumes = real_highResVolumes.to(device)

            #print(real_lowResVolumes.shape)
            #print(real_highResVolumes.shape)
            #print(real_highResVolumes[])
            ################################################################
            discS_opt.zero_grad()  # zero out the gradient before backpropagation.
            fake1_highResVolumes = gen(real_lowResVolumes)
            #print(fake1_highResVolumes.shape)
            discS_fake1_pred = discS(fake1_highResVolumes.detach())
            #print(discS_fake1_pred.shape)
            #discS_fake1_loss = torch.sum((discS_fake1_pred - 0) ** 2)
            discS_fake1_loss = mseCriterion(discS_fake1_pred, torch.zeros_like(discS_fake1_pred))
            discS_real1_pred = discS(real_highResVolumes)
            #discS_real1_loss = torch.sum((discS_real1_pred - 1) ** 2)
            discS_real1_loss = mseCriterion(discS_real1_pred, torch.ones_like(discS_real1_pred))
            discS_loss = 0.5 * (discS_fake1_loss + discS_real1_loss)
            discS_loss.backward(retain_graph=True)
            discS_opt.step()
            scheduler_dis.step()
            #####
            #torch_cdf_loss(discS_fake1_pred, discS_real1_pred)
            #####
            #mean_discS_loss += discS_loss.item()

            discS_opt.zero_grad()  # zero out the gradient before backpropagation.
            fake1_highResVolumes = gen(real_lowResVolumes)
            discS_fake1_pred = discS(fake1_highResVolumes.detach())
            # print(discS_fake1_pred.shape)
            #discS_fake1_loss = torch.sum((discS_fake1_pred - 0) ** 2)
            discS_fake1_loss = mseCriterion(discS_fake1_pred, torch.zeros_like(discS_fake1_pred))
            # get fake internal features of disc.
            discS_fake_intFeat0 = discS.intFeat0.detach()
            discS_fake_intFeat1 = discS.intFeat1.detach()
            discS_fake_intFeat2 = discS.intFeat2.detach()
            discS_fake_intFeat3 = discS.intFeat3.detach()
            discS_fake_intFeat4 = discS.intFeat4.detach()
            print(discS_fake_intFeat0.shape)
            print(discS_fake_intFeat1.shape)
            print(discS_fake_intFeat2.shape)
            print(discS_fake_intFeat3.shape)
            print(discS_fake_intFeat4.shape)
            print(discS_fake1_pred.shape)
            # add on 2022.4.11.
            # (3.1.2)compute disc_real_loss.
            discS_real1_pred = discS(real_highResVolumes)
            #discS_real1_loss = torch.sum((discS_real1_pred - 1) ** 2)
            discS_real1_loss = mseCriterion(discS_real1_pred, torch.ones_like(discS_real1_pred))

            discS_real_intFeat0 = discS.intFeat0.detach()
            discS_real_intFeat1 = discS.intFeat1.detach()
            discS_real_intFeat2 = discS.intFeat2.detach()
            discS_real_intFeat3 = discS.intFeat3.detach()
            discS_real_intFeat4 = discS.intFeat4.detach()
            # add on 2022.4.11.
            discS_loss = (discS_fake1_loss + discS_real1_loss) / 2
            discS_loss.backward(retain_graph=True)
            discS_opt.step()
            scheduler_dis.step()
            mean_discS_loss += discS_loss.item()
            # (3.2)update Generator.
            gen_opt.zero_grad()

            # Adversarial loss
            fake2_highResVolumes = gen(real_lowResVolumes)  # fake2_highResVolumes: [5, 4, 24, 32, 96].
            discS_fake2_pred = discS(fake2_highResVolumes[:, :, :, :, :])
            #gen_discS_loss = torch.sum((discS_fake2_pred - 1) ** 2)
            gen_discS_loss = mseCriterion(discS_fake2_pred, torch.ones_like(discS_fake2_pred))
            gen_adversarial_loss = gen_discS_loss
            # print(fake2_highResVolumes.shape)
            # print(real_highResVolumes.shape)
            # Content loss
            #content_loss_discS = mseCriterion(fake2_highResVolumes, real_highResVolumes)
            content_loss_discS = smoothL1Loss(fake2_highResVolumes, real_highResVolumes)
            mean_content_loss += content_loss_discS.item()
            # Feature loss+

            # 新损失--

            featureLoss_discS = 0
            intFeatRealGroupS = [discS_real_intFeat0, discS_real_intFeat1,
                                 discS_real_intFeat2, discS_real_intFeat3, discS_real_intFeat4]
            intFeatFakeGroupS = [discS_fake_intFeat0, discS_fake_intFeat1,
                                 discS_fake_intFeat2, discS_fake_intFeat3, discS_fake_intFeat4]
            #Nk = 1
            for i in range(5):
                #for j in range(5):
                    #Nk = Nk * intFeatRealGroupS[i].shape[j]
                featureLoss_discS += mseCriterion(intFeatRealGroupS[i], intFeatFakeGroupS[i])
                #Nk = 1
            mean_feature_loss += featureLoss_discS.item()
            gen_loss = lambda1 * gen_adversarial_loss + lambda2 * content_loss_discS + lambda3 * featureLoss_discS
            # (vii)update gradients.
            gen_loss.backward()
            # (viii)update Generator's parameters e.g., weights, bias.
            gen_opt.step()
            scheduler_gen.step()
            mean_gen_loss += gen_loss.item()
            ###v8 20 修改了advloss 和conten loss
            # correct.

            # (5.4)increase curStep.
            curStep += 1
            if mean_gen_loss < 0.00000001:
                print("mean_gen_loss")
                print(mean_gen_loss)
                print("梯度消失")
                endloop = True
                break
            elif mean_gen_loss > 10000000:
                print("mean_gen_loss")
                print(mean_gen_loss)
                print("梯度爆炸")
                print(gen_adversarial_loss.item())
                print(content_loss_discS.item())
                print(featureLoss_discS.item())
                print(discS_loss.item())
                ssim_loss = pytorch_ssim.SSIM3D(window_size=11)
                ll = ssim_loss(fake2_highResVolumes, real_highResVolumes).item()
                print(f"ssim{ll}")
                endloop = True
                break

            # (3.4)for every display_step, do something here.
            if epoch % display_epoch == 0:
                # for every display_step:
                # clear mean_gen_loss, mean_discS_loss, mean_discT_loss.
                ssim_loss = pytorch_ssim.SSIM3D(window_size=11)
                ll = ssim_loss(fake2_highResVolumes, real_highResVolumes).item()
                #print(f"ssim: {ll}")
                ############## 计算PSNR
                maxvalue = 1
                mse = mseCriterion(fake2_highResVolumes, real_highResVolumes)
                mse = mse.cpu()
                mse = mse.detach().numpy()
                psnr = 10 * np.log10((maxvalue ** 2) / mse)
                #print(f"psnr: {psnr}")
                #total_psnr += psnr
                ##############
                epoch_psnr += psnr
                epoch_ssim += ll

        if epoch % display_epoch == 0:
            epoch_ssim = round(epoch_ssim / curStep, 4)
            epoch_psnr = round(epoch_psnr / curStep, 4)
            mean_gen_loss = round(mean_gen_loss / curStep, 6)
            mean_discS_loss = round(mean_discS_loss / curStep, 5)
            mean_content_loss = round(mean_content_loss / curStep, 5)
            mean_feature_loss = round(mean_feature_loss / curStep, 5)
            index = get_index(epoch_ssim, epoch_psnr)
            index = round(index, 2)
            #print("!")
            print(
                f"epoch {epoch}: index={index}, g_loss={mean_gen_loss}, "
                f"dS_loss={mean_discS_loss}, cont_loss={mean_content_loss}, "
                f"fea_loss={mean_feature_loss}, "
                f"ssim={epoch_ssim}, psnr={epoch_psnr}")
            # Feel free to change save_model=True, if you'd like to save the model.

            # save fake2_highResVolumes ([5, 3, 24, 32, 96]) as .raw file, for visualizing the evolution.
            # saveRawFile2(curStep, 't1', fake2_highResVolumes[0, 0, :, :, :])
            # saveRawFile2(curStep, 't2', fake2_highResVolumes[0, 1, :, :, :])
            # saveRawFile2(curStep, 't3', fake2_highResVolumes[0, 2, :, :, :])
        # if (epoch % display_epoch_2 == 0 and epoch >= 4000) or epoch == 10:
        if (epoch % display_epoch_2 == 0) or epoch == 10:
            saveModel(epoch)



# compute GAN training time. Refer to:
# (i)https://discuss.pytorch.org/t/how-to-measure-time-in-pytorch/26964.
# (ii)https://discuss.pytorch.org/t/how-to-measure-execution-time-in-pytorch/111458
# waits for everything to finish running.
torch.cuda.synchronize()

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()
trainGAN()
end.record()

# waits for everything to finish running.
torch.cuda.synchronize()
print(f"training time consumed: {start.elapsed_time(end)}")

# 4. save final model.
saveModel(n_epochs)