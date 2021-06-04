import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import torch.nn.init as init
import numpy as np
from models.Blocks import ChannelSELayer3D

class DownSampling(nn.Module):
    # 3x3x3 convolution and 1 padding as default
    def __init__(self, inChans, outChans, stride=2, kernel_size=3, padding=1, dropout_rate=None):
        super(DownSampling, self).__init__()

        self.dropout_flag = False
        self.conv1 = nn.Conv3d(in_channels=inChans,
                               out_channels=outChans,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=padding,
                               bias=False)
        if dropout_rate is not None:
            self.dropout_flag = True
            self.dropout = nn.Dropout3d(dropout_rate, inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        if self.dropout_flag:
            out = self.dropout(out)
        return out


class EncoderBlock(nn.Module):
    '''
    Encoder block
    '''

    def __init__(self, inChans, outChans, stride=1, padding=1, num_groups=8, activation="relu",
                 normalizaiton="group_normalization"):
        super(EncoderBlock, self).__init__()

        if normalizaiton == "group_normalization":
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=inChans)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=inChans)
        if activation == "relu":
            self.actv1 = nn.ReLU(inplace=True)
            self.actv2 = nn.ReLU(inplace=True)
        elif activation == "elu":
            self.actv1 = nn.ELU(inplace=True)
            self.actv2 = nn.ELU(inplace=True)
        self.conv1 = nn.Conv3d(in_channels=inChans, out_channels=outChans, kernel_size=3, stride=stride,
                               padding=padding)
        self.conv2 = nn.Conv3d(in_channels=inChans, out_channels=outChans, kernel_size=3, stride=stride,
                               padding=padding)

    def forward(self, x):
        residual = x

        out = self.norm1(x)
        out = self.actv1(out)
        out = self.conv1(out)
        out = self.norm2(out)
        out = self.actv2(out)
        out = self.conv2(out)

        out += residual

        return out


class LinearUpSampling(nn.Module):
    '''
    Trilinear interpolate to upsampling
    '''

    def __init__(self, inChans, outChans, scale_factor=2, mode="trilinear", align_corners=True):
        super(LinearUpSampling, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners
        self.conv1 = nn.Conv3d(in_channels=inChans, out_channels=outChans, kernel_size=1)
        self.conv2 = nn.Conv3d(in_channels=inChans, out_channels=outChans, kernel_size=1)

    def forward(self, x, skipx=None):
        out = self.conv1(x)
        # out = self.up1(out)
        out = nn.functional.interpolate(out, scale_factor=self.scale_factor, mode=self.mode,
                                        align_corners=self.align_corners)

        if skipx is not None:
            out = torch.cat((out, skipx), 1)
            out = self.conv2(out)

        return out


class DecoderBlock(nn.Module):
    '''
    Decoder block
    '''

    def __init__(self, inChans, outChans, stride=1, padding=1, num_groups=8, activation="relu",
                 normalizaiton="group_normalization"):
        super(DecoderBlock, self).__init__()

        if normalizaiton == "group_normalization":
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=outChans)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=outChans)
        if activation == "relu":
            self.actv1 = nn.ReLU(inplace=True)
            self.actv2 = nn.ReLU(inplace=True)
        elif activation == "elu":
            self.actv1 = nn.ELU(inplace=True)
            self.actv2 = nn.ELU(inplace=True)
        self.conv1 = nn.Conv3d(in_channels=inChans, out_channels=outChans, kernel_size=3, stride=stride,
                               padding=padding)
        self.conv2 = nn.Conv3d(in_channels=outChans, out_channels=outChans, kernel_size=3, stride=stride,
                               padding=padding)

    def forward(self, x):
        residual = x

        out = self.norm1(x)
        out = self.actv1(out)
        out = self.conv1(out)
        out = self.norm2(out)
        out = self.actv2(out)
        out = self.conv2(out)

        out += residual

        return out


class OutputTransition(nn.Module):
    '''
    Decoder output layer
    output the prediction of segmentation result
    '''

    def __init__(self, inChans, outChans):
        super(OutputTransition, self).__init__()

        self.conv1 = nn.Conv3d(in_channels=inChans, out_channels=outChans, kernel_size=1)
        self.actv1 = torch.nn.Softmax(dim=1)

    def forward(self, x):
        return self.actv1(self.conv1(x))


class VDResampling(nn.Module):
    '''
    Variational Auto-Encoder Resampling block
    '''

    def __init__(self, inChans=256, outChans=256, dense_features=(8, 8, 8), stride=2, kernel_size=3, padding=1,
                 activation="relu", normalizaiton="group_normalization"):
        super(VDResampling, self).__init__()

        midChans = int(inChans / 2)
        self.dense_features = dense_features
        if normalizaiton == "group_normalization":
            self.gn1 = nn.GroupNorm(num_groups=8, num_channels=inChans)
        if activation == "relu":
            self.actv1 = nn.ReLU(inplace=True)
            self.actv2 = nn.ReLU(inplace=True)
        elif activation == "elu":
            self.actv1 = nn.ELU(inplace=True)
            self.actv2 = nn.ELU(inplace=True)
        self.conv1 = nn.Conv3d(in_channels=inChans, out_channels=inChans, kernel_size=kernel_size, stride=stride,
                               padding=padding)
        self.amp = nn.AdaptiveAvgPool3d(1)
        self.dense1 = nn.Linear(in_features=256,
                                out_features=inChans)
        self.dense2 = nn.Linear(in_features=midChans,
                                out_features=midChans * dense_features[0] * dense_features[1] * dense_features[2])
        self.up0 = LinearUpSampling(midChans, outChans)

    def forward(self, x):
        out = self.gn1(x)
        out = self.actv1(out)
        out = self.conv1(out)

        out = self.amp(out)
        out = out.view(-1, self.num_flat_features(out))
        out_vd = self.dense1(out)
        distr = out_vd
        out = VDraw(out_vd)
        out = self.dense2(out)
        out = self.actv2(out)
        out = out.view((1, 128, self.dense_features[0], self.dense_features[1], self.dense_features[2]))
        out = self.up0(out)

        return out, distr

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s

        return num_features


def VDraw(x):
    # Generate a Gaussian distribution with the given mean(128-d) and std(128-d)
    return torch.distributions.Normal(x[:, :128], x[:, 128:]).sample()


class VDecoderBlock(nn.Module):
    '''
    Variational Decoder block
    '''

    def __init__(self, inChans, outChans, activation="relu", normalizaiton="group_normalization", mode="trilinear"):
        super(VDecoderBlock, self).__init__()

        self.up0 = LinearUpSampling(inChans, outChans, mode=mode)
        self.block = DecoderBlock(outChans, outChans, activation=activation, normalizaiton=normalizaiton)

    def forward(self, x):
        out = self.up0(x)
        out = self.block(out)

        return out


class VAE(nn.Module):
    '''
    Variational Auto-Encoder : to group the features extracted by Encoder
    '''

    def __init__(self, inChans=256, outChans=4, dense_features=(8, 8, 8), activation="relu",
                 normalizaiton="group_normalization", mode="trilinear"):
        super(VAE, self).__init__()

        self.vd_resample = VDResampling(inChans=inChans, outChans=inChans, dense_features=dense_features)
        self.vd_block2 = VDecoderBlock(inChans, inChans // 2)
        self.vd_block1 = VDecoderBlock(inChans // 2, inChans // 4)
        self.vd_block0 = VDecoderBlock(inChans // 4, inChans // 8)
        self.vd_end = nn.Conv3d(inChans // 8, outChans, kernel_size=1)

    def forward(self, x):
        out, distr = checkpoint(self.vd_resample,x) #256*16*16*16
        out = checkpoint(self.vd_block2,out)        #128*32*32*32
        out = checkpoint(self.vd_block1,out)        #64*64*64*64
        out = checkpoint(self.vd_block0,out)       #32*128*128*128
        out = checkpoint(self.vd_end,out)        #4*128*128*128

        return out, distr


class NvNet(nn.Module):
    def __init__(self):
        super(NvNet, self).__init__()

        # some critical parameters
        self.inChans = 4
        self.input_shape = [1,4,128,128,128]
        self.seg_outChans = 4
        self.activation = "relu"
        self.normalizaiton = "group_normalization"
        self.mode = "trilinear"

        # Encoder Blocks
        self.in_conv0 = DownSampling(inChans=self.inChans, outChans=32, stride=1, dropout_rate=0.2)
        self.en_block0 = EncoderBlock(32, 32, activation=self.activation, normalizaiton=self.normalizaiton)
        self.en_down1 = DownSampling(32, 64)
        self.en_block1_0 = EncoderBlock(64, 64, activation=self.activation, normalizaiton=self.normalizaiton)
        self.en_block1_1 = EncoderBlock(64, 64, activation=self.activation, normalizaiton=self.normalizaiton)
        self.en_down2 = DownSampling(64, 128)
        self.en_block2_0 = EncoderBlock(128, 128, activation=self.activation, normalizaiton=self.normalizaiton)
        self.en_block2_1 = EncoderBlock(128, 128, activation=self.activation, normalizaiton=self.normalizaiton)
        self.en_down3 = DownSampling(128, 256)
        self.en_block3_0 = EncoderBlock(256, 256, activation=self.activation, normalizaiton=self.normalizaiton)
        self.en_block3_1 = EncoderBlock(256, 256, activation=self.activation, normalizaiton=self.normalizaiton)
        self.en_block3_2 = EncoderBlock(256, 256, activation=self.activation, normalizaiton=self.normalizaiton)
        self.en_block3_3 = EncoderBlock(256, 256, activation=self.activation, normalizaiton=self.normalizaiton)

        # Decoder Blocks
        self.de_up2 = LinearUpSampling(256, 128, mode=self.mode)
        self.de_block2 = DecoderBlock(128, 128, activation=self.activation, normalizaiton=self.normalizaiton)
        self.de_up1 = LinearUpSampling(128, 64, mode=self.mode)
        self.de_block1 = DecoderBlock(64, 64, activation=self.activation, normalizaiton=self.normalizaiton)
        self.de_up0 = LinearUpSampling(64, 32, mode=self.mode)
        self.de_block0 = DecoderBlock(32, 32, activation=self.activation, normalizaiton=self.normalizaiton)
        self.de_end = OutputTransition(32, self.seg_outChans)

        # Variational Auto-Encoder
        self.dense_features = (self.input_shape[2] // 16, self.input_shape[3] // 16, self.input_shape[4] // 16)
        self.vae = VAE(256, outChans=self.inChans, dense_features=self.dense_features)

    def forward(self, x):
        inputs = x + torch.zeros(1, dtype=x.dtype, device=x.device, requires_grad=True)
        out_init = checkpoint(self.in_conv0,inputs)
        out_en0 = checkpoint(self.en_block0,out_init)
        out_en1 = checkpoint(self.en_block1_1,checkpoint(self.en_block1_0,self.en_down1(out_en0)))
        out_en2 = checkpoint(self.en_block2_1,checkpoint(self.en_block2_0,self.en_down2(out_en1)))
        out_en3 = checkpoint(self.en_block3_3,
            checkpoint(self.en_block3_2,
                checkpoint(self.en_block3_1,
                        self.en_down3(out_en2))))

        out_de2 = checkpoint(self.de_block2,self.de_up2(out_en3, out_en2))
        out_de1 = checkpoint(self.de_block1,self.de_up1(out_de2, out_en1))
        out_de0 = checkpoint(self.de_block0,self.de_up0(out_de1, out_en0))
        out_end = checkpoint(self.de_end,out_de0)


        out_vae, out_distr = self.vae(out_en3)

        return out_end,out_vae,out_distr


'''
    ResUNet en--conv  MTAN  
    1. rebuild
    2. Segment
    3. Classification
'''

'''
    Ordinary UNet Conv Block
'''
class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=3, activation=F.leaky_relu):
        super(UNetConvBlock, self).__init__()
        self.conv = nn.Conv3d(in_size, out_size, kernel_size, stride=1, padding=1)
        self.gn = nn.GroupNorm(num_groups=8,num_channels=out_size)
        self.conv2 = nn.Conv3d(out_size, out_size, kernel_size, stride=1, padding=1)
        self.gn2 = nn.GroupNorm(num_groups=8,num_channels=out_size)
        self.activation = activation


        init.xavier_uniform_(self.conv.weight, gain = np.sqrt(2.0))
        init.constant_(self.conv.bias,0)
        init.xavier_uniform_(self.conv2.weight, gain = np.sqrt(2.0))
        init.constant_(self.conv2.bias,0)
    def forward(self, x):
        out = self.activation(self.gn(self.conv(x)))
        out = self.activation(self.gn2(self.conv2(out)))

        return out


'''    
 two-layer residual unit: two conv with GN/relu and identity mapping
'''
class residualUnit(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=3,stride=1, padding=1, activation=F.relu, SE=False):
        super(residualUnit, self).__init__()
        self.conv1 = nn.Conv3d(in_size, out_size, kernel_size, stride=1, padding=1)
        init.xavier_uniform_(self.conv1.weight, gain = np.sqrt(2.0)) #or gain=1
        init.constant_(self.conv1.bias, 0)
        self.conv2 = nn.Conv3d(out_size, out_size, kernel_size, stride=1, padding=1)
        init.xavier_uniform_(self.conv2.weight, gain = np.sqrt(2.0)) #or gain=1
        init.constant_(self.conv2.bias, 0)
        self.activation = activation
        self.gn1 = nn.GroupNorm(num_groups=8,num_channels=out_size)
        self.gn2 = nn.GroupNorm(num_groups=8,num_channels=out_size)
        self.in_size = in_size
        self.out_size = out_size
        self.SE = SE
        if SE:
            self.seblock = ChannelSELayer3D(num_channels=out_size)
        if in_size != out_size:
            self.convX = nn.Conv3d(in_size, out_size, kernel_size=1, stride=1, padding=0)
            self.gnX = nn.GroupNorm(num_groups=8,num_channels=out_size)

    def forward(self, x):
        out1 = self.conv1(self.activation(self.gn1(x)))
        out2 = self.conv2(self.activation(self.gn2(out1)))
        if self.SE:
            out = self.seblock(out2)
        else:
            out = out2
        if self.in_size!=self.out_size:
            bridge = self.convX(self.activation(self.gnX(x)))
        elif self.in_size==self.out_size:
            bridge = x
        output = torch.add(out, bridge)

        return output

'''
    Ordinary Residual UNet-Up Conv Block
'''
class UNetUpResBlock(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=3, activation=F.relu,SE=False):
        super(UNetUpResBlock, self).__init__()
        self.up = nn.ConvTranspose3d(in_size, out_size, 2, stride=2)
        self.gnup = nn.GroupNorm(num_groups=8,num_channels=out_size)

        init.xavier_uniform_(self.up.weight, gain = np.sqrt(2.0))
        init.constant_(self.up.bias,0)

        self.activation = activation

        self.resUnit = residualUnit(in_size, out_size, kernel_size = kernel_size,SE=SE)

    def forward(self, x, bridge):
        #print 'x.shape: ',x.shape
        up = self.activation(self.gnup(self.up(x)))
        crop1 = bridge
        out = torch.cat([up, crop1], 1)

        out = self.resUnit(out)
        # out = self.activation(self.bn2(self.conv2(out)))

        return out

# with Attetion Gate
class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.GroupNorm(num_groups=8,num_channels=F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.GroupNorm(num_groups=8,num_channels=F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi

class UNetUpResBlock2(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=3, activation=F.relu, space_dropout=False,SE=False):
        super(UNetUpResBlock2, self).__init__()
        self.up = nn.ConvTranspose3d(in_size, out_size, 2, stride=2)
        self.gnup = nn.GroupNorm(num_groups=8,num_channels=out_size)

        init.xavier_uniform_(self.up.weight, gain = np.sqrt(2.0))
        init.constant_(self.up.bias,0)

        self.activation = activation
        self.attentionGate = Attention_block(F_g=out_size, F_l=out_size, F_int=int(out_size/2))
        self.resUnit = residualUnit(out_size, out_size, kernel_size = kernel_size,SE=SE)

    def center_crop(self, layer, target_size):
        batch_size, n_channels, layer_width, layer_height, layer_depth = layer.size()
        xy1 = (layer_width - target_size) // 2
        return layer[:, :, xy1:(xy1 + target_size), xy1:(xy1 + target_size), xy1:(xy1 + target_size)]

    def forward(self, x, bridge):
        #print 'x.shape: ',x.shape
        up = self.activation(self.gnup(self.up(x)))
        #crop1 = self.center_crop(bridge, up.size()[2])
        #print 'up.shape: ',up.shape, ' crop1.shape: ',crop1.shape
        crop1 = bridge
        out = self.attentionGate(crop1, up)

        out = self.resUnit(out)
        # out = self.activation(self.bn2(self.conv2(out)))

        return out

class AttetionUnet(nn.Module):
    def __init__(self, in_channel=4,init_weights=True,CheckPoint=False):
        super(AttetionUnet, self).__init__()

        filters = [32, 64, 128, 256]
        self.checkpoint = CheckPoint
        self.activation = F.relu

        self.down1 = nn.Conv3d(filters[0], filters[1], kernel_size=3,padding=1, stride=2)
        self.down2 = nn.Conv3d(filters[1], filters[2], kernel_size=3,padding=1, stride=2)
        self.down3 = nn.Conv3d(filters[2], filters[3], kernel_size=3,padding=1, stride=2)

        self.conv_block1_32 = UNetConvBlock(in_channel, filters[0])
        self.conv_block64_1 = residualUnit(filters[1], filters[1],SE=False)
        self.conv_block64_2 = residualUnit(filters[1], filters[1], SE=False)
        self.conv_block128_1 = residualUnit(filters[2], filters[2],SE=False)
        self.conv_block128_2 = residualUnit(filters[2], filters[2], SE=False)
        self.conv_block256_1 = residualUnit(filters[3], filters[3],SE=False)
        self.conv_block256_2 = residualUnit(filters[3], filters[3], SE=False)
        self.conv_block256_3 = residualUnit(filters[3], filters[3], SE=False)
        self.conv_block256_4 = residualUnit(filters[3], filters[3], SE=False)

        # Task2
        self.up2_block256_128 = UNetUpResBlock2(filters[3], filters[2], SE=True)
        self.up2_block128_64 = UNetUpResBlock2(filters[2], filters[1], SE=True)
        self.up2_block64_32 = UNetUpResBlock2(filters[1], filters[0], SE=True)
        self.segment = nn.Conv3d(filters[0], 4, 1, stride=1)
        self.softmax = nn.Softmax(dim=1)

        if init_weights:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        if self.checkpoint:
            inputs = x + torch.zeros(1, dtype=x.dtype, device=x.device, requires_grad=True)
            block1 = checkpoint(self.conv_block1_32,inputs)
            pool1 = checkpoint(self.down1,block1)

            block2 = checkpoint(self.conv_block64_1,pool1)
            block2 = checkpoint(self.conv_block64_2, block2)
            pool2 = checkpoint(self.down2,block2)

            block3 = checkpoint(self.conv_block128_1,pool2)
            block3 = checkpoint(self.conv_block128_2, block3)
            pool3 = checkpoint(self.down3,block3)

            block4 = checkpoint(self.conv_block256_1,pool3)
            block4 = checkpoint(self.conv_block256_2, block4)
            block4 = checkpoint(self.conv_block256_3, block4)
            block4 = checkpoint(self.conv_block256_4, block4)

            # Task2
            task2_2 = checkpoint(self.up2_block256_128, block4, block3)
            task2_3 = checkpoint(self.up2_block128_64, task2_2, block2)
            task2_4 = checkpoint(self.up2_block64_32, task2_3, block1)
            segment = self.softmax(checkpoint(self.segment, task2_4))

        return segment

class AttetionUnet_VAE(nn.Module):
    def __init__(self, in_channel=4,init_weights=True,CheckPoint=False):
        super(AttetionUnet_VAE, self).__init__()

        filters = [32, 64, 128, 256]
        self.checkpoint = CheckPoint
        self.activation = F.relu

        self.down1 = nn.Conv3d(filters[0], filters[1], kernel_size=3,padding=1, stride=2)
        self.down2 = nn.Conv3d(filters[1], filters[2], kernel_size=3,padding=1, stride=2)
        self.down3 = nn.Conv3d(filters[2], filters[3], kernel_size=3,padding=1, stride=2)

        self.conv_block1_32 = UNetConvBlock(in_channel, filters[0])
        self.conv_block64_1 = residualUnit(filters[1], filters[1],SE=False)
        self.conv_block64_2 = residualUnit(filters[1], filters[1], SE=False)
        self.conv_block128_1 = residualUnit(filters[2], filters[2],SE=False)
        self.conv_block128_2 = residualUnit(filters[2], filters[2], SE=False)
        self.conv_block256_1 = residualUnit(filters[3], filters[3],SE=False)
        self.conv_block256_2 = residualUnit(filters[3], filters[3], SE=False)
        self.conv_block256_3 = residualUnit(filters[3], filters[3], SE=False)
        self.conv_block256_4 = residualUnit(filters[3], filters[3], SE=False)

        # Task1
        # Variational Auto-Encoder
        self.dense_features = (8, 8, 8)
        self.vae = VAE(256, outChans=4, dense_features=self.dense_features)

        # Task2
        self.up2_block256_128 = UNetUpResBlock2(filters[3], filters[2], SE=True)
        self.up2_block128_64 = UNetUpResBlock2(filters[2], filters[1], SE=True)
        self.up2_block64_32 = UNetUpResBlock2(filters[1], filters[0], SE=True)
        self.segment = nn.Conv3d(filters[0], 4, 1, stride=1)
        self.softmax = nn.Softmax(dim=1)

        if init_weights:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        if self.checkpoint:
            inputs = x + torch.zeros(1, dtype=x.dtype, device=x.device, requires_grad=True)
            block1 = checkpoint(self.conv_block1_32,inputs)
            pool1 = checkpoint(self.down1,block1)

            block2 = checkpoint(self.conv_block64_1,pool1)
            block2 = checkpoint(self.conv_block64_2, block2)
            pool2 = checkpoint(self.down2,block2)

            block3 = checkpoint(self.conv_block128_1,pool2)
            block3 = checkpoint(self.conv_block128_2, block3)
            pool3 = checkpoint(self.down3,block3)

            block4 = checkpoint(self.conv_block256_1,pool3)
            block4 = checkpoint(self.conv_block256_2, block4)
            block4 = checkpoint(self.conv_block256_3, block4)
            block4 = checkpoint(self.conv_block256_4, block4)

            # Task1
            rebuild, out_distr = checkpoint(self.vae,block4)

            # Task2
            task2_2 = checkpoint(self.up2_block256_128, block4, block3)
            task2_3 = checkpoint(self.up2_block128_64, task2_2, block2)
            task2_4 = checkpoint(self.up2_block64_32, task2_3, block1)
            segment = self.softmax(checkpoint(self.segment, task2_4))

        return segment, rebuild, out_distr

class WNet_withoutVAE(nn.Module):
    def __init__(self):
        super(WNet_withoutVAE, self).__init__()

        # Stage 1
        self.stage1 = AttetionUnet(in_channel=4, CheckPoint=True)
        # Stage 2
        self.stage2 = AttetionUnet(in_channel=8, CheckPoint=True)

    def forward(self, x):
        inputs = x + torch.zeros(1, dtype=x.dtype, device=x.device, requires_grad=True)
        segment1 = checkpoint(self.stage1,inputs)  # (4, 128, 128, 128)
        mid = torch.cat((inputs,segment1), dim=1)
        segment2 = checkpoint(self.stage2,mid)

        return segment1, segment2


## Ours
class WNet_withVAE(nn.Module):
    def __init__(self):
        super(WNet_withVAE, self).__init__()

        # Stage 1
        self.stage1 = AttetionUnet(in_channel=4, CheckPoint=True)
        # Stage 2
        self.stage2 = AttetionUnet_VAE(in_channel=8, CheckPoint=True)


    def forward(self, x):
        inputs = x + torch.zeros(1, dtype=x.dtype, device=x.device, requires_grad=True)
        segment1 = checkpoint(self.stage1,inputs)  # (4, 128, 128, 128)
        mid = torch.cat((inputs,segment1), dim=1)
        segment2, rebuild, out_distr = checkpoint(self.stage2,mid)


        return segment1, segment2, rebuild, out_distr

class WNet_withVAE_C(nn.Module):
    def __init__(self):
        super(WNet_withVAE_C, self).__init__()

        # Stage 1
        self.stage1 = AttetionUnet(in_channel=4, CheckPoint=True)
        # Stage 2
        self.stage2 = AttetionUnet_VAE(in_channel=8, CheckPoint=True)
        #Classification
        self.linear1 = nn.Linear(257, 64)
        self.linear2 = nn.Linear(64, 1)

    def forward(self, img, age):
        inputs = img + torch.zeros(1, dtype=img.dtype, device=img.device, requires_grad=True)
        inputs_age = age + torch.zeros(1, dtype=age.dtype, device=age.device, requires_grad=True)
        segment1 = checkpoint(self.stage1,inputs)  # (4, 128, 128, 128)
        mid = torch.cat((inputs,segment1), dim=1)
        segment2, rebuild, out_distr = checkpoint(self.stage2,mid)
        vae_age = torch.cat((out_distr, inputs_age.unsqueeze(0)), dim=1)
        out_reg = checkpoint(self.linear1, vae_age)
        out_reg = checkpoint(self.linear2, out_reg)

        return segment1, segment2, rebuild, out_distr, out_reg