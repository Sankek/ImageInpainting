import torch
import torch.nn as nn
import torchvision


class VGG16FE(nn.Module):
    def __init__(self):
        super().__init__()
        self.vgg16 = torchvision.models.vgg16(
            weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1
        )
        
        self.feature_layers = [
            self.vgg16.features[:5], 
            self.vgg16.features[5:10], 
            self.vgg16.features[10:17]
        ]
        
        for layer in self.feature_layers:
            for param in layer.parameters():
                param.requires_grad = False
                
    def forward(self, input):
        f0 = self.feature_layers[0](input)
        f1 = self.feature_layers[0](f0)
        f2 = self.feature_layers[0](f1)

        return [f0, f1, f2]
    
    
class PConv2d(nn.Module):
    """
    Implementation of Partial Convolution from NVIDIA paper 
    https://arxiv.org/abs/1804.07723
    
    Input
    ----------
    input : torch.FloatTensor of shape (batch_size, in_channels, H, W)
    input_mask : torch.FloatTensor of shape (batch_size, in_channels, H, W)
        Mask with 0.0 on hole positions and 1.0 on valid positions.
        
    Output
    ----------
    output : torch.FloatTensor of shape (batch_size, out_channels, H, W)
    output_mask : torch.FloatTensor of shape (batch_size, 1, H, W)
        Updated mask. It is input_mask where zeros are replaced to ones on
        positions which output was conditioned on at least one valid input position.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(*args, **kwargs)
        self.use_bias = True if self.conv.bias is not None else False
        self.kernel_size = self.conv.kernel_size
        
        mask_args, mask_kwargs = list(args), kwargs
        self._edit_args(mask_args, mask_kwargs, 7, 'bias', False)
            
        self.mask_conv = nn.Conv2d(*mask_args, **mask_kwargs)
        self.mask_conv.weight.data.fill_(1.0)
        self.mask_conv.weight.requires_grad = False        
        
    @staticmethod
    def _edit_args(args, kwargs, pos, key, value):
        if len(args) > pos:
            args[pos] = value
        else:
            kwargs[key] = value
        
    def forward(self, input, input_mask):
        mask_sum = self.mask_conv(input_mask) 

        stop_mask = mask_sum == 0
        
        mask_sum_inv = (1/mask_sum)
        mask_sum_inv[stop_mask] = 0
        normalization = self.kernel_size[0]*self.kernel_size[1] * mask_sum_inv
        if self.use_bias:
            bias = self.conv.bias.view(1, -1, 1, 1)
            output = (self.conv(input*input_mask)-bias) * normalization + bias
        else:
            output = self.conv(input*input_mask)*normalization
        
        output_mask = (~stop_mask).float()
        output[stop_mask] = 0
        
        return output, output_mask
    
    
class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros',
                 mode='conv', use_relu=True, ReLU_nslope=0.2, use_dropout=False, 
                 dropout_p=0.5, use_batchnorm = True):
        super().__init__()
        
        conv_args = (
            in_channels, out_channels, kernel_size, stride,
            padding, dilation, groups, bias, padding_mode
        )

        self.use_batchnorm = use_batchnorm
        self.use_dropout = use_dropout
        self.use_relu = use_relu
        self.mode = mode

        if self.use_batchnorm:
            self.bias = False
        else:
            self.bias = True

        available_modes = ['conv', 'deconv', 'pconv']
        if not self.mode in available_modes:
            print(f"{self.mode} is not correct; correct modes: {available_modes}")
            raise NotImplementedError

        if self.mode == 'conv':
            self.conv = nn.Conv2d(*conv_args)
        elif self.mode == 'pconv':
            self.conv = PConv2d(*conv_args)
        else:
            self.conv = nn.ConvTranspose2d(*conv_args)

        if use_dropout:
            self.dropout = nn.Dropout(dropout_p) 
        if use_batchnorm:
            self.bn = nn.BatchNorm2d(out_channels)
        if use_relu:
            self.relu = nn.LeakyReLU(ReLU_nslope) if ReLU_nslope != 0.0 else nn.ReLU()


    def forward(self, *args, **kwargs):
        if self.mode == 'pconv':
            out, out_mask = self.conv(*args, **kwargs)
        else:
            out = self.conv(*args, **kwargs)

        if self.use_batchnorm:
            out = self.bn(out)
        if self.use_dropout:
            out = self.dropout(out)
        if self.use_relu:
            out = self.relu(out)
            
        if self.mode == 'pconv':
            return out, out_mask
        else:
            return out
        
        
class MultiArgSequential(nn.Sequential):
    """
    Class to combine multiple models. Sequential allowing multiple inputs.
    from https://discuss.pytorch.org/t/nn-sequential-layers-forward-with-multiple-inputs-error/35591/7
    """

    def __init__(self, *args):
        super().__init__(*args)

    def forward(self, x, *args, **kwargs):
        for i, module in enumerate(self):
            if i == 0:
                x = module(x, *args, **kwargs)
            else:
                x = module(*x, **kwargs)
            if not isinstance(x, tuple) and i != len(self) - 1:
                x = (x,)
        return x

        
class PConvUNet(nn.Module):
    def __init__(self, channels=3):
        super().__init__()
        self.train_encoder_bn = True

        self.conv0 = MultiArgSequential(
          ConvLayer(channels, 64, 5, padding=2, mode = 'pconv'),
        )
        self.pool0 = ConvLayer(64, 128, 3, stride=2, padding=1, mode = 'pconv') # 256 -> 128
        self.conv1 = MultiArgSequential(
          ConvLayer(128, 128, 3, padding=1, mode = 'pconv'),
        )
        self.pool1 = ConvLayer(128, 256, 3, stride=2, padding=1, mode = 'pconv') # 128 -> 64
        self.conv2 = MultiArgSequential(
          ConvLayer(256, 256, 3, padding=1, mode = 'pconv'),
          ConvLayer(256, 256, 3, padding=1, mode = 'pconv'),
        )
        self.pool2 = ConvLayer(256, 256, 3, stride=2, padding=1, mode = 'pconv') # 64 -> 32
        
        self.encoder = ([self.conv0, self.pool0, self.conv1, self.pool1, self.conv2, self.pool2])
        
        # bottleneck
        self.bottleneck = MultiArgSequential(
          ConvLayer(256, 256, 3, padding=2, dilation=2, mode = 'pconv'),
          ConvLayer(256, 256, 3, padding=4, dilation=4, mode = 'pconv'),
        )
        
        self.pool3 = nn.Upsample(scale_factor=2, mode='nearest') # 32 -> 64
        self.conv3 = MultiArgSequential(
          ConvLayer(256+256, 256, 3, padding=1, mode = 'pconv'),
          ConvLayer(256, 256, 3, padding=1, mode = 'pconv'),
        )
        self.pool4 = nn.Upsample(scale_factor=2, mode='nearest') # 64 -> 128
        self.conv4 = MultiArgSequential(
          ConvLayer(256+128, 128, 3, padding=1, mode = 'pconv'),
        )
        self.pool5 = nn.Upsample(scale_factor=2, mode='nearest') # 128 -> 256
        self.conv5 = MultiArgSequential(
          ConvLayer(128+64, 64, 3, padding=1, mode = 'pconv'),
          PConv2d(64, channels, kernel_size=3, 
                  padding=1, stride=1), # no activation
        )
        
    def train(self, T=True):
        super().train(T)
        if not self.train_encoder_bn and T:
            for submodule in self.encoder:
                for name, module in submodule.named_modules():
                    if isinstance(module, nn.BatchNorm2d):
                        module.eval()

    def forward(self, input, input_mask):
        # encoder
        conv0 = self.conv0(input, input_mask)
        out = self.pool0(*conv0)
        conv1 = self.conv1(*out)
        out = self.pool1(*conv1)
        conv2 = self.conv2(*out)
        out = self.pool2(*conv2)

        # bottleneck
        out = self.bottleneck(*out)

        # decoder
        out, out_mask = self.pool3(out[0]), self.pool3(out[1])
        out = torch.cat([out, conv2[0]], dim=1)
        out_mask = torch.cat([out_mask, conv2[1]], dim=1)
        out = self.conv3(out, out_mask)
        out, out_mask = self.pool4(out[0]), self.pool4(out[1])
        out = torch.cat([out, conv1[0]], dim=1)
        out_mask = torch.cat([out_mask, conv1[1]], dim=1)
        out = self.conv4(out, out_mask)
        out, out_mask = self.pool5(out[0]), self.pool5(out[1])
        out = torch.cat([out, conv0[0]], dim=1)
        out_mask = torch.cat([out_mask, conv0[1]], dim=1)
        
        out, out_mask = self.conv5(out, out_mask)
        
        return out, out_mask
    

class PConvUNet_v2(nn.Module):
    def __init__(self, channels=3):
        super().__init__()
        self.train_encoder_bn = True

        self.conv0 = MultiArgSequential(
          ConvLayer(channels, 64, 5, padding=2, mode = 'pconv'),
        )
        self.pool0 = ConvLayer(64, 128, 3, stride=2, padding=1, mode = 'pconv') # 256 -> 128
        self.conv1 = MultiArgSequential(
          ConvLayer(128, 128, 3, padding=1, mode = 'pconv'),
        )
        self.pool1 = ConvLayer(128, 256, 3, stride=2, padding=1, mode = 'pconv') # 128 -> 64
        self.conv2 = MultiArgSequential(
          ConvLayer(256, 256, 3, padding=2, dilation=2, mode = 'pconv'),
          ConvLayer(256, 256, 3, padding=2, dilation=2, mode = 'pconv'),
        )
        self.pool2 = ConvLayer(256, 512, 3, stride=2, padding=1, mode = 'pconv') # 64 -> 32
        self.conv3 = MultiArgSequential(
          ConvLayer(512, 512, 3, padding=2, dilation=2, mode = 'pconv'), 
          ConvLayer(512, 512, 3, padding=2, dilation=2, mode = 'pconv'),
        )
        self.pool3 = ConvLayer(512, 1024, 3, stride=2, padding=1, mode = 'pconv') # 32 -> 16

        self.encoder = ([self.conv0, self.pool0, self.conv1, self.pool1, self.conv2, self.pool2, self.conv3, self.pool3])
        
        # bottleneck
        self.bottleneck = MultiArgSequential(
          ConvLayer(1024, 512, 3, padding=1, mode = 'pconv'),
        )

        self.pool4 = nn.Upsample(scale_factor=2, mode='nearest') # 16 -> 32
        self.conv4 = MultiArgSequential(
          ConvLayer(512+512, 512, 3, padding=2, dilation=2, mode = 'pconv'),
          ConvLayer(512, 256, 3, padding=2, dilation=2, mode = 'pconv'),
        )       
        self.pool5 = nn.Upsample(scale_factor=2, mode='nearest') # 32 -> 64
        self.conv5 = MultiArgSequential(
          ConvLayer(256+256, 256, 3, padding=2, dilation=2, mode = 'pconv'),
          ConvLayer(256, 128, 3, padding=2, dilation=2, mode = 'pconv'),
        )
        self.pool6 = nn.Upsample(scale_factor=2, mode='nearest') # 64 -> 128
        self.conv6 = MultiArgSequential(
          ConvLayer(128+128, 64, 3, padding=1, mode = 'pconv'),
        )
        self.pool7 = nn.Upsample(scale_factor=2, mode='nearest') # 128 -> 256
        self.conv7 = MultiArgSequential(
          ConvLayer(64+64, 64, 3, padding=1, mode = 'pconv'),
          PConv2d(64, channels, kernel_size=3, 
                  padding=1, stride=1), # no activation
        )
        
    def train(self, T=True):
        super().train(T)
        if not self.train_encoder_bn and T:
            for submodule in self.encoder:
                for name, module in submodule.named_modules():
                    if isinstance(module, nn.BatchNorm2d):
                        module.eval()

    def forward(self, input, input_mask):
        # encoder
        conv0 = self.conv0(input, input_mask)
        out = self.pool0(*conv0)
        conv1 = self.conv1(*out)
        out = self.pool1(*conv1)
        conv2 = self.conv2(*out)
        out = self.pool2(*conv2)
        conv3 = self.conv3(*out)
        out = self.pool3(*conv3)

        # bottleneck
        out = self.bottleneck(*out)

        # decoder
        out, out_mask = self.pool4(out[0]), self.pool4(out[1])
        out = torch.cat([out, conv3[0]], dim=1)
        out_mask = torch.cat([out_mask, conv3[1]], dim=1)
        out = self.conv4(out, out_mask)
        out, out_mask = self.pool5(out[0]), self.pool5(out[1])
        out = torch.cat([out, conv2[0]], dim=1)
        out_mask = torch.cat([out_mask, conv2[1]], dim=1)
        out = self.conv5(out, out_mask)
        out, out_mask = self.pool6(out[0]), self.pool6(out[1])
        out = torch.cat([out, conv1[0]], dim=1)
        out_mask = torch.cat([out_mask, conv1[1]], dim=1)
        out = self.conv6(out, out_mask)
        out, out_mask = self.pool7(out[0]), self.pool7(out[1])
        out = torch.cat([out, conv0[0]], dim=1)
        out_mask = torch.cat([out_mask, conv0[1]], dim=1)
        
        out, out_mask = self.conv7(out, out_mask)
        
        return out, out_mask
    
    
class DiscriminatorNet(nn.Module):
    def __init__(self):
        super().__init__(input_channels=3, conditional_channels=1)
        
        c_dims = [64, 128, 256, 256, 256, 256]

        self.model = nn.Sequential(
            ConvLayer(input_channels+conditional_channels, c_dims[0], 5, stride=1, padding=2, ReLU_nslope=0) # 1 -> 1
            ConvLayer(c_dims[0], c_dims[1], 5, stride=2, padding=2, ReLU_nslope=0) #    1 -> 1/2
            ConvLayer(c_dims[1], c_dims[2], 5, stride=2, padding=2, ReLU_nslope=0) #  1/2 -> 1/4
            ConvLayer(c_dims[2], c_dims[3], 5, stride=2, padding=2, ReLU_nslope=0) #  1/4 -> 1/8
            ConvLayer(c_dims[3], c_dims[4], 5, stride=2, padding=2, ReLU_nslope=0) #  1/8 -> 1/16
            ConvLayer(c_dims[4], c_dims[5], 5, stride=2, padding=2, use_batchnorm=False, use_relu=False,) # 1/16 -> 1/32
            nn.Sigmoid()
        )

        
    def forward(self, input, condition):
        output = self.model(torch.cat([input, condition], dim=1))
        output = output.view(output.shape[0], -1).mean(dim=1)
        
        return output