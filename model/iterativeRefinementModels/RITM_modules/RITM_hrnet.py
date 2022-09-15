import os
import numpy as np
import torch
import torch.nn as nn
import torch._utils
import torch.nn.functional as F
from model.iterativeRefinementModels.RITM_modules.RITM_ocr import SpatialOCR_Module, SpatialGather_Module


relu_inplace = True


class HighResolutionModule(nn.Module):
    def __init__(self, num_branches, blocks, num_blocks, num_inchannels,
                 num_channels, fuse_method,multi_scale_output=True,
                 norm_layer=nn.BatchNorm2d, align_corners=True):
        super(HighResolutionModule, self).__init__()
        self._check_branches(num_branches, num_blocks, num_inchannels, num_channels)

        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches
        self.norm_layer = norm_layer
        self.align_corners = align_corners

        self.multi_scale_output = multi_scale_output

        self.branches = self._make_branches(
            num_branches, blocks, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(inplace=relu_inplace)

    def _check_branches(self, num_branches, num_blocks, num_inchannels, num_channels):
        if num_branches != len(num_blocks):
            error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(
                num_branches, len(num_blocks))
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_CHANNELS({})'.format(
                num_branches, len(num_channels))
            raise ValueError(error_msg)

        if num_branches != len(num_inchannels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_INCHANNELS({})'.format(
                num_branches, len(num_inchannels))
            raise ValueError(error_msg)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels,
                         stride=1):
        downsample = None
        if stride != 1 or \
                self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.num_inchannels[branch_index],
                          num_channels[branch_index] * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                self.norm_layer(num_channels[branch_index] * block.expansion),
            )

        layers = []
        layers.append(block(self.num_inchannels[branch_index],
                            num_channels[branch_index], stride,
                            downsample=downsample, norm_layer=self.norm_layer))
        self.num_inchannels[branch_index] = \
            num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(block(self.num_inchannels[branch_index],
                                num_channels[branch_index],
                                norm_layer=self.norm_layer))

        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []

        for i in range(num_branches):
            branches.append(
                self._make_one_branch(i, block, num_blocks, num_channels))

        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(nn.Sequential(
                        nn.Conv2d(in_channels=num_inchannels[j],
                                  out_channels=num_inchannels[i],
                                  kernel_size=1,
                                  bias=False),
                        self.norm_layer(num_inchannels[i])))
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_inchannels[j],
                                          num_outchannels_conv3x3,
                                          kernel_size=3, stride=2, padding=1, bias=False),
                                self.norm_layer(num_outchannels_conv3x3)))
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_inchannels[j],
                                          num_outchannels_conv3x3,
                                          kernel_size=3, stride=2, padding=1, bias=False),
                                self.norm_layer(num_outchannels_conv3x3),
                                nn.ReLU(inplace=relu_inplace)))
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []
        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                elif j > i:
                    width_output = x[i].shape[-1]
                    height_output = x[i].shape[-2]
                    y = y + F.interpolate(
                        self.fuse_layers[i][j](x[j]),
                        size=[height_output, width_output],
                        mode='bilinear', align_corners=self.align_corners)
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))

        return x_fuse


class SqueezeExitationBlock(nn.Module):
    def __init__(self, in_ch, mid_ch1, out_ch, SE_maxpool=False, SE_softmax=False):
        super(SqueezeExitationBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, mid_ch1, kernel_size=(1, 1))
        self.conv2 = nn.Conv2d(mid_ch1, out_ch, kernel_size=(1, 1))
        self.SE_maxpool = SE_maxpool
        self.SE_softmax = SE_softmax

    def forward(self, x):
        # ===  squeeze  ===
        # pool
        if self.SE_maxpool:
            x = x.max(-1)[0].max(-1)[0]
            x = x[:,:,None,None]
        else:
            x = x.mean(-1, keepdim=True).mean(-2, keepdim=True)  # b, in_ch, 1, 1
        # conv1, relu
        x = self.conv1(x).relu()  # b, mid_ch, 1, 1
        # conv2, relu
        x = self.conv2(x)

        if self.SE_softmax:
            x = x.softmax(1)
        else:
            x = x.sigmoid()  # b, out_ch, 1, 1
        return x
class ConvBlocks(nn.Module):
    def __init__(self, in_ch, channels, kernel_sizes=None, strides=None, dilations=None, paddings=None,
                 BatchNorm=nn.BatchNorm2d):
        super(ConvBlocks, self).__init__()
        self.num = len(channels)
        if kernel_sizes is None: kernel_sizes = [3 for c in channels]
        if strides is None: strides = [1 for c in channels]
        if dilations is None: dilations = [1 for c in channels]
        if paddings is None: paddings = [
            ((kernel_sizes[i] // 2) if dilations[i] == 1 else (kernel_sizes[i] // 2 * dilations[i])) for i in
            range(self.num)]
        convs_tmp = []
        for i in range(self.num):
            if channels[i] == 1:
                convs_tmp.append(
                    nn.Conv2d(in_ch if i == 0 else channels[i - 1], channels[i], kernel_size=kernel_sizes[i],
                              stride=strides[i], padding=paddings[i], dilation=dilations[i]))
            else:
                convs_tmp.append(nn.Sequential(
                    nn.Conv2d(in_ch if i == 0 else channels[i - 1], channels[i], kernel_size=kernel_sizes[i],
                              stride=strides[i], padding=paddings[i], dilation=dilations[i], bias=False),
                    BatchNorm(channels[i]), nn.ReLU()))
        self.convs = nn.Sequential(*convs_tmp)

        # weight initialization
        for m in self.convs.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        return self.convs(x)
class HintEncSENet(nn.Module):
    def __init__(self, se_output_channel, num_classes, SE_maxpool=False, SE_softmax=False, input_channel=256):
        super(HintEncSENet, self).__init__()
        self.SENet = SqueezeExitationBlock(256, 256//16, se_output_channel, SE_maxpool=SE_maxpool, SE_softmax=SE_softmax)
        self.hintEncoder = ConvBlocks(input_channel+num_classes, [256, 256, 256], [3, 3, 3], [2, 1, 1])


    def forward(self,x,hint):
        hint = F.interpolate(hint, size=x.size()[2:], mode='bilinear', align_corners=True)
        se = self.hintEncoder(torch.cat((x, hint),dim=1))
        se = self.SENet(se)
        return se


class HighResolutionNet(nn.Module):
    def __init__(self, width, num_classes, ocr_width=256, small=False,
                 norm_layer=nn.BatchNorm2d, align_corners=True, addHintEncSENet=False, SE_maxpool=False, SE_softmax=False):
        super(HighResolutionNet, self).__init__()


        self.norm_layer = norm_layer
        self.width = width
        self.ocr_width = ocr_width
        self.align_corners = align_corners

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = norm_layer(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = norm_layer(64)
        self.relu = nn.ReLU(inplace=relu_inplace)

        num_blocks = 2 if small else 4

        stage1_num_channels = 64
        self.layer1 = self._make_layer(BottleneckV1b, 64, stage1_num_channels, blocks=num_blocks)
        stage1_out_channel = BottleneckV1b.expansion * stage1_num_channels

        self.stage2_num_branches = 2
        num_channels = [width, 2 * width]
        num_inchannels = [
            num_channels[i] * BasicBlockV1b.expansion for i in range(len(num_channels))]
        self.transition1 = self._make_transition_layer(
            [stage1_out_channel], num_inchannels)
        self.stage2, pre_stage_channels = self._make_stage(
            BasicBlockV1b, num_inchannels=num_inchannels, num_modules=1, num_branches=self.stage2_num_branches,
            num_blocks=2 * [num_blocks], num_channels=num_channels)

        self.stage3_num_branches = 3
        num_channels = [width, 2 * width, 4 * width]
        num_inchannels = [
            num_channels[i] * BasicBlockV1b.expansion for i in range(len(num_channels))]
        self.transition2 = self._make_transition_layer(
            pre_stage_channels, num_inchannels)
        self.stage3, pre_stage_channels = self._make_stage(
            BasicBlockV1b, num_inchannels=num_inchannels,
            num_modules=3 if small else 4, num_branches=self.stage3_num_branches,
            num_blocks=3 * [num_blocks], num_channels=num_channels)

        self.stage4_num_branches = 4
        num_channels = [width, 2 * width, 4 * width, 8 * width]
        num_inchannels = [
            num_channels[i] * BasicBlockV1b.expansion for i in range(len(num_channels))]
        self.transition3 = self._make_transition_layer(
            pre_stage_channels, num_inchannels)
        self.stage4, pre_stage_channels = self._make_stage(
            BasicBlockV1b, num_inchannels=num_inchannels, num_modules=2 if small else 3,
            num_branches=self.stage4_num_branches,
            num_blocks=4 * [num_blocks], num_channels=num_channels)

        last_inp_channels = np.int(np.sum(pre_stage_channels))
        if self.ocr_width > 0:
            ocr_mid_channels = 2 * self.ocr_width
            ocr_key_channels = self.ocr_width

            self.conv3x3_ocr = nn.Sequential(
                nn.Conv2d(last_inp_channels, ocr_mid_channels,
                          kernel_size=3, stride=1, padding=1),
                norm_layer(ocr_mid_channels),
                nn.ReLU(inplace=relu_inplace),
            )
            self.ocr_gather_head = SpatialGather_Module(num_classes)

            self.ocr_distri_head = SpatialOCR_Module(in_channels=ocr_mid_channels,
                                                     key_channels=ocr_key_channels,
                                                     out_channels=ocr_mid_channels,
                                                     scale=1,
                                                     dropout=0.05,
                                                     norm_layer=norm_layer,
                                                     align_corners=align_corners)
            self.cls_head = nn.Conv2d(
                ocr_mid_channels, num_classes, kernel_size=1, stride=1, padding=0, bias=True)

            self.aux_head = nn.Sequential(
                nn.Conv2d(last_inp_channels, last_inp_channels,
                          kernel_size=1, stride=1, padding=0),
                norm_layer(last_inp_channels),
                nn.ReLU(inplace=relu_inplace),
                nn.Conv2d(last_inp_channels, num_classes,
                          kernel_size=1, stride=1, padding=0, bias=True)
            )
        else:
            self.cls_head = nn.Sequential(
                nn.Conv2d(last_inp_channels, last_inp_channels,
                          kernel_size=3, stride=1, padding=1),
                norm_layer(last_inp_channels),
                nn.ReLU(inplace=relu_inplace),
                nn.Conv2d(last_inp_channels, num_classes,
                          kernel_size=1, stride=1, padding=0, bias=True)
            )


        self.addHintEncSENet = addHintEncSENet
        if self.addHintEncSENet:
            self.HintEncSENet = HintEncSENet( se_output_channel=last_inp_channels, num_classes=num_classes, SE_maxpool=SE_maxpool, SE_softmax=SE_softmax)


    def _make_transition_layer(
            self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(nn.Sequential(
                        nn.Conv2d(num_channels_pre_layer[i],
                                  num_channels_cur_layer[i],
                                  kernel_size=3,
                                  stride=1,
                                  padding=1,
                                  bias=False),
                        self.norm_layer(num_channels_cur_layer[i]),
                        nn.ReLU(inplace=relu_inplace)))
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i + 1 - num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] \
                        if j == i - num_branches_pre else inchannels
                    conv3x3s.append(nn.Sequential(
                        nn.Conv2d(inchannels, outchannels,
                                  kernel_size=3, stride=2, padding=1, bias=False),
                        self.norm_layer(outchannels),
                        nn.ReLU(inplace=relu_inplace)))
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                self.norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(inplanes, planes, stride,
                            downsample=downsample, norm_layer=self.norm_layer))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes, norm_layer=self.norm_layer))

        return nn.Sequential(*layers)

    def _make_stage(self, block, num_inchannels,
                    num_modules, num_branches, num_blocks, num_channels,
                    fuse_method='SUM',
                    multi_scale_output=True):
        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True
            modules.append(
                HighResolutionModule(num_branches,
                                     block,
                                     num_blocks,
                                     num_inchannels,
                                     num_channels,
                                     fuse_method,
                                     reset_multi_scale_output,
                                     norm_layer=self.norm_layer,
                                     align_corners=self.align_corners)
            )
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels

    def forward(self, x, additional_features=None, input_hint_heatmap=None):
        feats = self.compute_hrnet_feats(x, additional_features, input_hint_heatmap)
        if self.ocr_width > 0:
            out_aux = self.aux_head(feats) # aux_head : conv norm relu conv (soft object regions), output channel: num_classes
            feats = self.conv3x3_ocr(feats) # conv3x3_ocr : conv norm relu (pixel representation

            context = self.ocr_gather_head(feats, out_aux) # context :  batch x c x num_keypoint x 1, feats: batch, c, H, W
            feats = self.ocr_distri_head(feats, context)
            out = self.cls_head(feats)
            return [out, out_aux]
        else:
            return [self.cls_head(feats), None]

    def compute_hrnet_feats(self, x, additional_features, input_hint_heatmap):
        x = self.compute_pre_stage_features(x, additional_features)
        x = self.layer1(x)

        if input_hint_heatmap is not None:
            hint_encoder_output = self.HintEncSENet(x, input_hint_heatmap)

        x_list = []
        for i in range(self.stage2_num_branches):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)

        x_list = []
        for i in range(self.stage3_num_branches):
            if self.transition2[i] is not None:
                if i < self.stage2_num_branches:
                    x_list.append(self.transition2[i](y_list[i]))
                else:
                    x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)

        x_list = []
        for i in range(self.stage4_num_branches):
            if self.transition3[i] is not None:
                if i < self.stage3_num_branches:
                    x_list.append(self.transition3[i](y_list[i]))
                else:
                    x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        x = self.stage4(x_list)

        out = self.aggregate_hrnet_features(x)
        if input_hint_heatmap is not None:
            return hint_encoder_output * out
        else:
            return out


    def compute_pre_stage_features(self, x, additional_features):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if additional_features is not None:
            x = x + additional_features
        x = self.conv2(x)
        x = self.bn2(x)
        return self.relu(x)

    def aggregate_hrnet_features(self, x):
        # Upsampling
        x0_h, x0_w = x[0].size(2), x[0].size(3)
        x1 = F.interpolate(x[1], size=(x0_h, x0_w),
                           mode='bilinear', align_corners=self.align_corners)
        x2 = F.interpolate(x[2], size=(x0_h, x0_w),
                           mode='bilinear', align_corners=self.align_corners)
        x3 = F.interpolate(x[3], size=(x0_h, x0_w),
                           mode='bilinear', align_corners=self.align_corners)

        return torch.cat([x[0], x1, x2, x3], 1)

    def load_pretrained_weights(self, pretrained_path=''):
        model_dict = self.state_dict()

        if not os.path.exists(pretrained_path):
            print(f'\nFile "{pretrained_path}" does not exist.')
            print('You need to specify the correct path to the pre-trained weights.\n'
                  'You can download the weights for HRNet from the repository:\n'
                  'https://github.com/HRNet/HRNet-Image-Classification')
            exit(1)
        pretrained_dict = torch.load(pretrained_path, map_location='cpu')
        pretrained_dict = {k.replace('last_layer', 'aux_head').replace('model.', ''): v for k, v in
                           pretrained_dict.items()}

        pretrained_dict = {k: v for k, v in pretrained_dict.items()
                           if k in model_dict.keys()}

        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict, strict=False)




import torch
import torch.nn as nn
GLUON_RESNET_TORCH_HUB = 'rwightman/pytorch-pretrained-gluonresnet'


class BasicBlockV1b(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None,
                 previous_dilation=1, norm_layer=nn.BatchNorm2d):
        super(BasicBlockV1b, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                               padding=dilation, dilation=dilation, bias=False)
        self.bn1 = norm_layer(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=previous_dilation, dilation=previous_dilation, bias=False)
        self.bn2 = norm_layer(planes)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu(out)

        return out


class BottleneckV1b(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None,
                 previous_dilation=1, norm_layer=nn.BatchNorm2d):
        super(BottleneckV1b, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = norm_layer(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=dilation, dilation=dilation, bias=False)
        self.bn2 = norm_layer(planes)

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = norm_layer(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu(out)

        return out