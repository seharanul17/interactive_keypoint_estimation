import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import numpy as np

from misc.heatmap_maker import HeatmapMaker, heatmap2hargmax_coord
from misc.loss import LossManager

from model.iterativeRefinementModels.RITM_modules.RITM_hrnet import HighResolutionNet

class RITM(nn.Module):
    def __init__(self, config):
        super(RITM, self).__init__()
        # default
        self.config = config
        self.device = config.MISC.device

        self.heatmap_maker = HeatmapMaker(config)
        self.LossManager = LossManager(config, self.heatmap_maker)


        # RITM hyper-param
        self.max_iter = 3

        # backbone
        self.ritm_hrnet = HighResolutionNet(width=32, ocr_width=128, small=False, num_classes=config.Dataset.num_keypoint,
                                            norm_layer=nn.BatchNorm2d, addHintEncSENet=True, SE_maxpool=config.Model.SE_maxpool, SE_softmax=config.Model.SE_softmax)
        self.ritm_hrnet.load_pretrained_weights('/home/nas1_userD/taesung/IKE_InteractiveKeypointEstimation/pretrained_models/hrnetv2_w32_imagenet_pretrained.pth')


        if self.config.Model.no_iterative_training:
            hintmap_input_channels = config.Dataset.num_keypoint
        else:
            hintmap_input_channels = config.Dataset.num_keypoint*2

        self.hintmap_encoder = nn.Sequential(
            nn.Conv2d(in_channels=hintmap_input_channels, out_channels=16, kernel_size=1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, stride=2, padding=1),
            ScaleLayer(init_value=0.05, lr_mult=1)
        )

    def forward_model(self, input_image, input_hint_heatmap, prev_pred_heatmap):
        if prev_pred_heatmap is None:
            hint_features = input_hint_heatmap
        else:
            hint_features = torch.cat((prev_pred_heatmap, input_hint_heatmap), dim=1)

        encoded_input_hint_heatmap = self.hintmap_encoder(hint_features)
        pred_logit, aux_pred_logit = self.ritm_hrnet(input_image, encoded_input_hint_heatmap, input_hint_heatmap=input_hint_heatmap)

        pred_logit = F.interpolate(pred_logit, size=self.config.Dataset.image_size, mode='bilinear', align_corners=True)
        aux_pred_logit = F.interpolate(aux_pred_logit, size=self.config.Dataset.image_size, mode='bilinear', align_corners=True)

        return pred_logit, aux_pred_logit

    def forward(self, batch):
        #make label, hint heatmap
        with torch.no_grad():
            batch.label.coord = batch.label.coord.to(self.device)
            batch.label.heatmap = self.heatmap_maker.coord2heatmap(batch.label.coord)
            batch.hint.heatmap = torch.zeros_like(batch.label.heatmap)
            for i in range(batch.label.heatmap.shape[0]):
                if batch.hint.index[i] is not None:
                    batch.hint.heatmap[i, batch.hint.index[i]] = batch.label.heatmap[i, batch.hint.index[i]]

        input_image = batch.input_image.to(self.device)
        input_hint_heatmap = batch.hint.heatmap.to(self.device)

        if batch.is_training:
            # 1. random number of iteration without update
            if self.config.Model.no_iterative_training:
                pred_heatmap = None
            else:
                with torch.no_grad():
                    self.eval()
                    num_iters = np.random.randint(0, self.max_iter)
                    pred_heatmap = torch.zeros_like(input_hint_heatmap)
                    for click_indx in range(num_iters):
                        # prediction
                        pred_logit, aux_pred_logit = self.forward_model(input_image, input_hint_heatmap, pred_heatmap)
                        pred_heatmap = pred_logit.sigmoid()

                        # hint update (training 때는 hint를 줘가면서 update하는 과정을 거침)
                        batch = self.get_next_points(batch, pred_heatmap)
                        for i in range(batch.label.heatmap.shape[0]):
                            if batch.hint.index[i] is not None:
                                batch.hint.heatmap[i, batch.hint.index[i]] = batch.label.heatmap[i, batch.hint.index[i]]

                    self.train()
            # 2. forward for model update
            pred_logit, aux_pred_logit = self.forward_model(input_image, input_hint_heatmap, pred_heatmap)
            out = self.LossManager(pred_heatmap=pred_logit, label=batch.label)
            out.loss += self.LossManager.get_heatmap_loss(pred_heatmap=aux_pred_logit, label_heatmap=batch.label.heatmap)[0]
        else: # test
            # forward
            if self.config.Model.no_iterative_training:
                pred_heatmap = None
            else:
                if batch.get('prev_heatmap', None) is None:
                    pred_heatmap = torch.zeros_like(input_hint_heatmap)
                else:
                    pred_heatmap = batch.prev_heatmap.to(input_hint_heatmap.device)
            pred_logit, aux_pred_logit = self.forward_model(input_image, input_hint_heatmap, pred_heatmap)
            out = self.LossManager(pred_heatmap=pred_logit, label=batch.label)
        return out, batch

    def get_next_points(self, batch, pred_heatmap):
        worst_index = self.find_worst_pred_index(batch, pred_heatmap)
        for i, idx in enumerate(batch.hint.index):
            if idx is None:
                batch.hint.index[i] = worst_index[i]  # (batch, 1)
            else:
                if not torch.is_tensor(idx):
                    batch.hint.index[i] = torch.tensor(batch.hint.index[i], dtype=torch.long, device=worst_index.device)
                batch.hint.index[i] = torch.cat((batch.hint.index[i], worst_index[i]))  # ... (batch, max hint)
        return batch

    def find_worst_pred_index(self, batch, pred_heatmap):
        # ==== calculate pixel MRE ====
        with torch.no_grad():
            hargmax_coord_pred = heatmap2hargmax_coord(pred_heatmap)
            batch_metric_value = torch.sqrt(torch.sum((hargmax_coord_pred-batch.label.coord)**2,dim=-1)) #MRE (batch, 13)
            for j, idx in enumerate(batch.hint.index):
                if idx is not None:
                    batch_metric_value[j, idx] = torch.full_like(batch_metric_value[j,idx], -1000)
            worst_index = batch_metric_value.argmax(-1, keepdim=True)
        return worst_index


class ScaleLayer(nn.Module):
    def __init__(self, init_value=1.0, lr_mult=1):
        super().__init__()
        self.lr_mult = lr_mult
        self.scale = nn.Parameter(
            torch.full((1,), init_value / lr_mult, dtype=torch.float32)
        )

    def forward(self, x):
        scale = torch.abs(self.scale * self.lr_mult)
        return x * scale