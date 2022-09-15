import torch
import torch.nn as nn
import torch.nn.functional as F
from munch import Munch

class LossManager():
    def __init__(self, config, heatmap_maker):
        self.config = config
        self.heatmap_maker = heatmap_maker
        self.mse_criterion = nn.MSELoss()
        self.mae_criterion = nn.L1Loss()
        self.bce_loss = nn.BCELoss()
        if self.config.Morph.use :
            if self.config.Morph.cosineSimilarityLoss:
                    self.angle_criterion = nn.CosineEmbeddingLoss()
            else:
                self.angle_criterion = self.mse_criterion

    def __call__(self, pred_heatmap, label):
        loss, pred_heatmap = self.get_heatmap_loss(pred_heatmap=pred_heatmap, label_heatmap=label.heatmap)
        pred_coord = self.heatmap_maker.get_heatmap2sargmax_coord(pred_heatmap=pred_heatmap)
        if self.config.Morph.use:
            morph_loss = self.get_morph_loss(pred_coord=pred_coord, label_coord=label.coord, morph_loss_mask=label.morph_loss_mask)
            loss = loss + morph_loss

        if self.config.Morph.coord_use:
            coord_loss = nn.L1Loss()(pred_coord, label.coord)
            loss = loss + 0.01 * coord_loss

        if torch.isnan(loss).item():
            print("========== ERROR ::: Loss is nan ===========")
            raise


        out = Munch.fromDict({'pred':{'sargmax_coord':pred_coord, 'heatmap':pred_heatmap}, 'loss':loss, })
        return out

    def get_heatmap_loss(self, pred_heatmap, label_heatmap, mse_flag=False):
        if mse_flag:
            heatmap_loss = self.mse_criterion(pred_heatmap, label_heatmap)
        else: # BCE loss
            pred_heatmap = pred_heatmap.sigmoid()
            heatmap_loss = self.bce_loss(pred_heatmap, label_heatmap)
        return heatmap_loss, pred_heatmap

    def get_morph_loss(self, pred_coord, label_coord, morph_loss_mask):
        pred_dist, pred_angle = self.heatmap_maker.get_morph(pred_coord)
        with torch.no_grad():
            label_dist, label_angle = self.heatmap_maker.get_morph(label_coord)

        if morph_loss_mask.sum()>0:
            pred_dist, pred_angle = pred_dist[morph_loss_mask], pred_angle[morph_loss_mask]
            label_dist, label_angle = label_dist[morph_loss_mask], label_angle[morph_loss_mask]

            if self.config.Morph.distance_l1:
                loss_dist = self.mae_criterion(pred_dist, label_dist)
            else:
                loss_dist = self.mse_criterion(pred_dist, label_dist)

            if self.config.Morph.cosineSimilarityLoss:
                N = pred_angle.shape[0] * pred_angle.shape[1]
                label_similairty = torch.ones(N, dtype=torch.long, device=pred_angle.device)
                loss_angle = self.angle_criterion(pred_angle.reshape(N, 2), label_angle.reshape(N, 2), label_similairty)
            else:
                pred_angle_normalized = torch.nn.functional.normalize(pred_angle, dim=-1)  # (batch, 13, 2)
                label_angle_normalized = torch.nn.functional.normalize(label_angle, dim=-1)
                loss_angle = self.angle_criterion(pred_angle_normalized, label_angle_normalized)

            return loss_dist * self.config.Morph.distance_lambda + loss_angle * self.config.Morph.angle_lambda
        return 0

