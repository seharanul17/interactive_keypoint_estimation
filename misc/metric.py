from munch import Munch
from misc.heatmap_maker import HeatmapMaker, heatmap2hargmax_coord
import numpy as np
import torch
import torch.nn as nn

class MetricManager():
    def __init__(self, save_manager):
        self.config = save_manager.config

        if save_manager.config.Train.decision_metric.split('_')[-1] in ['MAE', 'MSE', 'MRE']:
            self.minimize_metric = True
        elif save_manager.config.Train.decision_metric.split('_')[-1] in ['AUC', 'ACC']:
            self.minimize_metric = False
        else:
            save_manager.write_log('ERROR::: Specify the decision metric in metric.py')
            raise()
        self.heatmap_maker = HeatmapMaker(save_manager.config)
        self.init_running_metric()
        self.device = self.config.MISC.device


    def init_running_metric(self):
        self.running_metric=Munch.fromDict({})
        for name in ['hargmax_pixel','hargmax_mm','sargmax_pixel','sargmax_mm']:
            for metric in self.config.Train.metric:
                metric_name = '{}_{}'.format(name,metric)
                self.running_metric[metric_name] = []

                if metric == 'MRE':
                    for standard in self.config.Train.SR_standard:
                        self.running_metric['{}_SR[{}]'.format(name, standard)] = []

    def average_running_metric(self):

        self.metric = Munch.fromDict({})
        for metric in self.running_metric:
            try:
                self.metric[metric] = np.mean(self.running_metric[metric])
            except:
                self.metric[metric] = np.mean([m.mean() for m in self.running_metric[metric]])
        self.init_running_metric()
        return self.metric

    def init_best_metric(self):
        if self.minimize_metric:
            best_metric = {self.config.Train.decision_metric: 1e4}
        else:
            best_metric =  {self.config.Train.decision_metric: -1e4}
        return best_metric

    def is_new_best(self, old, new):
        if old[self.config.Train.decision_metric] > new[self.config.Train.decision_metric]:
            if self.minimize_metric:
                return True
            else:
                return False
        else:
            if self.minimize_metric:
                return False
            else:
                return True

    def measure_metric(self, pred, label, pspace, metric_flag, average_flag):
        with torch.no_grad():
            # to cuda
            pspace = pspace.detach().to(self.device)
            label.coord = label.coord.detach().to(self.device)
            label.heatmap = label.heatmap.detach().to(self.device)
            pred.sargmax_coord = pred.sargmax_coord.detach().to(self.device)
            pred.heatmap = pred.heatmap.detach().to(self.device)

            if self.config.Model.facto_heatmap:
                pred.hargmax_coord = pred.hard_coord.float().to(self.device)
            else:
                # pred coord (hard-argmax)
                pred.hargmax_coord = heatmap2hargmax_coord(pred.heatmap)
            pred.hargmax_coord_mm = self.pixel2mm(self.config, pred.hargmax_coord, pspace)

            # pred coord (soft-argmax, model output)
            pred.sargmax_coord_mm = self.pixel2mm(self.config, pred.sargmax_coord, pspace)

            label.coord_mm = self.pixel2mm(self.config, label.coord, pspace)

            zip = [(pred.hargmax_coord, label.coord, 'hargmax_pixel'),
                   (pred.hargmax_coord_mm, label.coord_mm, 'hargmax_mm'),
                   (pred.sargmax_coord, label.coord, 'sargmax_pixel'),
                    (pred.sargmax_coord_mm, label.coord_mm, 'sargmax_mm')]

            if metric_flag:
                metric_list = self.config.Train.metric
            else:
                metric_list = [self.config.Train.decision_metric.split('_')[-1]]

            for item in zip:
                pred_coord, label_coord, name = item

                if 'MAE' in metric_list:
                    if average_flag:
                        self.running_metric['{}_MAE'.format(name)].append(nn.L1Loss()(pred_coord, label_coord).item())
                    else:
                        self.running_metric['{}_MAE'.format(name)].append(
                            (nn.L1Loss(reduction='none')(pred_coord, label_coord)).cpu())

                if 'RMSE' in metric_list:
                    self.running_metric['{}_RMSE'.format(name)].append(torch.sqrt(nn.MSELoss()(pred_coord, label_coord)).item())

                if 'MRE' in metric_list:
                    # (batch, 13, 2)
                    y_diff_sq = (pred_coord[:, :, 0] - label_coord[:, :, 0]) ** 2
                    x_diff_sq = (pred_coord[:, :, 1] - label_coord[:, :, 1]) ** 2
                    sqrt_x2y2 = torch.sqrt(y_diff_sq + x_diff_sq)  # (batch,13)
                    if average_flag:
                        mre = torch.mean(sqrt_x2y2).item()
                    else:
                        mre = sqrt_x2y2.cpu()
                    self.running_metric['{}_MRE'.format(name)].append(mre)

                    for standard in self.config.Train.SR_standard:
                        self.running_metric['{}_SR[{}]'.format(name, standard)].append((sqrt_x2y2 < standard).float().mean().item())


    def pixel2mm(self, config, points, pspace):
        mm_points = torch.zeros_like(points)
        pspace = pspace.to(points.device)
        mm_points[:, :, 1] = points[:, :, 1] / config.Dataset.image_size[1] * pspace[:, 1:2] * pspace[:, 3:4]
        mm_points[:, :, 0] = points[:, :, 0] / config.Dataset.image_size[0] * pspace[:, 0:1] * pspace[:, 2:3]
        return mm_points










