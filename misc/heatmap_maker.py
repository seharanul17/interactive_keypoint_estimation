import torch

class HeatmapMaker():
    def __init__(self, config):
        self.config = config
        self.image_size = config.Dataset.image_size
        self.heatmap_std = config.Dataset.heatmap_std
        self.morph_pairs = config.Morph.pairs
        # self.morph_unit_vector_size = config.Morph.unit_vector_size

    def make_gaussian_heatmap(self, mean, size, std):
        # coord : (13,2)
        mean = mean.unsqueeze(1).unsqueeze(1)
        var = std ** 2  # 64, 1
        grid = torch.stack(torch.meshgrid([torch.arange(size[0]), torch.arange(size[1])]), dim=-1).unsqueeze(0)
        grid = grid.to(mean.device)
        x_minus_mean = grid - mean  # 13, 1024, 1024, 2

        # (x-u)^2: (13, 512, 512, 2)  inverse_cov: (1, 1, 1, 1) > (13, 512, 512)
        gaus = (-0.5 * (x_minus_mean.pow(2) / var)).sum(-1).exp()
        if self.config.Dataset.get('heatmap_encoding_maxone', False):
            gaus /= gaus.max(1, keepdim=True)[0].max(2, keepdim=True)[0] #(13, 512, 512)
        # (13, 512, 512)
        return gaus



    # subpixel paper encoding
    def sample_gaussian_heatmap(self, mean):
        integer = torch.floor(mean)
        rest = mean - integer
        grid_gaussian = self.make_gaussian_heatmap(rest, (2, 2), 0.5)
        sampled_offset = torch.multinomial(grid_gaussian.reshape(mean.shape[0], 4), num_samples=1)
        row = torch.floor(torch.div(sampled_offset, 2)) #rounding_mode='floor')  # (13,1)
        col = torch.remainder(sampled_offset, 2)  # (13,1)

        new_sampled_offset = torch.cat((row, col), dim=1)
        return new_sampled_offset + integer

    def coord2heatmap(self, coord):
        # coord : (batch, 13, 2), torch tensor, gpu
        with torch.no_grad():
            if self.config.Dataset.get('subpixel_heatmap_encoding', False):
                coord = torch.stack([self.sample_gaussian_heatmap(coord_item) for coord_item in coord])
            heatmap = torch.stack([
                self.make_gaussian_heatmap(coord_item, size=self.image_size, std=self.heatmap_std) for coord_item in coord])
        return heatmap

    def get_heatmap2sargmax_coord(self, pred_heatmap):
        if self.config.Dataset.subpixel_decoding:
            pred_coord = self.heatmap2subpixel_argmax_coord(pred_heatmap)
        else:
            pred_coord = self.heatmap2sargmax_coord(pred_heatmap)
        return pred_coord

    def heatmap2sargmax_coord(self, heatmap):
        # heatmap: batch, 13, 1024, 1024 (batch, c, row, column)

        if self.config.Dataset.label_smoothing:
            heatmap = torch.clamp((heatmap - 0.1) / 0.8, 0, 1)

        pred_col = torch.sum(heatmap, (-2))  # bach, c, column
        pred_row = torch.sum(heatmap, (-1))  # batch, c, row

        # 1, 1, 1024
        mesh_c = torch.arange(pred_col.shape[-1]).unsqueeze(0).unsqueeze(0).to(heatmap.device)
        mesh_r = torch.arange(pred_row.shape[-1]).unsqueeze(0).unsqueeze(0).to(heatmap.device)

        # batch, 13
        coord_c = torch.sum(pred_col * mesh_c, (-1)) / torch.sum(pred_col, (-1))
        coord_r = torch.sum(pred_row * mesh_r, (-1)) / torch.sum(pred_row, (-1))

        # batch, 13, 2 (row, column)
        coord = torch.stack((coord_r, coord_c), -1)

        return coord

    def heatmap2subpixel_argmax_coord(self, heatmap):  # row: y, col: x  tensor->(row, col)=(y,x)
        hard_coord = heatmap2hargmax_coord(heatmap).to(heatmap.device) #(batch, 13, 2)
        patch_size = self.config.Dataset.subpixel_decoding_patch_size # scalar
        reshaped_heatmap = heatmap.reshape(heatmap.shape[0] * heatmap.shape[1], 1, heatmap.shape[2], heatmap.shape[3])
        reshaped_hard_coord = hard_coord.reshape(hard_coord.shape[0] * hard_coord.shape[1], 1,
                                                 hard_coord.shape[2])  # (bk, 1, 2)

        patch_index = torch.arange(patch_size, device=heatmap.device) - patch_size // 2  # (5)
        patch_index_y = patch_index[:, None].expand(patch_size, patch_size)[None, :]  # (1, 5, 5)
        patch_index_x = patch_index[None, :].expand(patch_size, patch_size)[None, :]  # (1, 5, 5)
        patch_index_y = patch_index_y + reshaped_hard_coord[:, :, 0:1]  # (bk, 5, 5)
        patch_index_x = patch_index_x + reshaped_hard_coord[:, :, 1:2]

        # pad heatmap
        padded_reshaped_heatmap = torch.nn.functional.pad(reshaped_heatmap,
                                                          (patch_size, patch_size, patch_size, patch_size),
                                                          mode='constant', value=0.0) # pad (left, right, top, bottom)
        pad_patch_index_y = patch_size + patch_index_y
        pad_patch_index_x = patch_size + patch_index_x

        patch = padded_reshaped_heatmap[:, :, pad_patch_index_y.long(), pad_patch_index_x.long()]

        patch = patch.diagonal(dim1=0, dim2=2).permute(dims=[3, 0, 1, 2])
        patch = patch.reshape(heatmap.shape[0], heatmap.shape[1], patch_size, patch_size) #b, 13, 5, 5
        patch = (patch*10).reshape(patch.shape[0], patch.shape[1], -1).softmax(-1).reshape(patch.shape)

        soft_patch_offset = self.heatmap2sargmax_coord(patch) #batch, 13, 2
        final_coord = hard_coord + soft_patch_offset - patch_size//2
        return final_coord

    def get_morph(self, points):
        # normalize
        points = self.points_normalize(points, dim0=self.image_size[0], dim1=self.image_size[1])

        dist_pairs = torch.tensor(self.morph_pairs[0])
        vec_pairs = torch.tensor(self.morph_pairs[1])

        # dist
        # batch, 13, 2
        from_vec = points[:, dist_pairs[:, 0]]
        to_vec = points[:, dist_pairs[:, 1]]
        diffs = to_vec - from_vec
        pred_dist = torch.norm(diffs, dim = -1).unsqueeze(-1) # (batch, 16,1)

        # unit_vec_pairs
        if self.config.Morph.threePointAngle:
            # batch, 13, 2
            x = points[:, vec_pairs[:, 0]]
            y = points[:, vec_pairs[:, 1]]
            z = points[:, vec_pairs[:, 2]]
            pred_angle = self.get_angle(x,y,z)
        else:
            from_vec = points[:, vec_pairs[:, 0]]
            to_vec = points[:, vec_pairs[:, 1]]
            diffs = to_vec - from_vec
            pred_angle = diffs

        return pred_dist, pred_angle

    def get_angle(self, x, y, z):
        # y가 꼭짓점임. z-y-x
        N = x.shape[0] * x.shape[1]
        delta_vector_1 = torch.reshape(x - y, (N, 2))  # batch*16, 2
        delta_vector_2 = torch.reshape(z - y, (N, 2))

        # (y,x) = (row, col)
        angle_1 = torch.atan2(delta_vector_1[:, 0], delta_vector_1[:, 1] + 1e-8)
        angle_2 = torch.atan2(delta_vector_2[:, 0], delta_vector_2[:, 1] + 1e-8)

        delta_angle = angle_2 - angle_1

        angle_vector = torch.stack((torch.cos(delta_angle), torch.sin(delta_angle)), -1)

        angle_vector = torch.reshape(angle_vector, (x.shape[0], x.shape[1], 2))
        return angle_vector

    def points_normalize(self, points, dim0, dim1):
        new_coord = torch.zeros_like(points)
        new_coord[..., 0] = points[..., 0] / dim0
        new_coord[..., 1] = points[..., 1] / dim1

        return new_coord

def heatmap2hargmax_coord(heatmap):
    b, c, row, column = heatmap.shape
    heatmap = heatmap.reshape(b, c, -1)
    max_indices = heatmap.argmax(-1)
    keypoint = torch.zeros(b, c, 2, device=heatmap.device)
    # keypoint[:, :, 0] = torch.floor(torch.div(max_indices, column)) # old environment
    keypoint[:, :, 0] = torch.div(max_indices, column, rounding_mode='floor')
    keypoint[:, :, 1] = max_indices % column
    return keypoint

