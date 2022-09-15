import os
import copy
from munch import Munch
import numpy as np

import torch

import dataset.transforms as transforms
# from detectron2.data import transforms as T

class Dataset(torch.utils.data.Dataset):
    def __init__(self, config, split:str, data:list):

        config.DICT_KEY = Munch.fromDict({})
        config.DICT_KEY.IMAGE = 'image'
        config.DICT_KEY.BBOX = 'bbox_{}'.format(config.Dataset.image_size[0])
        config.DICT_KEY.POINTS = 'points_{}'.format(config.Dataset.image_size[0])
        config.DICT_KEY.RAW_SIZE = 'raw_size_row_col'
        config.DICT_KEY.PSPACE = 'pixelSpacing'




        # init
        self.config = config
        self.split = split
        self.data = data

        if self.split == 'train':
            self.transformer = transforms.default_aug(self.config.Dataset.image_size)

        else:
            self.transformer = transforms.fake

        self.loadimage = self.load_npy

        for item in self.data:
            item[self.config.DICT_KEY.IMAGE] = item[self.config.DICT_KEY.IMAGE].replace('.png', '.npy')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # indexing
        item = self.data[index]

        # image load
        img_path = os.path.join(self.config.PATH.DATA.IMAGE, item[self.config.DICT_KEY.IMAGE])
        img, row, column = self.loadimage(img_path, item[self.config.DICT_KEY.RAW_SIZE])

        # pixel spacing
        pspace_list = item[self.config.DICT_KEY.PSPACE] # row, column
        raw_size_and_pspace = torch.tensor([row, column] + pspace_list)

        # points load (13,2) (column, row)==(xy)
        coords = copy.deepcopy(item[self.config.DICT_KEY.POINTS])
        coords.append([1.0,1.0])


        transformed = self.transformer(image=img, keypoints=coords)
        img, coords = transformed["image"], transformed["keypoints"]
        additional = torch.tensor([])

        coords = np.array(coords)

        # np array to tensor (800, 640)=(row, col)
        img = torch.tensor(img, dtype=torch.float)
        img = img.permute(2, 0, 1)
        img /= 255.0 # 0~255 to 0~1
        img = img * 2 - 1 # 0~1 to -1~1


        coords = torch.tensor(copy.deepcopy(coords[:, ::-1]), dtype=torch.float)
        morph_loss_mask = (coords[-1] == torch.tensor([1.0, 1.0], dtype=torch.float)).all()
        coords = coords[:-1]

        # hint
        if self.split == 'train':
            # random hint
            num_hint = np.random.choice(range(self.config.Dataset.num_keypoint ), size=None, p=self.config.Hint.num_dist)
            hint_indices = np.random.choice(range(self.config.Dataset.num_keypoint ), size=num_hint, replace=False) #[1,2,3]
        else:
            hint_indices = None

        return img_path, img, raw_size_and_pspace, hint_indices, coords, additional, index, morph_loss_mask

    def load_npy(self, img_path, size=None):
        img = np.load(img_path)
        if size is not None:
            row, column = size
        else:
            row, column = img.shape[:2]
        return img, row, column

def collate_fn(batch):
    batch = list(zip(*batch))
    batch_dict = {
        'input_image_path':batch[0], # list
        'input_image':torch.stack(batch[1]),
        'label':{'morph_offset':torch.stack(batch[2]),
                'coord': torch.stack(batch[4]),
                'morph_loss_mask':torch.stack(batch[7])
                 },
        'pspace':torch.stack(batch[2]),
        'hint':{'index': list(batch[3])},
        'additional': torch.stack(batch[5]),
        'index':list(batch[6])
    }
    return Munch.fromDict(batch_dict)