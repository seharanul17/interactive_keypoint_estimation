import os
import json
import numpy as np
from PIL import Image
import scipy.io
from tqdm.auto import tqdm
import copy

from albumentations import ( Resize, Compose, KeypointParams)

source_path = './data/dataset16/boostnet_labeldata/'
target_path = './data/'

base_image_path= os.path.join(source_path,'data')
base_label_path= os.path.join(source_path,'labels')




train_image_paths = []
test_image_paths = []

for name in os.listdir(os.path.join(base_image_path,'training')):
    p = os.path.join(base_image_path, 'training/', name)
    train_image_paths.append(p)

for name in os.listdir(os.path.join(base_image_path , 'test')):
    p = os.path.join(base_image_path, 'test/', name)
    test_image_paths.append(p)


train_label_paths = [(i+'.mat').replace('boostnet_labeldata/data/','boostnet_labeldata/labels/') for i in train_image_paths]
test_label_paths = [(i+'.mat').replace('boostnet_labeldata/data/','boostnet_labeldata/labels/') for i in test_image_paths]


train_image_paths.sort()
test_image_paths.sort()
train_label_paths.sort()
test_label_paths.sort()


# select random validation dataset (val size = 128)
val_idx = sorted(np.random.choice(range(len(train_image_paths)), size=128, replace=False, p=None))
train_idx = [i for i in range(len(train_image_paths)) if i not in val_idx]

val_image_paths = np.array(train_image_paths)[val_idx].tolist()
train_image_paths = np.array(train_image_paths)[train_idx].tolist()

val_label_paths = np.array(train_label_paths)[val_idx].tolist()
train_label_paths = np.array(train_label_paths)[train_idx].tolist()


# make json items
def make_data(image_paths, label_paths):
    data = []



    for idx in range(len(image_paths)):

        item = {'image':None, 'label':None, 'raw_size_row_col':None, 'pixelSpacing':[1,1]}


        # indexing
        image_path = image_paths[idx]
        label_path = label_paths[idx]

        # loda data
        img = np.repeat(np.array(Image.open(os.path.join(image_path)))[:,:,None], 3, axis=-1) # (row, col) -> (row,col,3)
        label = scipy.io.loadmat(os.path.join(label_path))['p2'] # x,y

        # make items
        item['image'] = image_path.replace(source_path, '') # remove base path
        item['label'] = label.tolist()
        item['raw_size_row_col'] = (img.shape[0], img.shape[1])

        data.append(item)
    return data

train_data = make_data(train_image_paths, train_label_paths)
val_data = make_data(val_image_paths, val_label_paths)
test_data = make_data(test_image_paths, test_label_paths)


def inference_aug(img_size):
    return Compose([
        Resize(img_size[0], img_size[1]),
    ], keypoint_params=KeypointParams(format='xy'))


# check keypoints are inside of the corresponding image
print('train')
remove_list = []
for t, item in (enumerate(train_data)):
    row, col = item['raw_size_row_col']
    coord = np.array(item['label'])
    if coord[:, 1].max() > row:
        print('error:', t, '- exceed max y')
        remove_list.append(t)
    elif coord[:, 0].max() > col:
        print('error:', t, '- excced max x')
        remove_list.append(t)
for t in remove_list:
    del train_data[t]

print('val')
remove_list=[]
for t, item in (enumerate(val_data)):
    row, col = item['raw_size_row_col']
    coord = np.array(item['label'])
    if coord[:, 1].max() > row:
        print('error:', t, '- exceed max y')
    if coord[:, 0].max() > col:
        print('error:', t, '- excced max x')
for t in remove_list:
    del val_data[t]


map_dic = {}
for sizes in [(512, 256)]:
    aug = inference_aug((sizes[0], sizes[1]))
    print(sizes, 'starts')
    size = sizes[0]

    if not os.path.exists('{}/dataset16_{}'.format(target_path,size)):
        os.makedirs('{}/dataset16_{}'.format(target_path,size))

    for temp in ['train', 'val', 'test']:
        print(temp, 'starts')

        if temp == 'train':
            table = copy.deepcopy(train_data)
        elif temp == 'val':
            table = copy.deepcopy(val_data)
        else:
            table = copy.deepcopy(test_data)

        for t, item in tqdm(enumerate(table)):
            # image load
            img_path = os.path.join(
                source_path,
                item['image'])
            img = np.repeat(np.array(Image.open(img_path))[:, :, None], 3, axis=-1)  # (row, col) -> (row,col,3)

            points = item['label']

            if img_path not in map_dic:
                map_dic[img_path] = {'points': points}

            transformed = aug(image=img, keypoints=points)
            img, points = transformed["image"], transformed["keypoints"]

            img_save_path = os.path.join(target_path,'dataset16_{}'.format(size),
                                 item['image'].replace('.jpg', '.npy'))
            if not os.path.exists( os.path.dirname(img_save_path)):
                os.makedirs(os.path.dirname(img_save_path))
            np.save(img_save_path, img)
            item['points_{}'.format(size)] = points
            item['image'] = item['image'].replace('.jpg', '.npy')

            map_dic[img_path].update({'points_{}'.format(size): points})

        with open(os.path.join(target_path,'dataset16_{}'.format(size),'{}.json'.format(temp)), 'w') as f:
            json.dump(table, f)
