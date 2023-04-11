# Copyright (c) 2023, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://github.com/NVlabs/prismer/blob/main/LICENSE

import glob

from torch.utils.data import Dataset
from PIL import ImageFile
from dataset.utils import *

ImageFile.LOAD_TRUNCATED_IMAGES = True

class EyalDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.img_dir = root_dir
        self.img_list = os.listdir(root_dir)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_list[idx])
        with Image.open(img_path) as img:
            img_size = img.size
            img = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
        return img, img_path, img_size

class Dataset(Dataset):
    def __init__(self, data_path, transform):
        self.data_path = data_path
        self.transform = transform
        data_folders = glob.glob(f'{data_path}/*/')
        self.data_list = [data for f in data_folders for data in glob.glob(f + '*.JPEG')]
        self.data_list += [data for f in data_folders for data in glob.glob(f + '*.jpg')]

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        image_path = self.data_list[index]
        image = Image.open(image_path).convert('RGB')
        img_size = [image.size[0], image.size[1]]
        image = self.transform(image)
        return image, image_path, img_size
