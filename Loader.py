from PIL import Image
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch.utils.data as data
from os import listdir
from os.path import join
import numpy as np
import torch
import os
import torch.nn as nn
from torch.autograd import Variable
import numpy as np


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


def default_loader(path):
    return Image.open(path).convert('RGB')


class Dataset(data.Dataset):
    def __init__(self, contentPath, stylePath):
        super(Dataset, self).__init__()
        self.contentPath = contentPath
        self.image_list = [x for x in listdir(contentPath) if is_image_file(x)]
        self.stylePath = stylePath

    def __getitem__(self, index):
        contentImgPath = os.path.join(self.contentPath, self.image_list[index])
        styleImgPath = os.path.join(self.stylePath, self.image_list[index])
        contentImg = default_loader(contentImgPath)
        styleImg = default_loader(styleImgPath)

        # resize
        contentImg = contentImg.resize((224, 224))
        styleImg = styleImg.resize((224, 224))

        # Preprocess Images
        contentImg = transforms.ToTensor()(contentImg)
        styleImg = transforms.ToTensor()(styleImg)
        return contentImg.squeeze(0), styleImg.squeeze(0), self.image_list[index]

    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return len(self.image_list)
