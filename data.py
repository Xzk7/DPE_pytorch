import os
import numpy as np
from PIL import Image
from scipy import misc

import nltk
from nltk.tokenize import RegexpTokenizer

import torch
import torch.utils.data as data
from torch.utils.serialization import load_lua
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from utils import *
from parameter import *
import warnings


def split_sentence_into_words(sentence):
    tokenizer = RegexpTokenizer(r'\w+')
    return tokenizer.tokenize(sentence.lower())

class Load_Data(data.Dataset):
    def __init__(self, img_root1, img_root2, classes_filename1, classes_filename2, img_transform=None, suffix = ['.dng', '.tif']):
        super(Load_Data, self).__init__()

        self.img_transform_512 = transforms.Compose([
                                      transforms.Resize(512+8),
                                      transforms.RandomCrop(512),
                                      # transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        # if self.img_transform == None:
        #     self.img_transform = transforms.ToTensor()

        self.data1 = self._load_dataset(img_root1, classes_filename1, suffix[0])
        self.data2 = self._load_dataset(img_root2, classes_filename2, suffix[1])

    def _load_dataset(self, img_root, classes_filename, suffix):
        output = []

        with open(os.path.join(img_root, classes_filename)) as f:
            lines = f.readlines()
            count = 0
            for line in lines:
                filename = line.replace('\n', '')
                output.append(os.path.join(img_root, filename+suffix))
                count = count + 1
        print("total train num is %d"%count)
        return output

    def __len__(self):
        return len(self.data1)

    def __getitem__(self, index):
        img1_dir = self.data1[index]
        img1 = Image.open(img1_dir)
        img1 = self.img_transform_512(img1)

        img2_dir = self.data2[index]
        img2 = Image.open(img2_dir)
        img2 = self.img_transform_512(img2)
        return img1, img2

'''if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    config = get_parameters()
    dataset = Load_Data(config.img_root1, config.img_root2, config.train_file1, config.train_file2,
                        suffix=['.dng', '.tif'])

    data_loader = data.DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        drop_last=True
    )
    for i, (img1, img2) in enumerate(data_loader):
        print(img2.size())

#MIT_datasets = Load_Data('.\MIT_datasets', 'MIT_name.txt')'''

