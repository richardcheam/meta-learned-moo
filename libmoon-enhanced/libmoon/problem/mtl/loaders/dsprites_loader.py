import torch
from sklearn.model_selection import train_test_split
import os
import numpy as np

from libmoon.util.constant import root_name


def load_dataset(path, s_labels):
    dataset = np.load(path)

    labels = ['color', 'shape', 'scale', 'orientation', 'position_x', 'position_y']

    s_labels_index = [labels.index(s) for s in s_labels]

    imgs = dataset['imgs']
    latents_classes = dataset['latents_classes']

    s_latents_classes = latents_classes[:, s_labels_index]

    return imgs, s_latents_classes


class DSpritesData(torch.utils.data.Dataset):

    def __init__(self, split='train', attribute_1 ='shape', attribute_2='scale'):

        self.attribute1 = attribute_1
        self.attribute2 = attribute_2

        middle_folder_name = os.path.join('libmoon', 'problem', 'mtl','mtl_data','dsprites')
        self.path = os.path.join(root_name, middle_folder_name, 'dsprites.npz')

        s_labels = [attribute_1, attribute_2]
        
        imgs, labels = load_dataset(path=self.path, s_labels=s_labels)

        # train/val/test split: 70/10/20 %
        x_train, x_test, y_train, y_test = train_test_split(imgs, labels,test_size=.2, random_state=1)
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=.125, random_state=1)

        x_train = torch.from_numpy(x_train.reshape(len(x_train), 1, 64, 64)).float()
        y_train = torch.from_numpy(y_train).long()
        x_test = torch.from_numpy(x_test.reshape(len(x_test), 1, 64, 64)).float()
        y_test = torch.from_numpy(y_test).long()
        x_val = torch.from_numpy(x_val.reshape(len(x_val), 1, 64, 64)).float()
        y_val = torch.from_numpy(y_val).long()

        if split == 'train':
            self.x = x_train
            self.y = y_train
        elif split == 'val':
            self.x = x_val
            self.y = y_val
        elif split == 'test':
            self.x = x_test
            self.y = y_test

    def __getitem__(self, index):
        return {'data': self.x[index], f'labels_{self.attribute1}': self.y[index, 0], f'labels_{self.attribute2}': self.y[index, 1]}

    def __len__(self):
        return len(self.x)

    def task_names(self):
        return [self.attribute1, self.attribute2]


if __name__ == '__main__':
    print("ok")

    middle_folder_name = os.path.join('libmoon', 'problem', 'mtl','mtl_data','dsprites')
    path = os.path.join(root_name, middle_folder_name, 'dsprites.npz')

    s_labels = ['shape', 'scale']
        
    imgs, labels = load_dataset(path=path, s_labels=s_labels)


