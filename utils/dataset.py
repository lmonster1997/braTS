import os
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from utils.common import *
from scipy import ndimage
import numpy as np
from torchvision import transforms as T
import torch, os
from torch.utils.data import Dataset, DataLoader


class braTS_DataSet(Dataset):
    def __init__(self, crop_size, dataset_path, mode=None,fold=None,WTE=False):
        self.crop_size = crop_size
        self.dataset_path = dataset_path
        self.n_classes = 4
        self.augumentation = False
        self.random_crop = False
        self.mode = mode
        self.wte = WTE

        if mode == 'train':
            self.filename_list = load_file_name_list(os.path.join(dataset_path, 'name_list/train_name_list'+fold+'.txt'))
            self.augumentation = True
            self.random_crop = True

        elif mode == 'val':
            self.filename_list = load_file_name_list(os.path.join(dataset_path, 'name_list/val_name_list'+fold+'.txt'))
        elif mode == 'train_sample':
            self.filename_list = load_file_name_list(os.path.join(dataset_path, 'name_list/train_sample_name_list.txt'))

        elif mode == 'train_C':
            self.filename_list = load_file_name_list(
                os.path.join(dataset_path, 'name_list/train_C_name_list' + fold + '.txt'))
            self.augumentation = True
            self.random_crop = True

        elif mode == 'val_C':
            self.filename_list = load_file_name_list(
                os.path.join(dataset_path, 'name_list/val_C_name_list' + fold + '.txt'))
        elif mode == 'train_C_sample':
            self.filename_list = load_file_name_list(os.path.join(dataset_path, 'name_list/train_C_sample_name_list.txt'))

        else:
            raise TypeError('Dataset mode error!!! ')

    def __getitem__(self, index):
        if self.mode=='train_C' or self.mode=='val_C' or self.mode=='train_C_sample':
            data, target, cls, os_info, age = self.get_train_batch_by_index(crop_size=self.crop_size, index=index)
            return data, target, cls, os_info, age, self.filename_list[index]
        else:
            data, target, cls = self.get_train_batch_by_index(crop_size=self.crop_size, index=index)
            return data, target, cls, self.filename_list[index]

    def __len__(self):
        return len(self.filename_list)

    def get_train_batch_by_index(self, crop_size, index):
        if self.mode=='train_C' or self.mode=='val_C' or self.mode=='train_C_sample':
            data, mask, os_info, age = self.get_np_data(self.filename_list[index])
        else:
            data, mask = self.get_np_data(self.filename_list[index])
        if int(self.filename_list[index][-3:]) <= 259 or int(self.filename_list[index][-3:]) >= 336:
            cls = np.array(1)
        else:
            cls = np.array(0)
        if self.random_crop:
            sub_data, sub_mask = random_crop_3d(data, mask, crop_size)
        else:
            sub_data, sub_mask = data, mask
        if self.augumentation:
            sub_data, sub_mask_onehot = self.__data_aumentation(sub_data, sub_mask)
        else:
            if not self.wte:
                sub_data, sub_mask_onehot = torch.from_numpy(sub_data), torch.nn.functional.one_hot(
                    torch.from_numpy(np.array(sub_mask[0], dtype=int)), self.n_classes)
                sub_mask_onehot = np.transpose(sub_mask_onehot, (3, 0, 1, 2))
            else:
                sub_data, sub_mask_onehot = torch.from_numpy(sub_data), sub_mask.copy()
        if self.mode=='train_C' or self.mode=='val_C' or self.mode=='train_C_sample':
            return sub_data, sub_mask_onehot, cls, os_info, age
        else:
            return sub_data, sub_mask_onehot, cls

    def get_np_data(self, filename):

        # 读取数据
        data_path = self.dataset_path + '/data/' + filename
        data_img = data_path + '/data.npy'
        if self.wte:
            mask_img = data_path + '/mask_wte.npy'
        else:
            mask_img = data_path + '/mask.npy'
        data_np = np.load(os.path.join(data_path, data_img))
        mask_np = np.load(os.path.join(data_path, mask_img))

        if self.mode == 'train_C' or self.mode == 'val_C' or self.mode == 'train_C_sample':
            osagepath = data_path + '/osage.npy'
            osage_np = np.load(os.path.join(data_path, osagepath))
            os_np = osage_np[0,0]
            age_np = osage_np[0,1]

            return data_np, mask_np, os_np, age_np
        return data_np, mask_np

    def __data_aumentation(self, data, mask):

        im, gt = data, mask
        aug_choice = np.random.randint(4)

        # flip
        if aug_choice == 1:
            im, gt = flip3D(im, gt)
        # rotation
        if aug_choice == 2:
            im, gt = rotation3D(im, gt)
        # flip + rotation
        if aug_choice == 3:
            im, gt = flip3D(im, gt)
            im, gt = rotation3D(im, gt)

        # 幂律伽马变换
        if np.random.random_sample()<0.5:
            im, gt = brightness(im, gt)

        # 弹性变形
        if np.random.random_sample() < 0.5:
            im, gt = elastic(im, gt)

        im = torch.from_numpy(im.copy())
        if not self.wte:
            gt = torch.nn.functional.one_hot(torch.from_numpy(np.array(gt[0].copy(), dtype=int)), self.n_classes)
            gt = np.transpose(gt, (3, 0, 1, 2))
        else:
            gt = gt.copy()
        return im, gt

class braTS_DataSet_W(Dataset):
    def __init__(self, crop_size, dataset_path, mode=None,fold=None,WTE=False):
        self.crop_size = crop_size
        self.dataset_path = dataset_path
        self.n_classes = 4
        self.augumentation = False
        self.random_crop = False
        self.mode = mode
        self.wte = WTE

        if mode == 'train':
            self.filename_list = load_file_name_list(os.path.join(dataset_path, 'name_list/train_name_list'+fold+'.txt'))
            self.augumentation = True
            self.random_crop = True

        elif mode == 'val':
            self.filename_list = load_file_name_list(os.path.join(dataset_path, 'name_list/val_name_list'+fold+'.txt'))
        elif mode == 'train_sample':
            self.filename_list = load_file_name_list(os.path.join(dataset_path, 'name_list/train_sample_name_list.txt'))

        else:
            raise TypeError('Dataset mode error!!! ')

    def __getitem__(self, index):
        data, target, sub_pred_mask_onehot = self.get_train_batch_by_index(crop_size=self.crop_size, index=index)
        return data, target, sub_pred_mask_onehot, self.filename_list[index]

    def __len__(self):
        return len(self.filename_list)

    def get_train_batch_by_index(self, crop_size, index):
        data, mask,pred_mask = self.get_np_data(self.filename_list[index])
        mask = np.concatenate((mask, pred_mask[np.newaxis,:]), axis=0)
        if self.random_crop:
            sub_data, sub_mask = random_crop_3d(data, mask, crop_size)
        else:
            sub_data, sub_mask = data, mask
        if self.augumentation:
            sub_data, sub_mask = self.__data_aumentation(sub_data, sub_mask)

        sub_mask_onehot, sub_pred_mask_onehot = sub_mask[0].copy(), sub_mask[1].copy()
        sub_data, sub_mask_onehot = torch.from_numpy(sub_data.copy()), torch.nn.functional.one_hot(
            torch.from_numpy(np.array(sub_mask_onehot, dtype=int).copy()), self.n_classes)
        sub_mask_onehot = np.transpose(sub_mask_onehot, (3, 0, 1, 2))
        sub_pred_mask_onehot = torch.nn.functional.one_hot(
            torch.from_numpy(np.array(sub_pred_mask_onehot, dtype=int).copy()), self.n_classes)
        sub_pred_mask_onehot = np.transpose(sub_pred_mask_onehot, (3, 0, 1, 2))

        return sub_data, sub_mask_onehot, sub_pred_mask_onehot

    def get_np_data(self, filename):

        # 读取数据
        data_path = self.dataset_path + '/data/' + filename
        data_img = data_path + '/data.npy'
        if self.wte:
            mask_img = data_path + '/mask_wte.npy'
        else:
            mask_img = data_path + '/mask.npy'
        pred_mask = data_path + '/pred_mask.npy'
        data_np = np.load(os.path.join(data_path, data_img))
        mask_np = np.load(os.path.join(data_path, mask_img))
        pred_mask_np = np.load(os.path.join(data_path, pred_mask))

        return data_np, mask_np, pred_mask_np

    def __data_aumentation(self, data, mask):

        im, gt = data, mask
        aug_choice = np.random.randint(4)

        # flip
        if aug_choice == 1:
            im, gt = flip3D(im, gt)
        # rotation
        if aug_choice == 2:
            im, gt = rotation3D(im, gt)
        # flip + rotation
        if aug_choice == 3:
            im, gt = flip3D(im, gt)
            im, gt = rotation3D(im, gt)

        # 幂律伽马变换
        if np.random.random_sample()<0.5:
            im, gt = brightness(im, gt)

        # 弹性变形
        if np.random.random_sample() < 0.5:
            im, gt = elastic(im, gt)

        return im, gt


class braTS_DataSet2(Dataset):
    def __init__(self, crop_size, dataset_path, mode=None):
        self.crop_size = crop_size
        self.n_classes = 4
        self.dataset_path = dataset_path
        self.augumentation = False
        self.random_crop = False
        self.mode = mode

        if mode == 'train':
            self.filename_list = load_file_name_list(os.path.join(dataset_path, 'name_list/train_name_list1.txt'))

            self.augumentation = True
            self.random_crop = True

        elif mode == 'val':
            self.filename_list = load_file_name_list(os.path.join(dataset_path, 'name_list/val_name_list1.txt'))

        elif mode == 'train_sample':
            self.filename_list = load_file_name_list(os.path.join(dataset_path, 'name_list/train_name_list.txt'))
        else:
            raise TypeError('Dataset mode error!!! ')

    def __getitem__(self, index):
        data, target, survival_info = self.get_train_batch_by_index(crop_size=self.crop_size, index=index)

        return data, target, int(survival_info), self.filename_list[index]

    def __len__(self):
        return len(self.filename_list)

    def get_train_batch_by_index(self, crop_size, index):
        data, mask, survival_info = self.get_np_data(self.filename_list[index])
        if self.random_crop:
            sub_data, sub_mask = random_crop_3d(data, mask, crop_size)
        else:
            sub_data, sub_mask = data, mask
        if self.augumentation:
            sub_data, sub_mask_onehot = self.__data_aumentation(sub_data, sub_mask)
        else:
            sub_data, sub_mask_onehot = torch.from_numpy(sub_data), torch.nn.functional.one_hot(
                torch.from_numpy(np.array(sub_mask[0], dtype=int)), self.n_classes)
            sub_mask_onehot = np.transpose(sub_mask_onehot, (3, 0, 1, 2))
        return sub_data, sub_mask_onehot, survival_info

    def get_np_data(self, filename):

        # 读取数据
        data_path = self.dataset_path + '/data/' + filename
        data_img = data_path+'/data.npy'
        mask_img = data_path+'/mask.npy'
        survival_info_path = data_path+'/os.npy'
        data_np = np.load(data_img)
        mask_np = np.load(mask_img)
        survival_info = int(np.load(survival_info_path))
        if survival_info==0:
            survival_info_c=3
        elif survival_info>450:
            survival_info_c = 2
        elif survival_info<=450 and survival_info>300:
            survival_info_c = 1
        elif survival_info<=300 and survival_info>0:
            survival_info_c = 0
        else:
            survival_info_c = 10000
        return data_np, mask_np, survival_info_c

    def __data_aumentation(self, data, mask):

        im, gt = data, mask
        aug_choice = np.random.randint(4)

        # flip
        if aug_choice == 1:
            im, gt = flip3D(im, gt)
        # rotation
        if aug_choice == 2:
            im, gt = rotation3D(im, gt)
        # flip + rotation
        if aug_choice == 3:
            im, gt = flip3D(im, gt)
            im, gt = rotation3D(im, gt)

        # print(np.max(gt[1, :], ))

        # 幂律伽马变换
        if np.random.random_sample()<0.5:
            im, gt = brightness(im, gt)

        # 弹性变形
        if np.random.random_sample() < 0.5:
            im, gt = elastic(im, gt)


        im = torch.from_numpy(im.copy())
        gt = torch.nn.functional.one_hot(torch.from_numpy(np.array(gt[0].copy(), dtype=int)), self.n_classes)
        gt = np.transpose(gt, (3, 0, 1, 2))
        return im, gt

class braTS_DataSet_val(Dataset):
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path

        self.filename_list = load_file_name_list(os.path.join(dataset_path, 'name_list/val_name_list.txt'))

    def __getitem__(self, index):
        data = self.get_train_batch_by_index(index=index)

        return data, self.filename_list[index]

    def __len__(self):
        return len(self.filename_list)

    def get_train_batch_by_index(self, index):
        data = self.get_np_data(self.filename_list[index])

        return data

    def get_np_data(self, filename):

        # 读取数据
        data_path = self.dataset_path + '/data/' + filename
        data_img = data_path + '/data.npy'
        data_np = np.load(os.path.join(data_path, data_img))
        return data_np




class braTS_DataSet4(Dataset):
    def __init__(self, crop_size, dataset_path, mode=None):
        self.crop_size = crop_size
        self.dataset_path = dataset_path
        self.augumentation = False
        self.random_crop = False
        self.mode = mode

        if mode == 'train':
            self.filename_list = load_file_name_list(os.path.join(dataset_path, 'name_list/train_name_list1.txt'))

            self.augumentation = True
            self.random_crop = True
        elif mode == 'train_experiment':
            self.filename_list = load_file_name_list(
                os.path.join(dataset_path, 'name_list/train_name_list.txt'))
            # self.augumentation = True
            self.random_crop = False

        elif mode == 'val':
            self.filename_list = load_file_name_list(os.path.join(dataset_path, 'name_list/val_name_list1.txt'))

        elif mode == 'train_sample':
            self.filename_list = load_file_name_list(os.path.join(dataset_path, 'name_list/train_name_list.txt'))
        elif mode == 'test':
            self.filename_list = load_file_name_list(os.path.join(dataset_path, 'name_list/test_name_list.txt'))
        else:
            raise TypeError('Dataset mode error!!! ')

    def __getitem__(self, index):
        data, target, survival_info = self.get_train_batch_by_index(crop_size=self.crop_size, index=index)

        return data, target, int(survival_info), self.filename_list[index]

    def __len__(self):
        return len(self.filename_list)

    def get_train_batch_by_index(self, crop_size, index):
        data, mask, survival_info = self.get_np_data(self.filename_list[index])
        if self.random_crop:
            sub_data, sub_mask = random_crop_3d(data, mask, crop_size)
        else:
            sub_data, sub_mask = data, mask
        if self.augumentation:
            sub_data, sub_mask_onehot = self.__data_aumentation(sub_data, sub_mask)
        else:
            sub_data, sub_mask_onehot = torch.from_numpy(sub_data), torch.from_numpy(sub_mask)
        return sub_data, sub_mask_onehot, survival_info

    def get_np_data(self, filename):

        # 读取数据
        data_path = self.dataset_path + '/data/' + filename
        data_img = data_path+'/data.npy'
        mask_img = data_path+'/mask_wte.npy'
        survival_info_path = data_path+'/os.npy'
        data_np = np.load(data_img)
        mask_np = np.load(mask_img)
        survival_info = int(np.load(survival_info_path))
        if survival_info==0:
            survival_info_c=0
        else:
            survival_info_c = 1
        return data_np, mask_np, survival_info_c

    def __data_aumentation(self, data, mask):

        im, gt = data, mask
        aug_choice = np.random.randint(4)

        # flip
        if aug_choice == 1:
            im, gt = flip3D(im, gt)
        # rotation
        if aug_choice == 2:
            im, gt = rotation3D(im, gt)
        # flip + rotation
        if aug_choice == 3:
            im, gt = flip3D(im, gt)
            im, gt = rotation3D(im, gt)

        # print(np.max(gt[1, :], ))

        # 幂律伽马变换
        if np.random.random_sample()<0.5:
            im, gt = brightness(im, gt)

        # 弹性变形
        if np.random.random_sample() < 0.5:
            im, gt = elastic(im, gt)


        im = torch.from_numpy(im.copy())
        gt = torch.from_numpy(gt.copy())
        return im, gt

# 测试代码
def main():
    fixd_path = r'/media/tiger/Disk0/lyu/DATA/Brain/Segment/processed_2020_std_withVal'
    dataset = braTS_DataSet2([128, 128, 128], fixd_path, index=[], mode='train_experiment')  # batch size

    data_loader = DataLoader(dataset=dataset, batch_size=1, num_workers=1, shuffle=True)
    for batch_idx, (data, target, name) in enumerate(data_loader):
        print(name)
        print(torch.sum(target[0,0,:]),torch.sum(target[0,1,:]),torch.sum(target[0,2,:]))


if __name__ == '__main__':
    main()
