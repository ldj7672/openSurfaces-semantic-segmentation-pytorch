import json
import os

import cv2
import numpy as np
import torch
from scipy.misc import imread, imresize
from torchvision import transforms

# from . import opensurfaces_dataset import os_dataset
from .opensurfaces_dataset import os_dataset

import bisect
import warnings

from torch._utils import _accumulate
from torch import randperm


class Dataset(object):
    """An abstract class representing a Dataset.

    All other datasets should subclass it. All subclasses should override
    ``__len__``, that provides the size of the dataset, and ``__getitem__``,
    supporting integer indexing in range from 0 to len(self) exclusive.
    """

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __add__(self, other):
        return ConcatDataset([self, other])


class TensorDataset(Dataset):
    """Dataset wrapping data and target tensors.

    Each sample will be retrieved by indexing both tensors along the first
    dimension.

    Arguments:
        data_tensor (Tensor): contains sample data.
        target_tensor (Tensor): contains sample targets (labels).
    """

    def __init__(self, data_tensor, target_tensor):
        assert data_tensor.size(0) == target_tensor.size(0)
        self.data_tensor = data_tensor
        self.target_tensor = target_tensor

    def __getitem__(self, index):
        return self.data_tensor[index], self.target_tensor[index]

    def __len__(self):
        return self.data_tensor.size(0)


class ConcatDataset(Dataset):
    """
    Dataset to concatenate multiple datasets.
    Purpose: useful to assemble different existing datasets, possibly
    large-scale datasets as the concatenation operation is done in an
    on-the-fly manner.

    Arguments:
        datasets (iterable): List of datasets to be concatenated
    """

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    def __init__(self, datasets):
        super(ConcatDataset, self).__init__()
        assert len(datasets) > 0, 'datasets should not be an empty iterable'
        self.datasets = list(datasets)
        self.cumulative_sizes = self.cumsum(self.datasets)

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx]

    @property
    def cummulative_sizes(self):
        warnings.warn("cummulative_sizes attribute is renamed to "
                      "cumulative_sizes", DeprecationWarning, stacklevel=2)
        return self.cumulative_sizes


class Subset(Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)


def random_split(dataset, lengths):
    """
    Randomly split a dataset into non-overlapping new datasets of given lengths
    ds

    Arguments:
        dataset (Dataset): Dataset to be split
        lengths (iterable): lengths of splits to be produced
    """
    if sum(lengths) != len(dataset):
        raise ValueError("Sum of input lengths does not equal the length of the input dataset!")

    indices = randperm(sum(lengths))
    return [Subset(dataset, indices[offset - length:offset]) for offset, length in zip(_accumulate(lengths), lengths)]


# Round x to the nearest multiple of p and x' >= x
def round2nearest_multiple(x, p):
    return ((x - 1) // p + 1) * p


def decodeRG(im):
    return (im[:, :, 0] // 10) * 256 + im[:, :, 1]


def encodeRG(channel):
    result = np.zeros(channel.shape + (3,), dtype=np.uint8)
    result[:, :, 0] = (channel // 256) * 10
    result[:, :, 1] = channel % 256
    return result


def uint16_imresize(seg, shape):
    return decodeRG(imresize(encodeRG(seg), shape, interp="nearest"))


class TrainDataset_os(Dataset):
    def __init__(self, records, opt, max_sample=-1, batch_per_gpu=1):
        self.imgSize = list(opt.imgSizes)
        self.imgMaxSize = opt.imgMaxSize
        self.random_flip = True

        # max down sampling rate of network to avoid rounding during conv or pooling
        self.padding_constant = opt.padding_constant

        # down sampling rate of segm labe
        self.segm_downsampling_rate = opt.segm_downsampling_rate
        self.batch_per_gpu = batch_per_gpu

        # classify images into two classes: 1. h > w and 2. h <= w
        self.batch_record_list = [[], []]

        # override dataset length when trainig with batch_per_gpu > 1
        self.cur_idx = 0

        # mean and std
        #self.normalize = transforms.Compose([transforms.Normalize(mean=[102.9801, 115.9465, 122.7717], std=[1., 1., 1.])])
        self.normalize = transforms.Compose([transforms.Normalize(mean=[0.4038, 0.4546 ,0.4814], std=[1., 1., 1.])])


        self.list_sample = records

        self.if_shuffled = False
        if max_sample > 0:
            self.list_sample = self.list_sample[0:max_sample]
        self.num_sample = len(self.list_sample)
        assert self.num_sample > 0
        print('# samples: {}'.format(self.num_sample))

    def img_transform(self, img):
        # 0-255 to 0-1
        img = np.float32(np.array(img)) / 255.
        img = img.transpose((2, 0, 1))
        img = self.normalize(torch.from_numpy(img.copy()))
        return img

    def _get_sub_batch(self):
        while True:
            # get a sample record
            this_sample = self.list_sample[self.cur_idx]
            if this_sample['height'] > this_sample['width']:
                self.batch_record_list[0].append(this_sample)  # h > w, go to 1st class
            else:
                self.batch_record_list[1].append(this_sample)  # h <= w, go to 2nd class

            # update current sample pointer
            self.cur_idx += 1
            if self.cur_idx >= self.num_sample:
                self.cur_idx = 0
                np.random.shuffle(self.list_sample)

            if len(self.batch_record_list[0]) == self.batch_per_gpu:
                batch_records = self.batch_record_list[0]
                self.batch_record_list[0] = []
                break
            elif len(self.batch_record_list[1]) == self.batch_per_gpu:
                batch_records = self.batch_record_list[1]
                self.batch_record_list[1] = []
                break
        return batch_records
    def segm_transform(self, segm):
        # to tensor, -1 to 149
        segm = torch.from_numpy(np.array(segm)).long() - 1
        return segm
    def __getitem__(self, index):
        # NOTE: random shuffle for the first time. shuffle in __init__ is useless
        if not self.if_shuffled:
            np.random.shuffle(self.list_sample)
            self.if_shuffled = True

        # get sub-batch candidates
        batch_records = self._get_sub_batch()

        # resize all images' short edges to the chosen size
        if isinstance(self.imgSize, list):
            this_short_size = np.random.choice(self.imgSize)
        else:
            this_short_size = self.imgSize
        
        # calculate the BATCH's height and width
        # since we concat more than one samples, the batch's h and w shall be larger than EACH sample
        batch_resized_size = np.zeros((self.batch_per_gpu, 2), np.int32)
        for i in range(self.batch_per_gpu):
            img_height, img_width = batch_records[i]['height'], batch_records[i]['width']
            this_scale = min(this_short_size / min(img_height, img_width), self.imgMaxSize / max(img_height, img_width))
            img_resized_height, img_resized_width = img_height * this_scale, img_width * this_scale
            batch_resized_size[i, :] = img_resized_height, img_resized_width
        batch_resized_height = np.max(batch_resized_size[:, 0])
        batch_resized_width = np.max(batch_resized_size[:, 1])

        # Here we must pad both input image and segmentation map to size h' and w' so that p | h' and p | w'
        batch_resized_height = int(round2nearest_multiple(batch_resized_height, self.padding_constant))
        batch_resized_width = int(round2nearest_multiple(batch_resized_width, self.padding_constant))

        assert self.padding_constant >= self.segm_downsampling_rate, \
            'padding constant must be equal or large than segm downsamping rate'

        batch_images = torch.zeros((self.batch_per_gpu, 3, batch_resized_height, batch_resized_width))
        batch_material = torch.zeros((self.batch_per_gpu, batch_resized_height // self.segm_downsampling_rate,
                                      batch_resized_width // self.segm_downsampling_rate)).long()

        for i in range(self.batch_per_gpu):

            data = os_dataset.resolve_record(batch_records[i])

            img = data['img_data']
            seg_material = data["seg_label"]

            # random flip img obj part material
            if self.random_flip:
                random_flip = np.random.choice([0, 1])
                if random_flip == 1:
                    img = cv2.flip(img, 1)
                    seg_material = cv2.flip(seg_material, 1)

            # img
            img = imresize(img, (batch_resized_size[i, 0], batch_resized_size[i, 1]), interp='bilinear')
            img = img.astype(np.float32)#[:, :, ::-1]  # RGB to BGR!!!
            #img = img.transpose((2, 0, 1))
            img = self.img_transform(torch.from_numpy(img.copy()))
            batch_images[i][:, :img.shape[1], :img.shape[2]] = img

            
            segm = imresize(seg_material,
                            (batch_resized_size[i, 0], batch_resized_size[i, 1]), interp='nearest')
            segm_rounded_height = round2nearest_multiple(segm.shape[0], self.padding_constant)
            segm_rounded_width = round2nearest_multiple(segm.shape[1], self.padding_constant)
            segm_rounded = np.zeros((segm_rounded_height, segm_rounded_width), dtype='uint8')
            segm_rounded[:segm.shape[0], :segm.shape[1]] = segm
            segm = imresize(segm_rounded,
                            (segm_rounded.shape[0] // self.segm_downsampling_rate,
                                segm_rounded.shape[1] // self.segm_downsampling_rate), interp='nearest')
            
            segm = self.segm_transform(segm)
            
            #batch_material[i][:segm.shape[0], :segm.shape[1]] = torch.from_numpy(segm.copy())
            batch_material[i][:segm.shape[0], :segm.shape[1]] = segm

      
        output = dict()
        output['img_data'] = batch_images
        output['seg_label'] = batch_material
        return output

    def __len__(self):
        return int(1e6)  # It's a fake length due to the trick that every loader maintains its own list
        # return self.num_sampleclass


class ValDataset_os(Dataset):
    def __init__(self, records, opt, max_sample=-1, start_idx=-1, end_idx=-1):
        self.imgSize = list(opt.imgSizes)
        self.imgMaxSize = opt.imgMaxSize
        # max down sampling rate of network to avoid rounding during conv or pooling
        self.padding_constant = opt.padding_constant

        # mean and std
        #self.normalize = transforms.Compose([transforms.Normalize(mean=[102.9801, 115.9465, 122.7717], std=[1., 1., 1.])])
        self.normalize = transforms.Compose([transforms.Normalize(mean=[0.4038, 0.4546 ,0.4814], std=[1., 1., 1.])])

        self.list_sample = records

        if max_sample > 0:
            self.list_sample = self.list_sample[0:max_sample]

        if start_idx >= 0 and end_idx >= 0:  # divide file list
            self.list_sample = self.list_sample[start_idx:end_idx]

        self.num_sample = len(self.list_sample)
        assert self.num_sample > 0
        print('# samples: {}'.format(self.num_sample))
    def img_transform(self, img):
        # 0-255 to 0-1
        img = np.float32(np.array(img)) / 255.
        img = img.transpose((2, 0, 1))
        img = self.normalize(torch.from_numpy(img.copy()))
        return img

    def segm_transform(self, segm):
        # to tensor, -1 to 149
        segm = torch.from_numpy(np.array(segm)).long() - 1
        return segm

    def __getitem__(self, index):
        data = os_dataset.resolve_record(self.list_sample[index])
        output = {}

        # image
        img = data['img_data']
        segm = data['seg_label']
        assert(img.shape[0] == segm.shape[0])
        assert(img.shape[1] == segm.shape[1])
        # img = torch.from_numpy(img)
        # img = img.permute(2,0,1)
        
        #img = img[:, :, ::-1]  # BGR to RGB!!!
        ori_height, ori_width, _ = img.shape
        img_resized_list = []
        for this_short_size in self.imgSize:
            # calculate target height and width
            scale = min(this_short_size / float(min(ori_height, ori_width)),
                        self.imgMaxSize / float(max(ori_height, ori_width)))
            target_height, target_width = int(ori_height * scale), int(ori_width * scale)

            # to avoid rounding in network
            target_height = round2nearest_multiple(target_height, self.padding_constant)
            target_width = round2nearest_multiple(target_width, self.padding_constant)

            # resize
            img_resized = cv2.resize(img.copy(), (target_width, target_height))

            # image to float
            img_resized = img_resized.astype(np.float32)
            # img_resized = img_resized.transpose((2, 0, 1))
            # img_resized = torch.from_numpy(img_resized)
            img_resized = self.img_transform(img_resized)
            img_resized = img_resized.reshape(1,img_resized.size(0),img_resized.size(1),img_resized.size(2))
            img_resized_list.append(img_resized)

        segm = self.segm_transform(segm)
        #batch_segms = torch.unsqueeze(segm, 0)
        output = dict()
        output['img_ori'] = np.array(img)
        output['img_data'] = [x.contiguous() for x in img_resized_list]
        #output['seg_label'] = batch_segms.contiguous()
        #output['seg_label'] = torch.from_numpy(segm).contiguous()
        output['seg_label'] = segm.contiguous()


        #output['info'] = "xx/"+str(index)+".jpg"
        output['info'] = data['info']

        return output

    def __len__(self):
        return self.num_sample


class TestDataset(Dataset):
    def __init__(self, odgt, opt, max_sample=-1):
        self.imgSize = opt.imgSize
        self.imgMaxSize = opt.imgMaxSize
        # max down sampling rate of network to avoid rounding during conv or pooling
        self.padding_constant = opt.padding_constant
        # down sampling rate of segm labe
        self.segm_downsampling_rate = opt.segm_downsampling_rate

        # mean and std
        self.img_transform = transforms.Compose([
            transforms.Normalize(mean=[102.9801, 115.9465, 122.7717], std=[1., 1., 1.])
        ])

        if isinstance(odgt, list):
            self.list_sample = odgt
        elif isinstance(odgt, str):
            self.list_sample = [json.loads(x.rstrip()) for x in open(odgt, 'r')]

        if max_sample > 0:
            self.list_sample = self.list_sample[0:max_sample]
        self.num_sample = len(self.list_sample)
        assert self.num_sample > 0
        print('# samples: {}'.format(self.num_sample))

    def __getitem__(self, index):
        this_record = self.list_sample[index]
        # load image and label
        image_path = this_record['fpath_img']
        img = imread(image_path, mode='RGB')
        img = img[:, :, ::-1]  # BGR to RGB!!!

        ori_height, ori_width, _ = img.shape

        img_resized_list = []
        for this_short_size in self.imgSize:
            # calculate target height and width
            scale = min(this_short_size / float(min(ori_height, ori_width)),
                        self.imgMaxSize / float(max(ori_height, ori_width)))
            target_height, target_width = int(ori_height * scale), int(ori_width * scale)

            # to avoid rounding in network
            target_height = round2nearest_multiple(target_height, self.padding_constant)
            target_width = round2nearest_multiple(target_width, self.padding_constant)

            # resize
            img_resized = cv2.resize(img.copy(), (target_width, target_height))

            # image to float
            img_resized = img_resized.astype(np.float32)
            img_resized = img_resized.transpose((2, 0, 1))
            img_resized = self.img_transform(torch.from_numpy(img_resized))

            img_resized = torch.unsqueeze(img_resized, 0)
            img_resized_list.append(img_resized)

        # segm = torch.from_numpy(segm.astype(np.int)).long()

        # batch_segms = torch.unsqueeze(segm, 0)

        # batch_segms = batch_segms - 1 # label from -1 to 149
        output = dict()
        output['img_ori'] = img.copy()
        output['img_data'] = [x.contiguous() for x in img_resized_list]
        # output['seg_label'] = batch_segms.contiguous()
        output['info'] = this_record['fpath_img']
        return output

    def __len__(self):
        return self.num_sample