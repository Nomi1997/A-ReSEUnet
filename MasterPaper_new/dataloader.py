import numpy as np
import scipy.io as sio
import torch.utils.data
from PIL import Image
from normalizeStaining import normalizeStaining, GaussianBlur
from torchvision.transforms import transforms
import torchvision.transforms.functional as F
import time

# import cupy as cp
# cp.cuda.Device(0).use()
# from cupy._core.dlpack import toDlpack, fromDlpack
# from torch.utils.dlpack import to_dlpack, from_dlpack
import torch

####
class FileLoader(torch.utils.data.Dataset):
    """Data Loader. Loads images from a file list and 
    performs augmentation with the albumentation library.
    After augmentation, horizontal and vertical maps are 
    generated.

    Args:
        file_list: list of filenames to load
        input_shape: shape of the input [h,w] - defined in config.py
        mask_shape: shape of the output [h,w] - defined in config.py
        mode: 'train' or 'valid'
    """
    def __init__(self, file_list, label_file_list, with_type=False, input_shape=None, mask_shape=None, preprocessing_mode="modistain", mode="train", model_type="Master"):
        assert input_shape is not None and mask_shape is not None
        self.mode = mode
        self.preprocessing_mode = preprocessing_mode
        self.model_type = model_type
        self.info_list = file_list
        self.label_list = label_file_list
        self.with_type = with_type
        self.mask_shape = mask_shape
        self.input_shape = input_shape
        self.total_time_train = 0
        self.total_time_val = 0
        color_jitter = transforms.ColorJitter(0.5 * 1, 0.5 * 1, 0.5 * 1, 0.2 * 1)
        self.data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=input_shape[1]),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.RandomApply([color_jitter], p=1),
                                              transforms.RandomGrayscale(p=0.2),
                                              GaussianBlur(kernel_size=int(0.1 * input_shape[1])),
                                              ])
        return

    def __len__(self):
        return len(self.info_list)

    def __getitem__(self, idx):
        start = time.time()#Timesmape

        img_path = self.info_list[idx]
        lab_path = self.label_list[idx]
        img = Image.open(img_path).convert("RGB")
        ann = Image.open(lab_path).convert("L")

        if self.preprocessing_mode == "nopreprocessing": 
            img_stain = 1 - np.array(img)/255.0

        elif self.preprocessing_mode == "normalstain":  
            # _, img_stain, _ = normalizeStaining(np.array(img), Io=240, alpha=1, beta=0.15)
            # img_stain = 1 - np.array(img_stain) / 255
            _, img_stain, _ = normalizeStaining(img, Io=240, alpha=1, beta=0.15)
            img_stain = 1 - img_stain / 255

        elif self.preprocessing_mode == "zscore":  
            img_stain = 1 - ((np.array(img) - np.mean(img))/np.std(img))

        elif self.preprocessing_mode == "modistain":  
            img = np.array(img)
            img[((img[...,0]) < img[...,1]) & (img[...,1] < img[...,2])] = 255
            img = Image.fromarray(img)

            brightness_factor = 1.1
            contrast_factor = 1.1
            staturation_factor = 1.05
            hue_factor = -0.105
          
            img2 = F.adjust_contrast(img, contrast_factor)
            img2 = F.adjust_hue(img2, hue_factor)
            img2 = F.adjust_saturation(img2, staturation_factor)
            img2 = F.adjust_brightness(img2, brightness_factor)

            # _, img_stain, _ = normalizeStaining(np.array(img2), Io=240, alpha=1, beta=0.15)
            _, img_stain, _ = normalizeStaining(img2, Io=240, alpha=1, beta=0.15)
            img_stain = 1 - img_stain / 255
        else:
            print("Error preprocessing!!")
        
        feed_dict = {"img": img_stain}
        # feed_dict = {"img": cp.asnumpy(img_stain)}
        # feed_dict = {"img": from_dlpack(toDlpack(img_stain))}
        
        if self.mode == "train" and (self.model_type == "Master" or self.model_type == "noAtt"):
            feed_dict["self_1"] = np.array(self.data_transforms(img)) / 255
            feed_dict["self_2"] = np.array(self.data_transforms(img)) / 255
        
        # feed_dict["np_map"] = np.array(ann)[:,:,np.newaxis] / 255
        ann = np.array(ann)[:,:,np.newaxis]
        ann[ann > 0] = 1
        
        feed_dict["np_map"] = ann

        end = time.time()  #Timesmape
        if self.mode == "train":
            self.total_time_train += (end-start)
            feed_dict["time"] = self.total_time_train
            # print("preprocessing total train time= ", self.total_time_train)
        else:
            self.total_time_val += (end-start)
            feed_dict["time"] = self.total_time_val
            # print("preprocessing total valid time= ", self.total_time_val)

        return feed_dict
