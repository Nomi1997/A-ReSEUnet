import numpy as np 
import matplotlib.pyplot as plt
import os
from PIL import Image
import scipy.io as sio

# w = patch size
# ori_length = ori size
# length = multiples of stride
# start = start
# end = length - w - start + 1 (How many times the distance)
# stride = overlap

def cropping(image_path, save_path, start=0):
    img_name_list = os.listdir(image_path)
    for img_name in img_name_list:
        img = Image.open(image_path + img_name).convert("RGB")
        img = np.array(img)
        length, w, stride, end = get_size(img)
        count = 0
        for y in range(start, end, stride):
            if w+y > length:
                y = length - stride*2
                img_row = img[y:, 0:length, :]
            else:
                img_row = img[y:w+y, 0:length, :]
            for x in range(start, end, stride):
                if w + x > length:
                    x = length - stride*2
                    img_new = img_row[0:w, x:, :]
                else:
                    img_new = img_row[0:w, x:w+x, :]
                
                plt.imsave(save_path + img_name.split(".")[0] +"_" + str(count) + ".png", img_new)
                count += 1

def cropping_lab(label_path, save_path, start=0):
    img_name_list = os.listdir(label_path)
    for img_name in img_name_list:
        img = Image.open(label_path + img_name).convert("L")
        img = np.array(img)
        length, w, stride, end = get_size(img, mode="histo")
        count = 0
        for y in range(start, end, stride):
            if w+y > length:
                y = length - stride*2
                img_row = img[y:, 0:length]
            else:
                img_row = img[y:w+y, 0:length]
            for x in range(start, end, stride):
                if w+x > length:
                    x = length - stride*2
                    img_new = img_row[0:w, x:]
                else:
                    img_new = img_row[0:w, x:w+x]
                
                img_new[img_new > 0] = 1
                plt.imsave(save_path + img_name.split(".")[0] +"_" + str(count) + ".png", img_new, cmap="gray")
                count += 1

def cropping_type(image_path, save_path, save_lab_path, start=0):
    img_name_list = os.listdir(image_path)
    for img_name in img_name_list:
        img = sio.loadmat(os.path.join(image_path, img_name.split(".")[0] + ".mat"))
        img = (img["type_map"]).astype("int32")
        length, w, stride, end = get_size(img, mode="histo")
        count = 0
        for y in range(start, end, stride):
            if w+y > length:
                y = length - stride*2
                img_row = img[y:, 0:length]
            else:
                img_row = img[y:w+y, 0:length]
            for x in range(start, end, stride):
                if w+x > length:
                    x = length - stride*2
                    img_new = img_row[0:w, x:]
                else:
                    img_new = img_row[0:w, x:w+x]
                
                img_bin = img_new.copy()
                img_bin[img_bin > 0] = 1
                save_file_name = img_name.split(".")[0] +"_" + str(count)

                plt.imsave(save_lab_path + img_name.split(".")[0] +"_" + str(count) + ".png", img_bin, cmap="gray")
                sio.savemat(save_path + save_file_name  + ".mat", {"type_map":img_new})

                # filter
                # sum_lab = np.sum(img_bin)
                # patches_sq = 256*256
                # thr = sum_lab / patches_sq
                # if thr > 0.05: 
                #     plt.imsave(save_path + img_name.split(".")[0] +"_" + str(count) + ".png", img_bin, cmap="gray")
                #     sio.savemat(save_path + save_file_name  + ".mat", {"type_map":img_new})
                
                count += 1

# get size, get end
def get_size(img, mode="histo"):
    if mode == "histo":
        w = 256
        stride = w // 2
    else:
        raise NotImplementedError
        
    ori_length = img.shape[0]
    n = 1
    while stride * n < ori_length: 
        n += 1
    length = stride * n
    # end = length - w - start + 1
    end = length - w - 0 + 1

    return ori_length, w, stride, end
def makeDir(path, dataset, subfolder, crop_type):
    if not os.path.isdir(path):
        if not os.path.isdir("./" + dataset + subfolder):
            os.mkdir("./" + dataset + subfolder)
        if not os.path.isdir("./" + dataset + subfolder + "/" + crop_type):
            os.mkdir("./" + dataset + subfolder + "/" + crop_type)
        os.mkdir(path)

if __name__ == '__main__':
    # w = 256
    # start = 0
    # end = 1024 - w - start + 1
    # stride = 128

    crop_type = "Train"
    dataset = "CoNSeP"
    subfolder = "/patches/"

    # To crop the original image
    image_path = "./" + dataset + "/" + crop_type + "/Images/"
    save_img_path = "./" + dataset + subfolder + crop_type + "/Images/"
    makeDir(save_img_path, dataset, subfolder, crop_type) 
    cropping(image_path, save_img_path)
    
    if dataset == "CoNSeP":
        type_path = "./" + dataset + "/" + crop_type + "/Labels/"
        save_type_path = "./" + dataset + subfolder + crop_type + "/Labels/"
        save_lab_path = "./" + dataset + subfolder + crop_type + "/Binary/"
        
        makeDir(save_lab_path, dataset, subfolder, crop_type)
        makeDir(save_type_path, dataset, subfolder, crop_type)
        cropping_type(type_path, save_type_path, save_lab_path)
    else:
        label_path = "./" + dataset + "/" + crop_type + "/Binary/" 
        save_lab_path = "./" + dataset + subfolder + crop_type + "/Binary/"
        
        makeDir(save_lab_path, dataset, subfolder, crop_type)    
        cropping_lab(label_path, save_lab_path)