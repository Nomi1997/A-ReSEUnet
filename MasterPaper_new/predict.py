import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import time
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from modeling.deeplab import *
from simclr_model import *
from loss import *
from dataloader import *
from normalizeStaining import normalizeStaining

# If you use cupy, you need to set it in the gpu environment, and change part of np to cp
# import cupy as cp
# cp.cuda.Device(0).use()
# from cupy._core.dlpack import toDlpack, fromDlpack
# from torch.utils.dlpack import to_dlpack, from_dlpack

def norm(img, preprocessing_mode):
    if preprocessing_mode == "nopreprocessing": # no adjustment at all
            img_stain = 1 - np.array(img)/255.0
            img_stain = np.asnumpy(img_stain)

    elif preprocessing_mode == "normalstain":  # normal stain normalize
        _, img_stain, _ = normalizeStaining(img, Io=240, alpha=1, beta=0.15)
        img_stain = 1 - np.array(img_stain) / 255
        img_stain = np.asnumpy(img_stain)

    elif preprocessing_mode == "zscore":  # Z score
        img_stain = 1 - ((np.array(img) - np.mean(img))/np.std(img))

    elif preprocessing_mode == "modistain":   # adjusted stain normalize
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

        _, img_stain, _ = normalizeStaining(np.array(img2), Io=240, alpha=1, beta=0.15)
        img_stain = 1 - img_stain / 255
        img_stain = np.asnumpy(img_stain)
    else:
        print("Error preprocessing!!")

    return img_stain

def expand(test):
    test = test.transpose((2,0,1))
    test = np.expand_dims(test,axis=0)
    test = torch.from_numpy(test)
    return test

for run_no in range(10):
    start = time.time()

    dataset_name = "CPM17"  # MoNuSeg  CPM17  CoNSeP
    train_mode_type = "_normal"  # "_finetune_10  _normal 
    model_type = "SimCLR"  # Master  noAtt  noSelf  noSelfnoAtt  Unet  DeepLab  SimCLR
    preprocessing_mode = "modistain"  # modistain  normalstain  nopreprocessing
    other_infor = "_nosplit_sparse06"
    log_path = "" + "/" # The first double quotes fill in the folder name (if there are additional additions)
    save_path = "./result/"

    weight_name = "Exam001_Test1120_" + dataset_name + "_" + model_type + "_" + preprocessing_mode + train_mode_type + other_infor

    path = "../Datasets/" + dataset_name + "/Test/Original/"
    file_list =  os.listdir(path)
    
    patch_size = 256
    sparse_radio = 0.5
    nr_types = 1
    
    for file_name in file_list:
        ori = Image.open(path + file_name).convert("RGB")
        img = np.array(ori)
        width, height = img.shape[:2]

        # Get the length and width of padding required
        x = patch_size - width % patch_size
        y = patch_size - height % patch_size

        # Cropping after padding
        img = norm(ori, preprocessing_mode)
        img = np.pad(img, ((0, y),(0, x),(0, 0)), "symmetric")
        length = img.shape[0]
        img_total = []
        for y in range(0, length, patch_size):
            img_row = img[y : patch_size + y, 0 : length, :]
            img_batch = []
            for x in range(0, length, patch_size):
                img_new = img_row[0 : patch_size, x : patch_size + x, :]
                img_new = expand(img_new)
                
                if model_type == "Master":
                    model = Master_all(sparse_radio=sparse_radio, backbone='resnet50', output_stride=16, num_classes=nr_types)

                elif model_type == "noAtt":
                    model = Master_noAtt(backbone='resnet50', output_stride=16, num_classes=nr_types)

                elif model_type == "noSelf":
                    model = Master_noSelf(sparse_radio=sparse_radio, output_stride=16, num_classes=nr_types)

                elif model_type == "noAspp":
                    model = Master_noAspp(sparse_radio=sparse_radio, output_stride=16, num_classes=nr_types)

                elif model_type == "noSelfnoAtt":
                    model = Master_noSelfnoAtt(output_stride=16, num_classes=nr_types)

                elif model_type == "Unet":
                    model = Unet(3, nr_types)

                elif model_type == "DeepLab":
                    model = DeepLab(num_classes=nr_types)

                elif model_type == "SimCLR":
                    model = SimCLR_addDecoder(freeze=True, num_classes=nr_types)

                checkpoint = torch.load("./logs/" + log_path + weight_name + "/checkpoint_10.tar")
                model.load_state_dict(checkpoint['net'])
                
                # Supervised SimCLR_histo load pretrained
                # model.load_state_dict(torch.load("./logs/check_time/" + weight_name + "/checkpoint_10.tar"))
                model.eval()

                if model_type == "Master" or model_type == "noSelf" or model_type == "noAspp":
                    output, att_output, att_output2 = model(img_new.float())
                    output = output.detach().numpy()
                else:
                    output = model(img_new.float())
                    output = output.detach().numpy()
                
                fin = output[0,0,:,:]        
                new_fin = fin
                new_fin[new_fin >= 0.5] = 1
                new_fin[new_fin < 0.5] = 0

                img_batch.append(new_fin)
            img_total.append(img_batch)

        # restore to original size
        img_re = []
        for row in img_total:
            img_re.append(np.hstack(row))

        img_test = np.vstack(img_re)
        img_test = img_test[:ori.size[0],:ori.size[1]]

        if not os.path.isdir(save_path + weight_name):
            os.mkdir(save_path + weight_name)
            os.mkdir(save_path + weight_name + "/pred")
        plt.imsave(save_path + weight_name + "/pred/" + file_name, img_test, cmap="gray")
        print("Run no. " + str(run_no) + ": " + file_name + " is Finish!")
    end = time.time()
    print("Run no. " + str(run_no) + ": Predict Time = ", end-start)
    print("Record Informataion: " + save_path + log_path + weight_name)