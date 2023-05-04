import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import cv2
import numpy as np
from PIL import Image
import scipy.io as sio
from scipy import ndimage
from skimage import measure
from skimage import morphology
from scipy import ndimage as ndi
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
import matplotlib.pylab as plt

def FillHole(im_in):
    im_floodfill = im_in.copy()
    # Mask is used for floodFill, the official requirement is length and width +2
    h, w = im_in.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    
    # The pixel corresponding to the seedPoint in the floodFill function must be the background
    isbreak = False
    for i in range(im_floodfill.shape[0]):
        for j in range(im_floodfill.shape[1]):
            if(im_floodfill[i][j]==0):
                seedPoint=(i,j)
                isbreak = True
                break
        if(isbreak):
            break

    cv2.floodFill(im_floodfill, mask, seedPoint, 255)
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    im_out = im_in | im_floodfill_inv
     
    return im_out

def post_old(img):
    img_post = img
    # img_post = cv2.morphologyEx(img, cv2.MORPH_OPEN, np.ones((1,1), np.uint8))
    
    img_post = ndimage.binary_opening(img_post, structure=morphology.disk(2)).astype(np.int)
    img_post = ndimage.binary_closing(img_post, structure=morphology.disk(2)).astype(np.int)

    img_post = ndimage.binary_opening(img_post, structure=morphology.disk(1)).astype(np.int)
    img_post = ndimage.binary_closing(img_post, structure=morphology.disk(1)).astype(np.int)

    # Image.fromarray(255*img_post[:512,:512].astype('uint8')).save('./logs/postfig-morph.png')
    distance = ndimage.morphology.distance_transform_edt(img_post)
    smoothed_distance = ndimage.gaussian_filter(distance, sigma=1)

    local_maxi = peak_local_max(smoothed_distance, indices=False, footprint=morphology.disk(1), labels=img_post) # morphology.disk(10)

    # local_maxi = peak_local_max(smoothed_distance, indices=False, footprint=morphology.disk(7), labels=img_post) # morphology.disk(10)
    markers = ndi.label(local_maxi)[0]

    # saving distance image with peaks
    toplt = np.array(Image.fromarray(255*(smoothed_distance[:512,:512]/np.max(smoothed_distance[:512,:512]))).convert('RGB'))
    buffedmaxi = ndimage.binary_dilation(local_maxi, structure=morphology.disk(2)).astype(np.int)
    toplt[buffedmaxi[:512,:512]==1] = np.array([255,0,0])
    # Image.fromarray(toplt).save('./logs/postfig-dist.png')

    img_post = watershed(-smoothed_distance, markers, mask=img_post)

    return img_post

def post(img):
    img_post = img

    img_post = ndimage.binary_opening(img_post, structure=morphology.disk(1)).astype(np.int)
    img_post = ndimage.binary_closing(img_post, structure=morphology.disk(1)).astype(np.int)

    # Image.fromarray(255*img_post[:512,:512].astype('uint8')).save('./logs/postfig-morph.png')

    distance = ndimage.morphology.distance_transform_edt(img_post)
    smoothed_distance = ndimage.gaussian_filter(distance, sigma=1)

    local_maxi = peak_local_max(smoothed_distance, indices=False, footprint=morphology.disk(1), labels=img_post) # morphology.disk(10)

    # local_maxi = peak_local_max(smoothed_distance, indices=False, footprint=morphology.disk(7), labels=img_post) # morphology.disk(10)
    markers = ndi.label(local_maxi)[0]

    # saving distance image with peaks
    toplt = np.array(Image.fromarray(255*(smoothed_distance[:512,:512]/np.max(smoothed_distance[:512,:512]))).convert('RGB'))
    buffedmaxi = ndimage.binary_dilation(local_maxi, structure=morphology.disk(2)).astype(np.int)
    toplt[buffedmaxi[:512,:512]==1] = np.array([255,0,0])
    # Image.fromarray(toplt).save('./logs/postfig-dist.png')

    img_post = watershed(-smoothed_distance, markers, mask=img_post)

    return img_post

def watershed_post(img):
    distance = ndimage.morphology.distance_transform_edt(img.astype(np.int))
    smoothed_distance = ndimage.gaussian_filter(distance, sigma=1)

    local_maxi = peak_local_max(smoothed_distance, indices=False, footprint=morphology.disk(3), labels=img)
    markers = ndi.label(local_maxi)[0]

    img_post = watershed(-smoothed_distance, markers, mask=img)
    return img_post

def dice(pred, true):
    pred[pred>=0.5] = 1
    pred[pred<0.5] = 0
    inter = np.sum(pred * true)
    uni = np.sum(pred) + np.sum(true)
    if uni == 0:
        if inter == 0:
            return 1
        else:
            return 0
    return inter * 2 / uni

if __name__ == '__main__':

    for run_no in range(10):
        dataset_name = "CPM17"  # MoNuSeg  CoNSeP  CPM17
        train_mode_type = "_normal"  # _finetune_10  _normal 
        model_type = "Master"  # Master  noAtt  noSelf  noSelfnoAtt  noAspp  Unet  DeepLab  SimCLR
        preprocessing_mode = "modistain"  # modistain  normalstain  nopreprocessing
        other_infor = "_nosplit_sparse06"
        post_type = "_post_new"  # _post_new  _post_old

        folder_name = "Exam001_Test1123_" + dataset_name + "_" + model_type + "_" + preprocessing_mode + train_mode_type + other_infor + "_" + str(run_no)

        gt_path = "../Datasets/" + dataset_name + "/Test/Binary/"
        ori_path = "../Datasets/" + dataset_name + "/Test/Original/"
        picture = "./result_finetune/"
        save_path = "./post/" + folder_name + post_type

        if not os.path.isdir(save_path):
            os.mkdir(save_path)
            os.mkdir(save_path + "/0/")
            os.mkdir(save_path + "/1/")
            os.mkdir(save_path + "/2/")
            os.mkdir(save_path + "/3/")

        images = os.listdir(picture + folder_name + "/pred/")
        for i in images:
            pred = Image.open(picture + folder_name + "/pred/" + i).convert("L")
            pred = np.array(pred)

            if post_type == "_post_old": # for supervised
                post_img = post_old(pred)
                post_img2 = watershed_post(pred)
            else: # for unsupervised
                post_img = post(pred)
                post_img[post_img>0] = 1
                pred_fill = FillHole(post_img)
                pred_fill[pred_fill==-1] = 1
                pred_fill[pred_fill<0] = 0
                post_img2 = watershed_post(pred_fill)

            # pred_fill = FillHole(pred)
            # post_img = post(pred_fill)
            # post_img2 = watershed_post(pred_fill)

            sio.savemat(save_path + "/0/" + i.split(".")[0] + ".mat", {"inst_map": post_img})
            sio.savemat(save_path + "/1/" + i.split(".")[0] + ".mat", {"inst_map": post_img2})

            post_img[post_img>0] = 1
            plt.imsave(save_path + "/2/" + i, post_img, cmap="gray")
            post_img2[post_img2>0] = 1
            plt.imsave(save_path + "/3/" + i, post_img2, cmap="gray")

            ori = Image.open(ori_path + i).convert("RGB")
            gt = Image.open(gt_path + i).convert("L")
            gt = np.array(gt) / 255

            print(i + " before Dice: ", dice(pred, gt))
            print(i + " post Dice: ", dice(post_img2, gt))
            print("------------------------------------------")