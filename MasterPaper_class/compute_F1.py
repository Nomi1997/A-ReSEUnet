import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import glob
import numpy as np
import scipy.io as sio
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score

def trans_type(image):
    image[image>4]=4
    return image

def draw_type(pred):
    pred = trans_type(pred)
    pred[pred>=0.5]=1
    pred[pred<0.5]=0
    
    pred_1 = pred[:,:,1]
    pred_2 = pred[:,:,2]
    pred_3 = pred[:,:,3]
    pred_4 = pred[:,:,4]

    return pred_1, pred_2, pred_3, pred_4

dataset_name = "CoNSeP"
train_mode_type = "_finetune_10"  # _finetune_100 _normal
model_type = "noSelfnoAtt"  # Master  noAtt  noSelf  noSelfnoAtt
preprocessing_mode = "modistain"  # modistain  normalstain nopreprocessing
other_infor = "_epoch10"

for run_no in range(1):
    folder_name = "Exam002_Test1213_" + dataset_name + "_" + model_type + "_" + preprocessing_mode + train_mode_type + other_infor + "_" + str(run_no)
    
    pred_dir = "./result_ablation/" + folder_name + "/"
    true_dir = "../Datasets/" + dataset_name + "/Test/Labels_mat/"
    if not os.path.isdir("./post/" + folder_name):
        os.mkdir("./post/" + folder_name)

    with open("./post/" + folder_name + "/" + "metrics.txt", 'a', encoding="utf-8", newline='') as outFile:
        print("mat_name   F_all   F_inflammatory   F_epithelial   F_spindle" , file=outFile)

    mat_list = os.listdir(true_dir)
    for mat_name in mat_list:
        instance = sio.loadmat(true_dir + mat_name)["inst_map"].astype("int32")
        types = sio.loadmat(true_dir + mat_name)["type_map"].astype("int32")
        pred = sio.loadmat(pred_dir + mat_name)["type_map"]
        
        types[(types == 3) | (types == 4)] = 3
        types[(types == 5) | (types == 6) | (types == 7)] = 4

        pred_1, pred_2, pred_3, pred_4 = draw_type(pred)
        
        pred_tmp = pred_1*1 + pred_2*2 + pred_3*3 + pred_4*4
        pred_save = pred_tmp.copy()

        types_inflammatory = types == 2
        types_epithelial = types == 3
        types_spindle = types == 4
        
        F_inflammatory = f1_score(types_inflammatory.flatten(), pred_2.flatten())
        F_epithelial = f1_score(types_epithelial.flatten(), pred_3.flatten())
        F_spindle = f1_score(types_spindle.flatten(), pred_4.flatten())
        
        pred_tmp[pred_tmp>0]=1
        types[types>0]=1
        F_all = f1_score(types.flatten(), pred_tmp.flatten())

        print(mat_name, F_all, F_inflammatory, F_epithelial, F_spindle)
        
        plt.imsave("./post/" + folder_name + "/" + mat_name.split(".")[0] + ".png", pred_save)
        with open("./post/" + folder_name + "/" + "metrics.txt", 'a', encoding="utf-8", newline='') as outFile:
            print(mat_name, F_all, F_inflammatory, F_epithelial, F_spindle , file=outFile)
    print("--------------------------------------------")