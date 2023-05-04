import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import time
import glob
import numpy as np
import argparse

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torchsummary import summary

from utils import yaml_config_hook, load_optimizer, save_model, load_model_weights
from dataloader import *
from simclr_model import *
from modeling.deeplab import *
from loss import *

def basic(args, model):
    # First train
    print("Start training!")
    run(args, model)
    print("Training Finish!")

def run(args, model):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train_loader = get_datagen(args, run_mode="train")
    valid_loader = get_datagen(args, run_mode="valid")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model.to(args.device)

    if args.is_fine_tune:
        criterion = DiceBCELoss()
        criterion_val = DiceBCELoss()
    else:
        criterion = NT_Xent(args.batch_size, args.temperature, 1)
        criterion_val = DiceBCELoss()
    
    optimizer, scheduler = load_optimizer(args, model=model)

    log_str = "Start time of training : " + str(time.ctime())
    print(log_str)
    with open(args.log_path  + "/" + "log.txt", 'a', encoding="utf-8", newline='') as outFile:
        print(log_str, file=outFile)

    for epoch in range(args.epochs):
        # type
        if args.model_type == "Master" or args.model_type == "noAspp":
            model, train_loss, train_dice = train_step_Master(args, epoch+1, train_loader, model, criterion, optimizer, scheduler)
            log_str = "Epoch [{}/{}] - Train Loss = {}, Train Dice = {}\n".format(epoch+1, args.epochs, train_loss, train_dice)
            log_str += "****************************************************\n"

            model, valid_loss, valid_dice = valid_step_Master(args, epoch+1, valid_loader, model, criterion_val)
            log_str += "Epoch [{}/{}] - Valid Loss = {}, Valid Dice = {}\n".format(epoch+1, args.epochs, valid_loss, valid_dice)
            log_str += "====================================================\n"

        elif args.model_type == "noAtt":
            model, train_loss, train_dice = train_step_noAtt(args, epoch+1, train_loader, model, criterion, optimizer, scheduler)
            log_str = "Epoch [{}/{}] - Train Loss = {}, Train Dice = {}\n".format(epoch+1, args.epochs, train_loss, train_dice)
            log_str += "****************************************************\n"

            model, valid_loss, valid_dice = valid_step_noAtt(args, epoch+1, valid_loader, model, criterion_val)
            log_str += "Epoch [{}/{}] - Valid Loss = {}, Valid Dice = {}\n".format(epoch+1, args.epochs, valid_loss, valid_dice)
            log_str += "====================================================\n"

        elif args.model_type == "noSelf":
            model, train_loss, train_dice = train_step_noSelf(args, epoch+1, train_loader, model, criterion, optimizer, scheduler)
            log_str = "Epoch [{}/{}] - Train Loss = {}, Train Dice = {}\n".format(epoch+1, args.epochs, train_loss, train_dice)
            log_str += "****************************************************\n"

            model, valid_loss, valid_dice = valid_step_noSelf(args, epoch+1, valid_loader, model, criterion_val)
            log_str += "Epoch [{}/{}] - Valid Loss = {}, Valid Dice = {}\n".format(epoch+1, args.epochs, valid_loss, valid_dice)
            log_str += "====================================================\n"

        elif args.model_type == "noSelfnoAtt" or args.model_type == "Unet" or args.model_type == "DeepLab" or args.model_type == "SimCLR":
            model, train_loss, train_dice = train_step_noSelfnoAtt(args, epoch+1, train_loader, model, criterion, optimizer, scheduler)
            log_str = "Epoch [{}/{}] - Train Loss = {}, Train Dice = {}\n".format(epoch+1, args.epochs, train_loss, train_dice)
            log_str += "****************************************************\n"

            model, valid_loss, valid_dice = valid_step_noSelfnoAtt(args, epoch+1, valid_loader, model, criterion_val)
            log_str += "Epoch [{}/{}] - Valid Loss = {}, Valid Dice = {}\n".format(epoch+1, args.epochs, valid_loss, valid_dice)
            log_str += "====================================================\n"

        
        with open(args.log_path  + "/" + "log.txt", 'a', encoding="utf-8", newline='') as outFile:
            print(log_str, file=outFile)

        if epoch % args.save_per_step == 0:
            scheduler.step()
        
        save_model(args, model, optimizer, args.epochs)
        print("Save weight successfully!")
    
    log_str = "End time of training : " + str(time.ctime())
    print(log_str)
    with open(args.log_path  + "/" + "log.txt", 'a', encoding="utf-8", newline='') as outFile:
        print(log_str, file=outFile)

# all
def train_step_Master(args, now_epoch, train_loader, model, criterion, optimizer, scheduler):
    model.train()
    scaler = GradScaler(enabled=False)
    loss_epoch, dice_epoch = 0, 0
    self_criterion = NT_Xent(args.batch_size, args.temperature, 1)
    for step, (batch_data) in enumerate(train_loader):
        ori_img = batch_data["img"].permute(0, 3, 1, 2).contiguous()
        aug_imgs_1 = batch_data["self_1"].permute(0, 3, 1, 2).contiguous()
        aug_imgs_2 = batch_data["self_2"].permute(0, 3, 1, 2).contiguous()
        true_tp = batch_data["tp_map"]
        true_np = batch_data["np_map"]
        input_imgs = torch.cat((ori_img, aug_imgs_1, aug_imgs_2), dim=0)
        with autocast(enabled=False):
            optimizer.zero_grad()
            input_imgs = input_imgs.to(args.device).type(torch.float32)
            image_output, self_output, att_output, att_output2 = model(input_imgs)

            image_output = image_output.permute(0, 2, 3, 1).contiguous()
            att_output = att_output.permute(0, 2, 3, 1).contiguous()
            att_output2 = att_output2.permute(0, 2, 3, 1).contiguous()

            att_output2[att_output2>= 0.5] = 1
            att_output2[att_output2< 0.5] = 0

            att_Red = att_output2[...,0]
            att_Green = att_output2[...,1]
            att_Blue = att_output2[...,2]

            mask_tmp = att_Red + att_Green + att_Blue
            mask_tmp = mask_tmp.unsqueeze(-1)
            A_Gray = 0.299 * att_Red + 0.587 * att_Green + 0.114 * att_Blue
            new_image_output = image_output[...,0] + image_output[...,1]+ image_output[...,2]+ image_output[...,3]+ image_output[...,4]
            new_image_output[new_image_output>1]=1

            if args.is_fine_tune:
                loss_att = criterion(att_output.float().to(args.device), true_np.float().to(args.device))
                loss_img = criterion(new_image_output.float().to(args.device), A_Gray.float().to(args.device))
                loss_0 = criterion(image_output[...,0].float().to(args.device), true_tp[...,0].float().to(args.device))
                loss_1 = criterion(image_output[...,1].float().to(args.device), true_tp[...,1].float().to(args.device))
                loss_2 = criterion(image_output[...,2].float().to(args.device), true_tp[...,2].float().to(args.device))
                loss_3 = criterion(image_output[...,3].float().to(args.device), true_tp[...,3].float().to(args.device))
                loss_4 = criterion(image_output[...,4].float().to(args.device), true_tp[...,4].float().to(args.device))
                loss = loss_att + loss_0 + loss_1 + loss_2 + loss_3 + loss_4 + loss_img

            # image_output = F.sigmoid(image_output)
            mask_tmp[mask_tmp<3] == 0
            mask_tmp[mask_tmp==3] == 1

            AttDice = dice(mask_tmp.cpu().detach().numpy(), true_np.cpu().detach().numpy())
            predDice0 = dice(image_output[...,0].cpu().detach().numpy(), true_tp[...,0].cpu().detach().numpy())
            predDice1 = dice(image_output[...,1].cpu().detach().numpy(), true_tp[...,1].cpu().detach().numpy())
            predDice2 = dice(image_output[...,2].cpu().detach().numpy(), true_tp[...,2].cpu().detach().numpy())
            predDice3 = dice(image_output[...,3].cpu().detach().numpy(), true_tp[...,3].cpu().detach().numpy())
            predDice4 = dice(image_output[...,4].cpu().detach().numpy(), true_tp[...,4].cpu().detach().numpy())
            
            print("Epoch " + str(now_epoch) + " : Att Dice = ", AttDice)
            print("Epoch " + str(now_epoch) + " : pred_0 Dice = ", predDice0)
            print("Epoch " + str(now_epoch) + " : pred_1 Dice = ", predDice1)
            print("Epoch " + str(now_epoch) + " : pred_2 Dice = ", predDice2)
            print("Epoch " + str(now_epoch) + " : pred_3 Dice = ", predDice3)
            print("Epoch " + str(now_epoch) + " : pred_4 Dice = ", predDice4)

            train_dice = (predDice0 + predDice1 + predDice2 + predDice3 + predDice4) / 5.0
            print("--------------------------------------------------")

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if step % args.save_per_step == 0:
            log_str = "Epoch [{}/{}], Step {}: LR={:.7f}, Loss={:.4f}, Dice={:.5f}".format(now_epoch, args.epochs, step, scheduler.get_last_lr()[0], loss.item(), train_dice)
            print(log_str)
            with open(args.log_path   + "/" + "log.txt", 'a', encoding="utf-8", newline='') as outFile:
                print(log_str +'\n', file=outFile)
     
        loss_epoch += loss.item()
        dice_epoch += train_dice

    print("****************************************************\n")
    return model, loss_epoch // (step + 1), dice_epoch // (step + 1)

def valid_step_Master(args, now_epoch, valid_loader, model, criterion):
    model.eval()
    loss_epoch, dice_epoch = 0, 0
    for step, (batch_data) in enumerate(valid_loader):
        ori_img = batch_data["img"].permute(0, 3, 1, 2).contiguous()
        true_np = batch_data["tp_map"]
        with torch.no_grad(): 
            ori_img = ori_img.to(args.device).type(torch.float32)
            true_np = true_np.to(args.device)

            image_output, att_output, att_output2 = model(ori_img)
            image_output = image_output.permute(0, 2, 3, 1).contiguous()
            att_output = att_output.permute(0, 2, 3, 1).contiguous()
            att_output2 = att_output2.permute(0, 2, 3, 1).contiguous()

            loss = criterion(image_output.float().to(args.device), true_np.float().to(args.device))
            
            att_output2[att_output2>= 0.5] = 1
            att_output2[att_output2< 0.5] = 0

            att_Red = att_output2[...,0]
            att_Green = att_output2[...,1]
            att_Blue = att_output2[...,2]

            mask_tmp = att_Red + att_Green + att_Blue
            mask_tmp = mask_tmp.unsqueeze(-1)
            mask_tmp[mask_tmp<3] == 0
            mask_tmp[mask_tmp==3] == 1

            predDice0 = dice(image_output[...,0].cpu().detach().numpy(), true_np[...,0].cpu().detach().numpy())
            predDice1 = dice(image_output[...,1].cpu().detach().numpy(), true_np[...,1].cpu().detach().numpy())
            predDice2 = dice(image_output[...,2].cpu().detach().numpy(), true_np[...,2].cpu().detach().numpy())
            predDice3 = dice(image_output[...,3].cpu().detach().numpy(), true_np[...,3].cpu().detach().numpy())
            predDice4 = dice(image_output[...,4].cpu().detach().numpy(), true_np[...,4].cpu().detach().numpy())
            
            print("Epoch " + str(now_epoch) + " : pred0 Dice = ", predDice0)
            print("Epoch " + str(now_epoch) + " : pred1 Dice = ", predDice1)
            print("Epoch " + str(now_epoch) + " : pred2 Dice = ", predDice2)
            print("Epoch " + str(now_epoch) + " : pred3 Dice = ", predDice3)
            print("Epoch " + str(now_epoch) + " : pred4 Dice = ", predDice4)

            valid_dice = (predDice0 + predDice1 + predDice2 + predDice3 + predDice4) / 5.0
            print("--------------------------------------------------")

            if step % (args.save_per_step // 2) == 0:
                log_str = "Epoch [{}/{}], Step {}: Loss={:.4f} Dice={:.5f}".format(now_epoch, args.epochs, step, loss.item(), valid_dice)
                print(log_str)
                with open(args.log_path   + "/" + "log.txt", 'a', encoding="utf-8", newline='') as outFile:
                    print(log_str +'\n', file=outFile)
            
            loss_epoch += loss.item()
            dice_epoch += valid_dice
    
    print("====================================================\n")
    return model, loss_epoch // (step + 1), dice_epoch // (step + 1)


# noSelf
def train_step_noSelf(args, now_epoch, train_loader, model, criterion, optimizer, scheduler):
    model.train()
    scaler = GradScaler(enabled=False)
    loss_epoch, dice_epoch = 0, 0
    for step, (batch_data) in enumerate(train_loader):
        input_imgs = batch_data["img"].permute(0, 3, 1, 2).contiguous()
        true_tp = batch_data["tp_map"]
        true_np = batch_data["np_map"]
        with autocast(enabled=False):
            optimizer.zero_grad()
            input_imgs = input_imgs.to(args.device).type(torch.float32)
            image_output, att_output, att_output2 = model(input_imgs)

            image_output = image_output.permute(0, 2, 3, 1).contiguous()
            att_output = att_output.permute(0, 2, 3, 1).contiguous()
            att_output2 = att_output2.permute(0, 2, 3, 1).contiguous()

            att_output2[att_output2>= 0.5] = 1
            att_output2[att_output2< 0.5] = 0

            att_Red = att_output2[...,0]
            att_Green = att_output2[...,1]
            att_Blue = att_output2[...,2]

            mask_tmp = att_Red + att_Green + att_Blue
            mask_tmp = mask_tmp.unsqueeze(-1)
            mask_tmp[mask_tmp<3] == 0
            mask_tmp[mask_tmp==3] == 1
            
            if args.is_fine_tune:
                loss_att = criterion(att_output.float().to(args.device), true_np.float().to(args.device))
                loss_0 = criterion(image_output[...,0].float().to(args.device), true_tp[...,0].float().to(args.device))
                loss_1 = criterion(image_output[...,1].float().to(args.device), true_tp[...,1].float().to(args.device))
                loss_2 = criterion(image_output[...,2].float().to(args.device), true_tp[...,2].float().to(args.device))
                loss_3 = criterion(image_output[...,3].float().to(args.device), true_tp[...,3].float().to(args.device))
                loss_4 = criterion(image_output[...,4].float().to(args.device), true_tp[...,4].float().to(args.device))
                loss = loss_att + loss_0 + loss_1 + loss_2 + loss_3 + loss_4

            # image_output = F.sigmoid(image_output)

            AttDice = dice(mask_tmp.cpu().detach().numpy(), true_np.cpu().detach().numpy())
            predDice0 = dice(image_output[...,0].cpu().detach().numpy(), true_tp[...,0].cpu().detach().numpy())
            predDice1 = dice(image_output[...,1].cpu().detach().numpy(), true_tp[...,1].cpu().detach().numpy())
            predDice2 = dice(image_output[...,2].cpu().detach().numpy(), true_tp[...,2].cpu().detach().numpy())
            predDice3 = dice(image_output[...,3].cpu().detach().numpy(), true_tp[...,3].cpu().detach().numpy())
            predDice4 = dice(image_output[...,4].cpu().detach().numpy(), true_tp[...,4].cpu().detach().numpy())
            
            print("Epoch " + str(now_epoch) + " : Att Dice = ", AttDice)
            print("Epoch " + str(now_epoch) + " : pred_0 Dice = ", predDice0)
            print("Epoch " + str(now_epoch) + " : pred_1 Dice = ", predDice1)
            print("Epoch " + str(now_epoch) + " : pred_2 Dice = ", predDice2)
            print("Epoch " + str(now_epoch) + " : pred_3 Dice = ", predDice3)
            print("Epoch " + str(now_epoch) + " : pred_4 Dice = ", predDice4)

            train_dice = (predDice0 + predDice1 + predDice2 + predDice3 + predDice4) / 5.0
            print("--------------------------------------------------")

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if step % args.save_per_step == 0:
            log_str = "Epoch [{}/{}], Step {}: LR={:.7f}, Loss={:.4f}, Dice={:.5f}".format(now_epoch, args.epochs, step, scheduler.get_last_lr()[0], loss.item(), train_dice)
            print(log_str)
            with open(args.log_path   + "/" + "log.txt", 'a', encoding="utf-8", newline='') as outFile:
                print(log_str +'\n', file=outFile)
     
        loss_epoch += loss.item()
        dice_epoch += train_dice

    print("****************************************************\n")
    return model, loss_epoch // (step + 1), dice_epoch // (step + 1)

def valid_step_noSelf(args, now_epoch, valid_loader, model, criterion):
    model.eval()
    loss_epoch, dice_epoch = 0, 0
    for step, (batch_data) in enumerate(valid_loader):
        ori_img = batch_data["img"].permute(0, 3, 1, 2).contiguous()
        true_tp = batch_data["tp_map"]
        with torch.no_grad(): 
            ori_img = ori_img.to(args.device).type(torch.float32)
            true_tp = true_tp.to(args.device)

            image_output, att_output, att_output2 = model(ori_img)
            image_output = image_output.permute(0, 2, 3, 1).contiguous()
            att_output = att_output.permute(0, 2, 3, 1).contiguous()
            att_output2 = att_output2.permute(0, 2, 3, 1).contiguous()

            loss = criterion(image_output.float().to(args.device), true_tp.float().to(args.device))
            
            att_output2[att_output2>= 0.5] = 1
            att_output2[att_output2< 0.5] = 0

            att_Red = att_output2[...,0]
            att_Green = att_output2[...,1]
            att_Blue = att_output2[...,2]

            mask_tmp = att_Red + att_Green + att_Blue
            mask_tmp = mask_tmp.unsqueeze(-1)
            mask_tmp[mask_tmp<3] == 0
            mask_tmp[mask_tmp==3] == 1

            predDice0 = dice(image_output[...,0].cpu().detach().numpy(), true_tp[...,0].cpu().detach().numpy())
            predDice1 = dice(image_output[...,1].cpu().detach().numpy(), true_tp[...,1].cpu().detach().numpy())
            predDice2 = dice(image_output[...,2].cpu().detach().numpy(), true_tp[...,2].cpu().detach().numpy())
            predDice3 = dice(image_output[...,3].cpu().detach().numpy(), true_tp[...,3].cpu().detach().numpy())
            predDice4 = dice(image_output[...,4].cpu().detach().numpy(), true_tp[...,4].cpu().detach().numpy())
            
            print("pred0 Dice = ", predDice0)
            print("pred1 Dice = ", predDice1)
            print("pred2 Dice = ", predDice2)
            print("pred3 Dice = ", predDice3)
            print("pred4 Dice = ", predDice4)

            valid_dice = (predDice0 + predDice1 + predDice2 + predDice3 + predDice4) / 5.0
            print("--------------------------------------------------")

            if step % (args.save_per_step // 2) == 0:
                log_str = "Epoch [{}/{}], Step {}: Loss={:.4f} Dice={:.5f}".format(now_epoch, args.epochs, step, loss.item(), valid_dice)
                print(log_str)
                with open(args.log_path   + "/" + "log.txt", 'a', encoding="utf-8", newline='') as outFile:
                    print(log_str +'\n', file=outFile)
            
            loss_epoch += loss.item()
            dice_epoch += valid_dice
    
    print("====================================================\n")
    return model, loss_epoch // (step + 1), dice_epoch // (step + 1)


# noAtt
def train_step_noAtt(args, now_epoch, train_loader, model, criterion, optimizer, scheduler):
    model.train()
    scaler = GradScaler(enabled=False)
    loss_epoch, dice_epoch = 0, 0
    for step, (batch_data) in enumerate(train_loader):
        ori_img = batch_data["img"].permute(0, 3, 1, 2).contiguous()
        aug_imgs_1 = batch_data["self_1"].permute(0, 3, 1, 2).contiguous()
        aug_imgs_2 = batch_data["self_2"].permute(0, 3, 1, 2).contiguous()
        true_tp = batch_data["tp_map"]
        input_imgs = torch.cat((ori_img, aug_imgs_1, aug_imgs_2), dim=0)
        with autocast(enabled=False):
            optimizer.zero_grad()
            input_imgs = input_imgs.to(args.device).type(torch.float32)
            image_output, self_output = model(input_imgs)

            image_output = image_output.permute(0, 2, 3, 1).contiguous()

            Red = batch_data["img"][...,0]
            Green = batch_data["img"][...,1]
            Blue = batch_data["img"][...,2]
            
            Gray = 0.299 * Red + 0.587 * Green + 0.114 * Blue
            Gray = Gray.unsqueeze(-1)
            
            if args.is_fine_tune:
                loss_0 = criterion(image_output[...,0].float().to(args.device), true_tp[...,0].float().to(args.device))
                loss_1 = criterion(image_output[...,1].float().to(args.device), true_tp[...,1].float().to(args.device))
                loss_2 = criterion(image_output[...,2].float().to(args.device), true_tp[...,2].float().to(args.device))
                loss_3 = criterion(image_output[...,3].float().to(args.device), true_tp[...,3].float().to(args.device))
                loss_4 = criterion(image_output[...,4].float().to(args.device), true_tp[...,4].float().to(args.device))
                loss = loss_0 + loss_1 + loss_2 + loss_3 + loss_4

            predDice0 = dice(image_output[...,0].cpu().detach().numpy(), true_tp[...,0].cpu().detach().numpy())
            predDice1 = dice(image_output[...,1].cpu().detach().numpy(), true_tp[...,1].cpu().detach().numpy())
            predDice2 = dice(image_output[...,2].cpu().detach().numpy(), true_tp[...,2].cpu().detach().numpy())
            predDice3 = dice(image_output[...,3].cpu().detach().numpy(), true_tp[...,3].cpu().detach().numpy())
            predDice4 = dice(image_output[...,4].cpu().detach().numpy(), true_tp[...,4].cpu().detach().numpy())
            
            print("Epoch " + str(now_epoch) + " : pred_0 Dice = ", predDice0)
            print("Epoch " + str(now_epoch) + " : pred_1 Dice = ", predDice1)
            print("Epoch " + str(now_epoch) + " : pred_2 Dice = ", predDice2)
            print("Epoch " + str(now_epoch) + " : pred_3 Dice = ", predDice3)
            print("Epoch " + str(now_epoch) + " : pred_4 Dice = ", predDice4)

            train_dice = (predDice0 + predDice1 + predDice2 + predDice3 + predDice4) / 5.0
            print("--------------------------------------------------")

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if step % args.save_per_step == 0:
            log_str = "Epoch [{}/{}], Step {}: LR={:.7f}, Loss={:.4f}, Dice={:.5f}".format(now_epoch, args.epochs, step, scheduler.get_last_lr()[0], loss.item(), train_dice)
            print(log_str)
            with open(args.log_path   + "/" + "log.txt", 'a', encoding="utf-8", newline='') as outFile:
                print(log_str +'\n', file=outFile)
     
        loss_epoch += loss.item()
        dice_epoch += train_dice

    print("****************************************************\n")
    return model, loss_epoch // (step + 1), dice_epoch // (step + 1)

def valid_step_noAtt(args, now_epoch, valid_loader, model, criterion):
    model.eval()
    loss_epoch, dice_epoch = 0, 0
    for step, (batch_data) in enumerate(valid_loader):
        ori_img = batch_data["img"].permute(0, 3, 1, 2).contiguous()
        true_tp = batch_data["tp_map"]
        with torch.no_grad(): 
            ori_img = ori_img.to(args.device).type(torch.float32)
            true_tp = true_tp.to(args.device)

            image_output = model(ori_img)
            image_output = image_output.permute(0, 2, 3, 1).contiguous()

            loss = criterion(image_output.float().to(args.device), true_tp.float().to(args.device))
            valid_dice = dice(image_output.cpu().detach().numpy(), true_tp.cpu().detach().numpy())

            print("Epoch " + str(now_epoch) + " : red Dice = ", valid_dice)
            print("--------------------------------------------------")

            if step % (args.save_per_step // 2) == 0:
                log_str = "Epoch [{}/{}], Step {}: Loss={:.4f} Dice={:.5f}".format(now_epoch, args.epochs, step, loss.item(), valid_dice)
                print(log_str)
                with open(args.log_path   + "/" + "log.txt", 'a', encoding="utf-8", newline='') as outFile:
                    print(log_str +'\n', file=outFile)
            
            loss_epoch += loss.item()
            dice_epoch += valid_dice
    
    print("====================================================\n")
    return model, loss_epoch // (step + 1), dice_epoch // (step + 1)


# noSelfnoAtt
def train_step_noSelfnoAtt(args, now_epoch, train_loader, model, criterion, optimizer, scheduler):
    model.train()
    scaler = GradScaler(enabled=False)
    loss_epoch, dice_epoch = 0, 0
    for step, (batch_data) in enumerate(train_loader):
        input_imgs = batch_data["img"].permute(0, 3, 1, 2).contiguous()
        true_tp = batch_data["tp_map"]
        with autocast(enabled=False):
            optimizer.zero_grad()
            input_imgs = input_imgs.to(args.device).type(torch.float32)
            image_output = model(input_imgs)

            image_output = image_output.permute(0, 2, 3, 1).contiguous()

            Red = batch_data["img"][...,0]
            Green = batch_data["img"][...,1]
            Blue = batch_data["img"][...,2]
            
            Gray = 0.299 * Red + 0.587 * Green + 0.114 * Blue
            Gray = Gray.unsqueeze(-1)
            
            if args.is_fine_tune:
                loss_0 = criterion(image_output[...,0].float().to(args.device), true_tp[...,0].float().to(args.device))
                loss_1 = criterion(image_output[...,1].float().to(args.device), true_tp[...,1].float().to(args.device))
                loss_2 = criterion(image_output[...,2].float().to(args.device), true_tp[...,2].float().to(args.device))
                loss_3 = criterion(image_output[...,3].float().to(args.device), true_tp[...,3].float().to(args.device))
                loss_4 = criterion(image_output[...,4].float().to(args.device), true_tp[...,4].float().to(args.device))
                loss = loss_0 + loss_1 + loss_2 + loss_3 + loss_4

            predDice0 = dice(image_output[...,0].cpu().detach().numpy(), true_tp[...,0].cpu().detach().numpy())
            predDice1 = dice(image_output[...,1].cpu().detach().numpy(), true_tp[...,1].cpu().detach().numpy())
            predDice2 = dice(image_output[...,2].cpu().detach().numpy(), true_tp[...,2].cpu().detach().numpy())
            predDice3 = dice(image_output[...,3].cpu().detach().numpy(), true_tp[...,3].cpu().detach().numpy())
            predDice4 = dice(image_output[...,4].cpu().detach().numpy(), true_tp[...,4].cpu().detach().numpy())
            
            print("Epoch " + str(now_epoch) + " : pred_0 Dice = ", predDice0)
            print("Epoch " + str(now_epoch) + " : pred_1 Dice = ", predDice1)
            print("Epoch " + str(now_epoch) + " : pred_2 Dice = ", predDice2)
            print("Epoch " + str(now_epoch) + " : pred_3 Dice = ", predDice3)
            print("Epoch " + str(now_epoch) + " : pred_4 Dice = ", predDice4)

            train_dice = (predDice0 + predDice1 + predDice2 + predDice3 + predDice4) / 5.0
            print("--------------------------------------------------")

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if step % args.save_per_step == 0:
            log_str = "Epoch [{}/{}], Step {}: LR={:.7f}, Loss={:.4f}, Dice={:.5f}".format(now_epoch, args.epochs, step, scheduler.get_last_lr()[0], loss.item(), train_dice)
            print(log_str)
            with open(args.log_path   + "/" + "log.txt", 'a', encoding="utf-8", newline='') as outFile:
                print(log_str +'\n', file=outFile)
     
        loss_epoch += loss.item()
        dice_epoch += train_dice

    print("****************************************************\n")
    return model, loss_epoch // (step + 1), dice_epoch // (step + 1)

def valid_step_noSelfnoAtt(args, now_epoch, valid_loader, model, criterion):
    model.eval()
    loss_epoch, dice_epoch = 0, 0
    for step, (batch_data) in enumerate(valid_loader):
        ori_img = batch_data["img"].permute(0, 3, 1, 2).contiguous()
        true_tp = batch_data["tp_map"]
        with torch.no_grad(): 
            ori_img = ori_img.to(args.device).type(torch.float32)
            true_tp = true_tp.to(args.device)

            image_output = model(ori_img)
            image_output = image_output.permute(0, 2, 3, 1).contiguous()

            loss = criterion(image_output.float().to(args.device), true_tp.float().to(args.device))
            valid_dice = dice(image_output.cpu().detach().numpy(), true_tp.cpu().detach().numpy())

            print("Epoch " + str(now_epoch) + " : pred Dice = ", valid_dice)
            print("--------------------------------------------------")

            if step % (args.save_per_step // 2) == 0:
                log_str = "Epoch [{}/{}], Step {}: Loss={:.4f} Dice={:.5f}".format(now_epoch, args.epochs, step, loss.item(), valid_dice)
                print(log_str)
                with open(args.log_path   + "/" + "log.txt", 'a', encoding="utf-8", newline='') as outFile:
                    print(log_str +'\n', file=outFile)
            
            loss_epoch += loss.item()
            dice_epoch += valid_dice
    
    print("====================================================\n")
    return model, loss_epoch // (step + 1), dice_epoch // (step + 1)


# data loader
def get_datagen(args, run_mode):
    file_list, lab_list = [], []
    image_dir_list = args.data_path + args.dataset_name + "/Filter_new/Train/Images/"
    label_dir_list = args.data_path + args.dataset_name + "/Filter_new/Train/Types/"
    image_valid_list = args.data_path + args.dataset_name + "/Filter_new/Test/Images/"
    label_valid_list = args.data_path + args.dataset_name + "/Filter_new/Test/Types/"

    if run_mode == "train":
        file_list.extend(glob.glob("%s/*.png" % image_dir_list))
        lab_list.extend(glob.glob("%s/*.mat" % label_dir_list))
        file_list.sort()
        lab_list.sort()
    else:
        file_list.extend(glob.glob("%s/*.png" % image_valid_list))
        lab_list.extend(glob.glob("%s/*.mat" % label_valid_list))
        file_list.sort()
        lab_list.sort()

    assert len(file_list) > 0, ("No .npy found for `%s`, please check `%s` in `config.py`" % (run_mode, "%s_dir_list" % run_mode))
    print("Dataset %s: image = %d, label = %d" % (run_mode, len(file_list), len(lab_list)))
    
    if args.is_fine_tune:
        # slice_num = int(len(file_list)) - int(len(file_list) * args.label_radio)
        slice_num = int(len(file_list) * 0.9)
        # file_list = file_list[:slice_num + 1]
        # lab_list = lab_list[:slice_num + 1]
        file_list = file_list[slice_num:]
        lab_list = lab_list[slice_num:]
        print("After Slice Dataset %s: image = %d, label = %d" % (run_mode, len(file_list), len(lab_list)))

    input_dataset = FileLoader(file_list, lab_list, preprocessing_mode=args.preprocessing_mode, mode=run_mode, with_type=args.type_classification, input_shape=args.act_shape, mask_shape=args.out_shape)
    dataloader = DataLoader(input_dataset, batch_size=args.batch_size, shuffle=run_mode == "train", drop_last=run_mode == "train")
    return dataloader

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Master_code")
    config = yaml_config_hook("./config.yaml")
    log_str = ""
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
        log_str += str(k) + " : {}\n".format(v)

    args = parser.parse_args()

    for run_no in range(args.run_total):  # logs
        if args.is_fine_tune:
            args.experiment_name = "_" + args.model_type + "_" + args.preprocessing_mode + "_finetune_" + str(int(args.label_radio * 100)) + args.other_info
            args.log_path = args.save_log_path + "Exam" + args.experiment_id + args.dataset_name + args.experiment_name + "_" + str(run_no)
        else:
            args.experiment_name = "_" + args.model_type + "_" + args.preprocessing_mode + "_normal" + args.other_info
            args.log_path = args.save_log_path + "Exam" + args.experiment_id + args.dataset_name + args.experiment_name + "_" + str(run_no)
        
        log_str += ("Final_log_path : " + args.log_path)

        if not os.path.isdir(args.log_path):
            os.mkdir(args.log_path)
        
        with open(args.log_path  + "/" + args.log_path.split("/")[-1] + ".txt", 'w', encoding="utf-8", newline='') as outFile:
            print(log_str, file=outFile)

        args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if args.device == "cuda":
            args.num_gpus = torch.cuda.device_count()
            import os
            os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus_id
        
        # switch type
        if args.model_type == "Master":
            # both Att and self-supervised
            model = Master_all(sparse_radio=args.sparse_radio, backbone=args.backbone, output_stride=16, num_classes=args.nr_types)

        elif args.model_type == "noAspp":
            # remove Aspp
            model = Master_noAspp(sparse_radio=args.sparse_radio, output_stride=16, num_classes=args.nr_types)

        elif args.model_type == "noAtt":
            # add self-supervised but remove Att
            model = Master_noAtt(backbone=args.backbone, output_stride=16, num_classes=args.nr_types)

        elif args.model_type == "noSelf":
            # add Att but remove self-supervised
            model = Master_noSelf(sparse_radio=args.sparse_radio, output_stride=16, num_classes=args.nr_types)

        elif args.model_type == "noSelfnoAtt":
            model = Master_noSelfnoAtt(output_stride=16, num_classes=args.nr_types)

        elif args.model_type == "Unet":
            model = Unet(3, args.nr_types)

        elif args.model_type == "DeepLab":
            model = DeepLab(num_classes=args.nr_types)

        elif args.model_type == "SimCLR":
            model = SimCLR_addDecoder(args.freeze, num_classes=args.nr_types)

        args.pretrained_path = args.pretrained_path[:-19] + str(run_no) + "/checkpoint_10.tar"
        if args.is_fine_tune:
            model = load_model_weights(args, model)
        
        args.seed = args.seed + run_no
        basic(args, model)
    print("Final Save Path : ", args.log_path)