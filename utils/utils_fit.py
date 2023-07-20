import os
import torch
import random
import numpy as np
from tqdm import tqdm
from monai.losses import DiceCELoss

from .losses import CE_Loss, Dice_loss, Focal_Loss, weights_init
from .utils import get_lr, generate_click_prompt
from .utils_metrics import f_score


def fit_one_epoch(model_train, model, loss_history, eval_callback, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, Epoch, cuda, dice_loss, focal_loss, cls_weights, num_classes, \
    fp16, scaler, save_period, save_dir, alpha, local_rank=0):
    total_loss      = 0
    total_f_score   = 0

    val_loss        = 0
    val_f_score     = 0

    dice_ce = DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')

    if local_rank == 0:
        print('Start Train')
        pbar = tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)
    model_train.train()
    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step: 
            break
        imgs, pngs, seg_labels = batch
        seg_labels = seg_labels.permute(0, 3, 1, 2)

        with torch.no_grad():
            weights = torch.from_numpy(cls_weights)
            if cuda:
                imgs = imgs.cuda(local_rank)
                pngs = pngs.cuda(local_rank)
                seg_labels = seg_labels.cuda(local_rank)
                weights = weights.cuda(local_rank)
            imgs, pts, pngs = generate_click_prompt(imgs, pngs)
            point_labels = torch.ones(imgs.size(0))
            if point_labels[0] != -1:
                # point_coords = samtrans.ResizeLongestSide(longsize).apply_coords(pt, (h, w))
                point_coords = pts
                coords_torch = torch.as_tensor(point_coords, dtype=torch.float).cuda(local_rank)
                labels_torch = torch.as_tensor(point_labels, dtype=torch.int).cuda(local_rank)
                coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]
                pts = (coords_torch, labels_torch)
        #----------------------#
        #   清零梯度
        #----------------------#
        model_train.zero_grad()
        if not fp16:
            #----------------------#
            #   前向传播
            #----------------------#
            for n, value in model_train.module.image_encoder.named_parameters():
                if "Adapter" not in n:
                    value.requires_grad = False
            img_e = model_train.module.image_encoder(imgs)

            with torch.no_grad():
                sparse_e, dense_e = model_train.module.prompt_encoder(points=pts, boxes=None, masks=None)
            
            outputs, _ = model_train.module.mask_decoder(
                image_embeddings=img_e,
                image_pe=model_train.module.prompt_encoder.get_dense_pe(), 
                sparse_prompt_embeddings=sparse_e,
                dense_prompt_embeddings=dense_e, 
                multimask_output=False,
            )
            #----------------------#
            #   计算损失
            #----------------------#
            if focal_loss:
                loss = Focal_Loss(outputs, pngs, weights, num_classes = num_classes)
            else:
                # loss = CE_Loss(outputs, pngs, weights, num_classes = num_classes)
                loss = dice_ce(outputs, pngs.unsqueeze(1))

            if dice_loss:
                main_dice = Dice_loss(outputs, seg_labels.permute(0, 2, 3, 1))
                loss      = loss + main_dice

            with torch.no_grad():
                #-------------------------------#
                #   计算f_score
                #-------------------------------#
                _f_score = f_score(outputs, seg_labels.permute(0, 2, 3, 1))

            #----------------------#
            #   反向传播
            #----------------------#
            loss.backward()
            optimizer.step()
        else:
            from torch.cuda.amp import autocast
            with autocast():
                #----------------------#
                #   前向传播
                #----------------------#
                for n, value in model_train.module.image_encoder.named_parameters():
                    if "Adapter" not in n:
                        value.requires_grad = False
                img_e = model_train.module.image_encoder(imgs)

                with torch.no_grad():
                    sparse_e, dense_e = model_train.module.prompt_encoder(points=pts, boxes=None, masks=None)
                
                outputs, _ = model_train.module.mask_decoder(
                    image_embeddings=img_e,
                    image_pe=model_train.module.prompt_encoder.get_dense_pe(), 
                    sparse_prompt_embeddings=sparse_e,
                    dense_prompt_embeddings=dense_e, 
                    multimask_output=False,
                )
                #----------------------#
                #   计算损失
                #----------------------#
                if focal_loss:
                    loss = Focal_Loss(outputs, pngs, weights, num_classes = num_classes)
                else:
                    loss = CE_Loss(outputs, pngs, weights, num_classes = num_classes)

                if dice_loss:
                    main_dice = Dice_loss(outputs, seg_labels)
                    loss      = loss + main_dice

                with torch.no_grad():
                    #-------------------------------#
                    #   计算f_score
                    #-------------------------------#
                    _f_score = f_score(outputs, seg_labels)
                    
            #----------------------#
            #   反向传播
            #----------------------#
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        total_loss      += loss.item()
        total_f_score   += _f_score.item()
            
        if local_rank == 0:
            pbar.set_postfix(**{'seg_loss': total_loss / (iteration + 1), 
                                'f_score'   : total_f_score / (iteration + 1),
                                'lr'        : get_lr(optimizer)})
            pbar.update(1)

    if local_rank == 0:
        pbar.close()
        print('Finish Train')
        print('Start Validation')
        pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)

    model_train.eval()
    for iteration, batch in enumerate(gen_val):
        if iteration >= epoch_step_val:
            break
        imgs, pngs, seg_labels = batch
        with torch.no_grad():
            weights = torch.from_numpy(cls_weights)
            if cuda:
                imgs = imgs.cuda(local_rank)
                pngs = pngs.cuda(local_rank)
                seg_labels = seg_labels.cuda(local_rank)
                weights = weights.cuda(local_rank)

            imgs, pts, pngs = generate_click_prompt(imgs, pngs)
            point_labels = torch.ones(imgs.size(0))
            if point_labels[0] != -1:
                # point_coords = samtrans.ResizeLongestSide(longsize).apply_coords(pt, (h, w))
                point_coords = pts
                coords_torch = torch.as_tensor(point_coords, dtype=torch.float).cuda(local_rank)
                labels_torch = torch.as_tensor(point_labels, dtype=torch.int).cuda(local_rank)
                coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]
                pts = (coords_torch, labels_torch)
            #----------------------#
            #   前向传播
            #----------------------#
            with torch.no_grad():
                img_e = model_train.module.image_encoder(imgs)

                sparse_e, dense_e = model_train.module.prompt_encoder(points=pts, boxes=None, masks=None)
            
                outputs, _ = model_train.module.mask_decoder(
                    image_embeddings=img_e,
                    image_pe=model_train.module.prompt_encoder.get_dense_pe(), 
                    sparse_prompt_embeddings=sparse_e,
                    dense_prompt_embeddings=dense_e, 
                    multimask_output=False,
                )
            #----------------------#
            #   计算损失
            #----------------------#
            if focal_loss:
                loss = Focal_Loss(outputs, pngs, weights, num_classes = num_classes)
            else:
                # loss = CE_Loss(outputs, pngs, weights, num_classes = num_classes)
                loss = dice_ce(outputs, pngs.unsqueeze(1))

            if dice_loss:
                main_dice = Dice_loss(outputs, seg_labels)
                loss  = loss + main_dice
            #-------------------------------#
            #   计算f_score
            #-------------------------------#
            _f_score    = f_score(outputs, seg_labels.permute(0, 2, 3, 1))

            val_loss    += loss.item()
            val_f_score += _f_score.item()
            
            if local_rank == 0:
                pbar.set_postfix(**{'seg_loss': val_loss / (iteration + 1), 
                                    'f_score'   : val_f_score / (iteration + 1),
                                    'lr'        : get_lr(optimizer)})
                pbar.update(1)
            
    if local_rank == 0:
        pbar.close()
        print('Finish Validation')
        loss_history.append_loss(epoch + 1, total_loss / epoch_step, val_loss / epoch_step_val)
        eval_callback.on_epoch_end(epoch + 1, model_train)
        print('Epoch:'+ str(epoch + 1) + '/' + str(Epoch))
        print('Total Loss: %.3f || Val Loss: %.3f ' % (total_loss / epoch_step, val_loss / epoch_step_val))
        
        #-----------------------------------------------#
        #   保存权值
        #-----------------------------------------------#
        if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
            torch.save(model.state_dict(), os.path.join(save_dir, 'ep%03d-loss%.3f-val_loss%.3f.pth' % (epoch + 1, total_loss / epoch_step, val_loss / epoch_step_val)))

        if len(loss_history.val_loss) <= 1 or (val_loss / epoch_step_val) <= min(loss_history.val_loss):
            print('Save best model to best_epoch_weights.pth')
            torch.save(model.state_dict(), os.path.join(save_dir, "best_epoch_weights.pth"))
            
        torch.save(model.state_dict(), os.path.join(save_dir, "last_epoch_weights.pth"))