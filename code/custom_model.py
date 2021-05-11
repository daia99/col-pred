from argparse import ArgumentParser
import os
import re

import numpy as np
import cv2
import torch
import pytorch_lightning as pl
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.metrics import ConfusionMatrix
import wandb
# from pytorch_lightning.loggers import TensorBoardLogger

from torchvision.datasets.mnist import MNIST
from torchvision import transforms
from torchvideotransforms import video_transforms, volume_transforms

from einops import rearrange, reduce, repeat

from vit_model import ViT
from model import CRW
import utils
from torchsummary import summary
from recorder import Recorder

from datamodule.custom_datamodule import RawDataModule


# class Backbone(torch.nn.Module):
#     def __init__(self, hidden_dim=128):
#         super().__init__()
#         self.l1 = torch.nn.Linear(28 * 28, hidden_dim)
#         self.l2 = torch.nn.Linear(hidden_dim, 10)

#     def forward(self, x):
#         x = x.view(x.size(0), -1)
#         x = torch.relu(self.l1(x))
#         x = torch.relu(self.l2(x))
#         return x


# compute rollout between attention layers
def compute_rollout_attention(attn, discard_ratio=0.8, fusion_mode='max', start_layer=0):
    # adding residual consideration- code adapted from https://github.com/samiraabnar/attention_flow and https://github.com/jacobgil/vit-explain
    # fuse heads and discard lowest attentions
    attn_fused = reduce(attn, 'b d head h w -> b d h w', fusion_mode) # get the mean attention from each head
    flat = attn_fused.view(attn_fused.size(0)*attn_fused.size(1), -1) # view original fused attn for simpler discarding
    _, indices = flat.topk(int(flat.size(-1)*discard_ratio), -1, largest=False) # get indices of smallest attns per attention layer
    for i, idx in enumerate(indices):
        idx = idx[idx != 0] # keep class idx
        flat[i, idx] = 0
    # to calculate A-hat, add identity matrix and re-normalize the result
    num_tokens = attn_fused.shape[2]
    batch_size = attn_fused.shape[0]
    layer_size = attn_fused.shape[1]
    eye = torch.eye(num_tokens).expand(batch_size, layer_size, num_tokens, num_tokens).cuda() # identity matrix
    attn_eye = attn_fused + eye
    attn_rollout = attn_eye / attn_eye.sum(dim=-1, keepdim=True) # renormalize
    rollout_batch = []
    a_hat = attn_rollout[:,0] # get rollout attention maps in first layer for whole batch
    for i in range(start_layer+1, layer_size):
        a_hat = a_hat.bmm(attn_rollout[:,i]) # output is size (b, p, p)
        
    return a_hat


# create heatmap from mask on image
def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return cam

# create heatmap from mask on image
def show_raw_map(mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    return heatmap


def generate_visualization(original_image, transformer_attribution, rollout_full):
    # transformer_attribution = transformer_attribution.reshape(1, 1, 8, 26)
    height = transformer_attribution.shape[0]*16
    width = transformer_attribution.shape[1]*16
    transformer_attribution = transformer_attribution[None, None]
    transformer_attribution = torch.nn.functional.interpolate(transformer_attribution, scale_factor=16, mode='bilinear')
    transformer_attribution = transformer_attribution.reshape(height, width).cpu().numpy()
    transformer_attribution = np.clip(((transformer_attribution - float(rollout_full.min())) / float(0.02 - rollout_full.min())),0.,1.)
    image_transformer_attribution = rearrange(original_image, 'c h w -> h w c')
    image_transformer_attribution = (image_transformer_attribution - image_transformer_attribution.min()) / (image_transformer_attribution.max() - image_transformer_attribution.min())
    vis = show_cam_on_image(image_transformer_attribution, transformer_attribution)
    # vis = show_raw_map(transformer_attribution)
    vis =  np.uint8(255 * vis)
    vis = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
    return vis # shape (h, w, c)


# def generate_visualization_raw(original_image, transformer_attribution):
#     # transformer_attribution = transformer_attribution.reshape(1, 1, 8, 26)
#     height = transformer_attribution.shape[0]*16
#     width = transformer_attribution.shape[1]*16
#     transformer_attribution = transformer_attribution[None, None]
#     transformer_attribution = torch.nn.functional.interpolate(transformer_attribution, scale_factor=16, mode='bilinear')
#     transformer_attribution = transformer_attribution.reshape(height, width).cpu().numpy()
#     transformer_attribution = (transformer_attribution - transformer_attribution.min()) / (transformer_attribution.max() - transformer_attribution.min())
#     image_transformer_attribution = rearrange(original_image, 'c h w -> h w c')
#     image_transformer_attribution = (image_transformer_attribution - image_transformer_attribution.min()) / (image_transformer_attribution.max() - image_transformer_attribution.min())
#     vis = show_cam_on_image(image_transformer_attribution, transformer_attribution)
#     vis =  np.uint8(255 * vis)
#     # vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)
#     return vis # shape (h, w, c)


class CollTransformer(pl.LightningModule):
    def __init__(self, args, transformer, backbone, vis=None):
        super().__init__()
        self.save_hyperparameters()        
        self.args = args

        # freeze CNN backbone layers
        self.feat_net = backbone
        for child in self.feat_net.children():
            for param in child.parameters():
                param.requires_grad = False
        
        # Initialize Transformer Module
        self.transformer_module = transformer

        weight_factor = args.bce_weight
        pos_weight = torch.ones([args.forward_context])*weight_factor # reduce weight of positive 'go' predictions
        self.out_loss = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
        # determine dims of some model parts
        self.forward_context_len = args.forward_context
        self.back_context_len = args.back_context
        # stats counters
        
        self.train_correct_pred_count = torch.zeros(args.forward_context, dtype=torch.float) # counter for number of correct predictions
        self.val_correct_pred_count = torch.zeros(args.forward_context, dtype=torch.float)
        
        self.num_train_go_preds = 0 # number of positive predictions (go), num_stop_preds = total_preds - go_preds
        self.num_val_go_preds = 0
        
        # self.num_train_go_labels = 538576 # number of positive labels (go), num_stop_labels = total_labels - go_labels
        # self.num_val_go_labels = 59718
        
        self.train_steps_in_epoch = 0
        self.val_steps_in_epoch = 0

        # extra stats to keep track of predictions containing two labels
        self.train_preds_partial_array = 0
        self.train_labels_partial_array = 0
        self.val_preds_partial_array = 0
        self.val_labels_partial_array = 0
        self.val_correct_part_pred_count = torch.zeros(args.forward_context, dtype=torch.float)

        # misc logging stats
        self.part_sample_count = 0 # count number of logged samples in set
        self.wandb_table = wandb.Table(columns=["Sample Index", "Predicted Label", "True Label"])
        # # confusion matrix
        # self.train_confusion_matrix = torch.zeros(2, 2)
        # self.val_confusion_matrix = torch.zeros(2, 2)

        # # extra stats to count first time step predictions and labels
        # self.train_first_label_pos = 0
        # self.val_first_label_pos = 0

        # if self.epochs_from_start is not None:
        #     self.current_epoch = self.epochs_from_start
        # else:
        #     self.epochs_from_start = 2
        
    def forward(self, x):
        # use forward for inference/predictions
        embedding = self.feat_net.encoder(x)
        feat_video = rearrange(embedding,'b c t h w -> t (b c) h w')
        feature = self.transformer_module(feat_video)
        return feature

    def training_step(self, sample, batch_idx):
        # video = sample['rgb_context'] # list of images (t,b,c,h,w)
        throttle_data = torch.stack(sample['throttle'], dim=0) # list of binary labels (for stop/go between set number of frames)
        throttle_data = rearrange(throttle_data, 't b -> b t').float()
        back_context = sample['rgb_context'] # video_transform outputs each batch item as (c, t, h, w)
        conv_feature = self.feat_net.encoder(back_context) # (b, 256, t, h, w)
        
        # prediction and loss
        pred_throttle_logit = self.transformer_module(conv_feature)
        loss = self.out_loss(pred_throttle_logit, throttle_data) # single loss value output
        
        # get predictions as labels
        pred_throttle = torch.sigmoid(pred_throttle_logit)
        pred_binary = torch.round(pred_throttle).float()
        throttle_data_round = torch.round(throttle_data).float()
        label = torch.isclose(throttle_data_round, pred_binary).float().cpu()
        
        # count number of pos preds and labels
        self.num_train_go_preds += torch.sum(pred_binary)
        # self.num_train_go_labels += torch.sum(throttle_data)

        self.train_correct_pred_count = self.train_correct_pred_count.add(torch.sum(label, dim=0))
        self.train_steps_in_epoch += 1
        
        # Log training loss
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)

        # extra stats (val_part_labels = 984, train_part_labels = 8720)
        sum_pred_arrays = torch.sum(pred_binary, dim=1)
        # sum_label_arrays = torch.sum(throttle_data_round, dim=1)
        num_partial_preds = torch.sum(torch.eq(sum_pred_arrays < self.args.forward_context, sum_pred_arrays > 0).float().cpu())
        # num_partial_labels = torch.sum(torch.eq(sum_label_arrays < self.args.forward_context, sum_label_arrays > 0).float().cpu())
        self.train_preds_partial_array += num_partial_preds
        # self.train_labels_partial_array += num_partial_labels

        return loss

    # def training_step_end(self, batch_parts):
    #     gpu_0_prediction = batch_parts[0]['pred']
    #     gpu_1_prediction = batch_parts[1]['pred']

    #     # do something with both outputs
    #     return (batch_parts[0]['loss'] + batch_parts[1]['loss']) / 2

    def training_epoch_end(self, training_step_outputs):
        # for us, a list of losses at each step
        # for out in training_step_outputs:
        #     # do something with preds
        batch_size = self.args.batch_size
        epoch_steps = self.train_steps_in_epoch
        forward_context = self.forward_context_len
        max_correct_preds = torch.ones(forward_context, dtype=torch.float)*epoch_steps*batch_size
        num_preds = epoch_steps*batch_size*forward_context
        frame_pred_accuracy = torch.div(self.train_correct_pred_count, max_correct_preds).numpy()
        
        # log total preds and labels
        # self.log('train_preds_total_elements', num_preds)
        self.log('train_preds_pos', self.num_train_go_preds)
        # self.log('train_labels_pos', self.num_train_go_labels)
        self.log('train_part_preds', self.train_preds_partial_array)
        # self.log('train_part_labels', self.train_labels_partial_array)
        
        # init back to zeros before next epoch
        self.train_steps_in_epoch = 0
        self.train_correct_pred_count = torch.zeros(forward_context, dtype=torch.float)
        self.num_train_go_preds = 0
        # self.num_train_go_labels = 0
        self.train_preds_partial_array = 0
        # self.train_labels_partial_array = 0
        self.train_confusion_matrix = torch.zeros(2, 2)

        for i in range(frame_pred_accuracy.size):
            str1 = 'train_acc_'
            str2 = str(i)
            label = str1 + str2
            self.log(label, frame_pred_accuracy[i])

    def validation_step(self, sample, batch_idx):
        # video = sample['rgb_context'] # list of images (t,b,c,h,w)
        throttle_data = torch.stack(sample['throttle'], dim=0) # list of binary labels (for stop/go between set number of frames)
        throttle_data = rearrange(throttle_data, 't b -> b t').float()
        back_context = sample['rgb_context'] # video_transform outputs each batch item as (c, t, h, w)
        conv_feature = self.feat_net.encoder(back_context) # (b, 256, t, h, w)

        # prediction and loss
        pred_throttle_logit = self.transformer_module(conv_feature)       
        loss = self.out_loss(pred_throttle_logit, throttle_data) # single loss value output
        
        # get predictions as labels
        pred_throttle = torch.sigmoid(pred_throttle_logit)
        pred_binary = torch.round(pred_throttle).float()
        throttle_data_round = torch.round(throttle_data).float()
        label = torch.isclose(throttle_data_round, pred_binary).float().cpu()

        # count number of pos preds and labels
        self.num_val_go_preds += torch.sum(pred_binary)
        # self.num_val_go_labels += torch.sum(throttle_data)

        self.val_correct_pred_count = self.val_correct_pred_count.add(torch.sum(label, dim=0))
        self.val_steps_in_epoch += 1
        # print(loss)
        # Log validation loss (will be automatically averaged over an epoch)
        self.log('valid_loss', loss)

        # extra stats
        sum_pred_arrays = torch.sum(pred_binary, dim=1) # array of dimension [b, 1]
        sum_label_arrays = torch.sum(throttle_data_round, dim=1)
        # has_part_pred_at_idx = torch.eq(sum_pred_arrays < self.args.forward_context,sum_pred_arrays > 0).long()
        has_part_label_at_idx = torch.eq(sum_label_arrays < self.args.forward_context,sum_label_arrays > 0).long()
        # idx_part_pred = torch.nonzero(has_part_pred_at_idx, as_tuple = True)
        idx_part_label = torch.nonzero(has_part_label_at_idx, as_tuple = True)
        num_partial_preds = torch.sum(torch.eq(sum_pred_arrays < self.args.forward_context,sum_pred_arrays > 0).float().cpu())
        num_partial_labels = torch.sum(torch.eq(sum_label_arrays < self.args.forward_context, sum_label_arrays > 0).float().cpu())
        # num_partial_preds = torch.sum(torch.where(sum_pred_arrays < self.args.forward_context != sum_pred_arrays > 0,torch.ones(16),torch.zeros(16))).float().cpu()
        # num_partial_labels = torch.sum(torch.where((sum_label_arrays < self.args.forward_context or sum_pred_arrays > 0),torch.ones(16),torch.zeros(16))).float().cpu()
        self.val_preds_partial_array += num_partial_preds
        self.val_labels_partial_array += num_partial_labels
        # run this if there are any indices for partial labels
        if torch.numel(idx_part_label[0]) != 0:
            # part_labels = torch.index_select(label, 0, idx_part_label)
            part_correct = label[idx_part_label] # number of correct part predictions
            self.val_correct_part_pred_count = self.val_correct_part_pred_count.add(torch.sum(part_correct, dim=0))
            # # get the partial ground truth and corresponding predictions
            # part_labels = throttle_data_round[idx_part_label]
            # part_preds = pred_binary[idx_part_label]
            # part_back_context = back_context[idx_part_label]
            # for i in range(len(idx_part_label)):
            #     part_label = part_labels[i]
            #     part_pred = part_preds[i]
            #     part_back_context_sample = part_back_context[i]
            #     pred_str = str(part_pred.int().cpu().numpy())
            #     label_str = str(part_label.int().cpu().numpy())
            #     vid_idx = str(self.part_sample_count)
            #     self.part_sample_count += 1
            #     self.wandb_table.add_data(vid_idx, pred_str, label_str)
            #     str1 = "Video_idx_" + vid_idx + " - label: " + label_str
            #     inv_norm = [
            #         video_transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], std=[1/0.229, 1/0.224, 1/0.225])
            #     ]
            #     video_transform = video_transforms.Compose(inv_norm)
            #     part_back_context_sample = video_transform(part_back_context_sample).cpu().numpy()
            #     part_back_context_sample = rearrange(part_back_context_sample, 'c t h w -> t c w h')
            #     self.logger.experiment.log({str1: wandb.Video(part_back_context_sample, fps=10, format="gif")})

        # return pred_throttle, throttle_data_round
        return {'pred': pred_throttle, 'label': throttle_data_round}

    def validation_epoch_end(self, validation_step_outputs): # validation_step_outputs is list of tuples
        batch_size = self.args.batch_size
        epoch_steps = self.val_steps_in_epoch
        forward_context = self.forward_context_len
        max_correct_preds = torch.ones(forward_context, dtype=torch.float)*epoch_steps*batch_size
        num_preds = epoch_steps*batch_size*forward_context
        frame_pred_accuracy = torch.div(self.val_correct_pred_count, max_correct_preds).numpy()
        frame_part_pred_accuracy = torch.div(self.val_correct_part_pred_count, self.val_labels_partial_array).numpy()

        # stats for confusion matrix and ROC
        preds = torch.cat([i['pred'] for i in validation_step_outputs], dim=0)
        labels = torch.cat([i['label'] for i in validation_step_outputs], dim=0)
        # create list of predictions and labels for each time step
        preds_at_time_steps = [preds[:,i].view(-1).double().cpu().numpy() for i in range(forward_context)]
        labels_at_time_steps = [labels[:,i].view(-1).long().cpu().numpy() for i in range(forward_context)]

        preds_array = preds.view(-1).double().cpu().numpy() # turn to single array
        labels_array = labels.view(-1).long().cpu().numpy()
        preds_array_rounded = np.round(preds_array, decimals=0)
        preds_probs = np.stack(((-preds_array + 1), preds_array),axis = -1)
        # confmat = ConfusionMatrix(num_classes=2)
        # confusion_matrix = confmat(preds_tensor, labels_tensor)
        self.logger.experiment.log({"val_conf_mat" : wandb.plot.confusion_matrix(
                        probs=None,
                        y_true=labels_array,
                        preds=preds_array_rounded,
                        class_names=["stop", "go"])
        })
        self.logger.experiment.log({"val_roc" : wandb.plot.roc_curve(
                        labels_array,
                        preds_probs,
                        labels=["stop", "go"])
                        # classes_to_plot=["go", "stop"])
        })
        # self.logger.experiment.log({"Partial Label Predictions": self.wandb_table})
        # log total preds and labels
        # self.log('val_preds_total_elements', num_preds)
        self.log('val_preds_pos', self.num_val_go_preds)
        # self.log('val_labels_pos', self.num_val_go_labels)
        self.log('val_part_preds', self.val_preds_partial_array)
        self.log('val_part_labels', self.val_labels_partial_array)

        # init back to zeros before next epoch
        self.val_steps_in_epoch = 0
        self.val_correct_pred_count = torch.zeros(forward_context, dtype=torch.float)
        self.val_correct_part_pred_count = torch.zeros(forward_context, dtype=torch.float)
        self.num_val_go_preds = 0
        # self.num_val_go_labels = 0
        self.val_preds_partial_array = 0
        self.val_labels_partial_array = 0
        # self.val_confusion_matrix = torch.zeros(2, 2)
        # self.part_sample_count = 0

        for i in range(forward_context):
            str1 = 'val_acc_'
            str2 = str(i)
            str3 = 'val_part_acc_'
            str4 = 'val_conf_mat_'
            str5 = 'val_roc_'
            label_log = str1 + str2
            part_label_log = str3 + str2
            conf_log = str4 + str2
            roc_log = str5 + str2
            preds_prob_at_time = np.stack(((-preds_at_time_steps[i] + 1), preds_at_time_steps[i]),axis = -1)
            preds_at_time_rounded = np.round(preds_at_time_steps[i], decimals=0)
            labels_at_time = labels_at_time_steps[i]
            self.log(label_log, frame_pred_accuracy[i])
            self.log(part_label_log, frame_part_pred_accuracy[i])
            self.logger.experiment.log({conf_log : wandb.plot.confusion_matrix(
                        probs=None,
                        y_true=labels_at_time,
                        preds=preds_at_time_rounded,
                        class_names=["stop", "go"])
            })
            self.logger.experiment.log({roc_log : wandb.plot.roc_curve(
                            labels_at_time,
                            preds_prob_at_time,
                            labels=["stop", "go"])
            })

    def test_step(self, sample, batch_idx):
        # video = sample['rgb_context'] # list of images (t,b,c,h,w)
        throttle_data = torch.stack(sample['throttle'], dim=0) # list of binary labels (for stop/go between set number of frames)
        throttle_data = rearrange(throttle_data, 't b -> b t').float()
        back_context = sample['rgb_context'] # video_transform outputs each batch item as (c, t, h, w)        
        conv_feature = self.feat_net.encoder(back_context) # (b, 256, t, h, w)

        # prediction and loss
        vit_hook = Recorder(self.transformer_module)
        pred_throttle_logit, attns = vit_hook(conv_feature)
        vit_hook = vit_hook.eject()       
        loss = self.out_loss(pred_throttle_logit, throttle_data) # single loss value output
        
        # rollout relevance visualization
        rollout = compute_rollout_attention(attns)
        rollout_map = rollout[:,0,1:] # access row for cls embedding only, excluding first token, shape (b, s-1)
        rollout_projected = rearrange(rollout_map, 'b (t h w) -> b t h w', t=self.args.back_context, h=int(self.args.img_height/16)) # rearrange batches of maps to correspond to video frame patches
        attn_raw_last = reduce(attns[:,-1], 'b head h w -> b h w', 'mean')
        attn_raw_last = attn_raw_last[:,0,1:]
        attn_raw_last = rearrange(attn_raw_last, 'b (t h w) -> b t h w', t=self.args.back_context, h=int(self.args.img_height/16))

        # get predictions as labels
        pred_throttle = torch.sigmoid(pred_throttle_logit)
        pred_binary = torch.round(pred_throttle).float()
        throttle_data_round = torch.round(throttle_data).float()
        label = torch.isclose(throttle_data_round, pred_binary).float().cpu()

        # count number of pos preds and labels
        self.num_val_go_preds += torch.sum(pred_binary)
        # self.num_val_go_labels += torch.sum(throttle_data)

        self.val_correct_pred_count = self.val_correct_pred_count.add(torch.sum(label, dim=0))
        self.val_steps_in_epoch += 1
        # print(loss)
        # Log validation loss (will be automatically averaged over an epoch)
        # self.log('valid_loss', loss)

        # extra stats
        sum_pred_arrays = torch.sum(pred_binary, dim=1) # array of dimension [b, 1]
        sum_label_arrays = torch.sum(throttle_data_round, dim=1)
        # has_part_pred_at_idx = torch.eq(sum_pred_arrays < self.args.forward_context,sum_pred_arrays > 0).long()
        has_part_label_at_idx = torch.eq(sum_label_arrays < self.args.forward_context, sum_label_arrays > 0).long()
        # idx_part_pred = torch.nonzero(has_part_pred_at_idx, as_tuple = True)
        idx_part_label = torch.nonzero(has_part_label_at_idx, as_tuple = True)
        num_partial_preds = torch.sum(torch.eq(sum_pred_arrays < self.args.forward_context, sum_pred_arrays > 0).float().cpu())
        num_partial_labels = torch.sum(torch.eq(sum_label_arrays < self.args.forward_context, sum_label_arrays > 0).float().cpu())
        self.val_preds_partial_array += num_partial_preds
        self.val_labels_partial_array += num_partial_labels
        # run this if there are any indices for partial labels
        if torch.numel(idx_part_label[0]) != 0:
            # part_labels = torch.index_select(label, 0, idx_part_label)
            part_correct = label[idx_part_label] # number of correct part predictions
            self.val_correct_part_pred_count = self.val_correct_part_pred_count.add(torch.sum(part_correct, dim=0))
            # get the partial ground truth and corresponding predictions
            part_labels = throttle_data_round[idx_part_label]
            part_preds = pred_binary[idx_part_label]
            part_back_context = back_context[idx_part_label]
            part_rollout = rollout_projected[idx_part_label]
            part_rollout_array = rollout_map[idx_part_label]
            part_attn_raw = attn_raw_last[idx_part_label]
            for i in range(len(idx_part_label)):
                part_label = part_labels[i]
                part_pred = part_preds[i]
                part_back_context_sample = part_back_context[i]
                part_rollout_sample = part_rollout[i]
                part_attn_sample = part_attn_raw[i]
                pred_str = str(part_pred.int().cpu().numpy())
                label_str = re.sub("\[|\]| ", "", str(part_label.int().cpu().numpy()))
                vid_idx = str(self.part_sample_count)
                self.part_sample_count += 1
                self.wandb_table.add_data(vid_idx, pred_str, label_str)
                str1 = "Video_idx_" + vid_idx + "_label_" + label_str
                str2 = "rollout_idx_" + vid_idx
                inv_norm = [
                    video_transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], std=[1/0.229, 1/0.224, 1/0.225])
                ]
                video_transform = video_transforms.Compose(inv_norm)
                part_back_context_sample = video_transform(part_back_context_sample).cpu().numpy()
                part_back_context_sample = rearrange(part_back_context_sample, 'c t h w -> t c h w')
                self.logger.experiment.log({str1: wandb.Video(np.uint8(part_back_context_sample*255), fps=10, format="gif")})
                # visualize rollout projection
                vis_frames = []
                rollout_sample = part_rollout_array[i]
                for j in range(part_back_context_sample.shape[0]):
                    frame = part_back_context_sample[j]
                    rollout_frame = part_rollout_sample[j]
                    frame_attn = part_attn_sample[j]
                    mapped_frame = generate_visualization(frame, rollout_frame, part_rollout_sample)
                    vis_frames.append(mapped_frame)
                vis_frames = rearrange(vis_frames, 't h w c -> t c h w')
                self.logger.experiment.log({str2: wandb.Video(vis_frames, fps=2, format="gif")})

        # return pred_throttle, throttle_data_round
        return {'pred': pred_throttle, 'label': throttle_data_round}

    def test_epoch_end(self, validation_step_outputs): # validation_step_outputs is list of tuples
        batch_size = self.args.batch_size
        epoch_steps = self.val_steps_in_epoch
        forward_context = self.forward_context_len
        max_correct_preds = torch.ones(forward_context, dtype=torch.float)*epoch_steps*batch_size
        num_preds = epoch_steps*batch_size*forward_context
        frame_pred_accuracy = torch.div(self.val_correct_pred_count, max_correct_preds).numpy()
        frame_part_pred_accuracy = torch.div(self.val_correct_part_pred_count, self.val_labels_partial_array).numpy()
        
        # plot line chart of accuracy versus timestep, both whole and partial
        pred_acc_list = frame_pred_accuracy.tolist()
        part_pred_acc_list = frame_part_pred_accuracy.tolist()
        x_axis = list(range(len(pred_acc_list)))
        acc_data = [[x, y] for (x, y) in zip(x_axis, pred_acc_list)]
        part_acc_data = [[x, y] for (x, y) in zip(x_axis, part_pred_acc_list)]
        table1 = wandb.Table(data=acc_data, columns = ["Timestep", "Accuracy"])
        self.logger.experiment.log({"Line_plot_val_acc" : wandb.plot.line(table1, "Timestep", "Accuracy",
                  title="Accuracy vs Timestep")})
        table2 = wandb.Table(data=part_acc_data, columns = ["Timestep", "Accuracy"])
        self.logger.experiment.log({"Line_plot_val_part_acc" : wandb.plot.line(table2, "Timestep", "Accuracy",
                  title="Accuracy vs Timestep")})

        # stats for confusion matrix and ROC
        preds = torch.cat([i['pred'] for i in validation_step_outputs], dim=0)
        labels = torch.cat([i['label'] for i in validation_step_outputs], dim=0)
        # create list of predictions and labels for each time step
        preds_at_time_steps = [preds[:,i].view(-1).double().cpu().numpy() for i in range(forward_context)]
        labels_at_time_steps = [labels[:,i].view(-1).long().cpu().numpy() for i in range(forward_context)]

        preds_array = preds.view(-1).double().cpu().numpy() # turn to single array
        labels_array = labels.view(-1).long().cpu().numpy()
        preds_array_rounded = np.round(preds_array, decimals=0)
        preds_probs = np.stack(((-preds_array + 1), preds_array),axis = -1)
        # confmat = ConfusionMatrix(num_classes=2)
        # confusion_matrix = confmat(preds_tensor, labels_tensor)
        self.logger.experiment.log({"val_conf_mat" : wandb.plot.confusion_matrix(
                        probs=None,
                        y_true=labels_array,
                        preds=preds_array_rounded,
                        class_names=["stop", "go"])
        })
        self.logger.experiment.log({"val_roc" : wandb.plot.roc_curve(
                        labels_array,
                        preds_probs,
                        labels=["stop", "go"])
                        # classes_to_plot=["go", "stop"])
        })
        self.logger.experiment.log({"Partial Label Predictions": self.wandb_table})
        # log total preds and labels
        # self.log('val_preds_total_elements', num_preds)
        # self.log('val_preds_pos', self.num_val_go_preds)
        # # self.log('val_labels_pos', self.num_val_go_labels)
        # self.log('val_part_preds', self.val_preds_partial_array)
        # self.log('val_part_labels', self.val_labels_partial_array)

        # init back to zeros before next epoch
        self.val_steps_in_epoch = 0
        self.val_correct_pred_count = torch.zeros(forward_context, dtype=torch.float)
        self.val_correct_part_pred_count = torch.zeros(forward_context, dtype=torch.float)
        self.num_val_go_preds = 0
        # self.num_val_go_labels = 0
        self.val_preds_partial_array = 0
        self.val_labels_partial_array = 0
        # self.val_confusion_matrix = torch.zeros(2, 2)
        self.part_sample_count = 0

        for i in range(forward_context):
            str1 = 'val_acc_'
            str2 = str(i)
            str3 = 'val_part_acc_'
            str4 = 'val_conf_mat_'
            str5 = 'val_roc_'
            label_log = str1 + str2
            part_label_log = str3 + str2
            conf_log = str4 + str2
            roc_log = str5 + str2
            preds_prob_at_time = np.stack(((-preds_at_time_steps[i] + 1), preds_at_time_steps[i]),axis = -1)
            preds_at_time_rounded = np.round(preds_at_time_steps[i], decimals=0)
            labels_at_time = labels_at_time_steps[i]
            # self.log(label_log, frame_pred_accuracy[i])
            # self.log(part_label_log, frame_part_pred_accuracy[i])
            self.logger.experiment.log({conf_log : wandb.plot.confusion_matrix(
                        probs=None,
                        y_true=labels_at_time,
                        preds=preds_at_time_rounded,
                        class_names=["stop", "go"])
            })
            self.logger.experiment.log({roc_log : wandb.plot.roc_curve(
                            labels_at_time,
                            preds_prob_at_time,
                            labels=["stop", "go"])
            })

    # def test_step(self, batch, batch_idx):
    #     x, y = batch
    #     y_hat = self.backbone(x)
    #     loss = F.cross_entropy(y_hat, y)
    #     self.log('test_loss', loss)

    def configure_optimizers(self):
        # self.hparams available because we called self.save_hyperparameters()
        return torch.optim.Adam(self.parameters(), lr=self.args.lr)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--learning_rate', type=float, default=0.0001)
        return parser


def cli_main():
    pl.seed_everything(1234)

    # ------------
    # args
    # ------------
    args = utils.arguments.train_args()

    # parser = ArgumentParser()
    # parser.add_argument('--batch_size', default=32, type=int)
    # parser.add_argument('--hidden_dim', type=int, default=128)
    # parser = pl.Trainer.add_argparse_args(parser)
    # parser = LitClassifier.add_model_specific_args(parser)
    # args = parser.parse_args()

    # ------------
    # data
    # ------------
    dm = RawDataModule(train_args=args)

    # ------------
    # model
    # ------------
    ### EXAMPLE checkpoint loader
    # # if you train and save the model like this it will use these values when loading
    # # the weights. But you can overwrite this
    # LitModel(in_dim=32, out_dim=10)

    # # uses in_dim=32, out_dim=10
    # model = LitModel.load_from_checkpoint(PATH)

    # # uses in_dim=128, out_dim=10
    # model = LitModel.load_from_checkpoint(PATH, in_dim=128, out_dim=10)
    ###
    # some hyperparameters are inspired from BERT: https://github.com/google-research/bert
    # model parameters
    cnn_feat_dim = (args.img_width)/8 # with modified ResNet18 layer 3 output, this is feature vector max dim
    patch_size = args.vit_patch_size
    embed_dim = 512
    attn_heads = embed_dim/64
    mlp_hid_dim = embed_dim*4 # following BERT recommended architecture
    head_dropout = args.head_dropout
    if args.include_curr_frame:
        video_len = args.back_context + 1
    else:
        video_len = args.back_context
    forward_labels = args.forward_context

    # Keep hyperparameters in ViT (ViT-base)
    # Figure out purpose of hyperparameters in terms of performance changes
    # Check NNNL Loss vs CrossEntropyLoss
    # init transformer head
    transformer = ViT(
        image_size = int(cnn_feat_dim),
        patch_size = patch_size, # in spatial dimension
        num_classes = forward_labels,
        dim = embed_dim,
        depth = args.vit_depth,
        heads = int(attn_heads),
        mlp_dim = mlp_hid_dim,
        channels = 256,
        video_len = video_len,
        dropout = 0.1, # for smaller datasets, dropout improves training fit
        emb_dropout = 0.1,
        train = True,
        num_rep = 1024, # number of hidden units for pretraining MLP classification head
        head_dropout = head_dropout
    )
    # init backbone
    backbone = CRW(args, vis=None)
    if os.path.isfile(args.resume):
        print('==> Resuming from checkpoint..')
        checkpoint = torch.load(args.resume)
        
        if args.model_type == 'scratch':
            state = {}
            for k,v in checkpoint['model'].items():
                if 'conv1.1.weight' in k or 'conv2.1.weight' in k:
                    state[k.replace('.1.weight', '.weight')] = v
                else:
                    state[k] = v
            utils.partial_load(state, backbone, skip_keys=['layer4','fc'])
        else:
            utils.partial_load(checkpoint['model'], backbone, skip_keys=['layer4','fc'])

        del checkpoint
    backbone.eval()
    # model = LitClassifier(Backbone(hidden_dim=args.hidden_dim), args.learning_rate)
    model = CollTransformer(args, transformer, backbone)#.to(torch.device("cuda"))
    print(model)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath='/content/gdrive/My Drive/KITTI/coll_ckpt',
        filename='c-{epoch}-{valid_loss:.2f}',
        # every_n_val_epochs=1,
        # monitor='valid_loss', 
        # mode='min', 
        save_top_k=-1
    )
   
    # wandb.init()
    # model = CollTransformer.load_from_checkpoint('/content/gdrive/My Drive/KITTI/coll_ckpt/epoch=2-valid_loss=0.07.ckpt')
    wandb_logger = WandbLogger(
        project='coll-predict',
        log_model=True,
        entity='coll-pred',
        # id='debug-13'
    )
    # os.environ["WANDB_RESUME"] = "must"
    # os.environ["WANDB_RUN_ID"] = "10wcd9r9"
    # summary(model, input_size=(3,8,192,640))
    # ------------
    # training
    # ------------
    # trainer = pl.Trainer.from_argparse_args(args)
    # trainer = pl.Trainer(fast_dev_run=True, gpus=1)
    trainer = pl.Trainer(
        logger=wandb_logger,
        log_every_n_steps=5,
        gpus=1,
        deterministic=True, # make random gen reproducible
        benchmark=True, # cudnn
        max_epochs=20,
        callbacks=[checkpoint_callback],
        precision=32,
        accumulate_grad_batches=8,
        gradient_clip_val=1.0,
        stochastic_weight_avg=args.swa,
        # resume_from_checkpoint='/content/gdrive/My Drive/KITTI/coll_ckpt/cc-dropout-epoch=13-valid_loss=0.03.ckpt',
        # val_check_interval=0.5
        # overfit_batches=16,
        # limit_train_batches=100,
        profiler="simple",
        # limit_val_batches=100
        # fast_dev_run=10
    )
    pretrained_model = CollTransformer.load_from_checkpoint('/content/gdrive/My Drive/KITTI/coll_ckpt/b-forward_50-epoch=7-valid_loss=0.04.ckpt')
    trainer.test(model=pretrained_model, datamodule=dm)
    # trainer.fit(model, dm)
    stats = torch.cuda.memory_stats()
    peak_bytes_requirement = stats["allocated_bytes.all.peak"]
    print(f"Peak memory requirement: {peak_bytes_requirement / 1024 ** 3:.2f} GB")
    # ------------
    # testing
    # ------------
    # result = trainer.test(test_dataloaders=test_loader)
    # print(result)


if __name__ == '__main__':
    cli_main()
