from model.model import set_parameter_requires_grad
from model.loss import LabelSmoothing
import os
import copy
from utils import convert_target, show_images
from matplotlib import pyplot as plt
import torch
import torch.nn as nn 
import torch.nn.functional as F
from tqdm import tqdm
import torchvision
# import wandb
from torch.utils.tensorboard import SummaryWriter
from utils import seed_everything
# from utils import plot_classes_preds
from dataloader.dataset import generate_cutmix_image
from torch.cuda.amp import GradScaler, autocast
#
from sklearn.metrics import f1_score

class Trainer:
    def __init__(self, model_set, config, train_loader, valid_loader, fold):
        self.config = config 
        self.device = config.SYSTEM.DEVICE
        self.save_path = config.PATH.SAVEDIR
        self.start_epoch = 0

        self.model = model_set['model'].to(self.device)
        self.optimizer = model_set['optimizer']
        self.scheduler = model_set['scheduler']
        self.criterion = model_set['criterion']
        
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.fold = fold
        self.total_step = 0
        self.training_time = 0 
        
        if config.TRAIN.RESUME:
            self._resume_checkpoint(config.PATH.RESUME)

        # self.writer = SummaryWriter('runs/ex1')

    def _train_epoch(self, epoch):
        """
        Train one epoch
        """
        print(f'Epoch: {epoch}')
        self.model.train()

        train_loss = 0
        correct = 0
        total = 0

        for batch_idx, (inputs, targets) in enumerate(tqdm(self.train_loader)):
            inputs = inputs.float().to(self.device)
            total += inputs.size(0)
            targets = [target.to(self.device) for target in targets]
            labels = targets[0]
            ages = targets[1]

            label_smoothing= LabelSmoothing(6, 0.4)
            soft_targets_1 = label_smoothing(inputs.size(0), labels)
            soft_targets_2 = convert_target(6, labels, ages)
            soft_targets = (soft_targets_1 + soft_targets_2) / 2

            if torch.rand(1) > 0.5:
            # targets = F.one_hot(targets[0], num_classes=6).float()
                inputs, soft_targets = generate_cutmix_image(inputs, soft_targets)
            # show_images(inputs)
            # print(targets)
            
            self.optimizer.zero_grad()
            with autocast():
                gender_age = self.model(inputs)

                loss = self.criterion(gender_age, soft_targets)
            
            self.scaler_1.scale(loss).backward()
            self.scaler_1.step(self.optimizer)

            self.scaler_1.update()

            train_loss += loss.item()

            _, predicted = gender_age.max(1)

            correct += predicted.eq(labels).sum().item()
        # print(f'[TRAIN][AGE] Loss: {train_loss/(batch_idx+1):.3f} | Acc: {100.*correct/total:.3f}')
        # # wandb.log({'train_acc': 100.*correct[1]/total, 'epoch': epoch})
        # self.writer.add_scalars('Loss/train', {'age': train_loss[0]/(batch_idx+1)}, epoch)
        # self.writer.add_scalars('Accuracy/train', {'age': 100.*correct[0]/total}, epoch)

    def _vaild_epoch(self, epoch):
        self.model.eval()

        val_loss = 0
        correct = 0
        total = 0
        epoch_f1 = 0

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.valid_loader):
                inputs = inputs.to(self.device)
                total += inputs.size(0)

                targets = [target.to(self.device) for target in targets]
                labels = targets[0]
                # targets = convert_target(6, targets[0], targets[1])
                targets = F.one_hot(targets[0], num_classes=6).float()

                gender_age = self.model(inputs)
                # loss = self.criterion(gender_age, targets)
                
                # val_loss += loss.item()

                _, predicted = gender_age.max(1)
                
                correct += predicted.eq(labels).sum().item()
                self.scheduler.step()
                batch_f1 = f1_score(labels.cpu(), predicted.cpu(), average='macro')
                epoch_f1 += batch_f1*inputs.size(0)
                
                # if (epoch % 2==0) and (batch_idx < 2):
                #     self.writer.add_figure('predictions VS ground truths',
                #                         plot_classes_preds(self.model, inputs, targets[0]),
                #                         global_step=epoch+batch_idx)
        print(f'[VALIDARION][AGE]  F1: {epoch_f1/total:.3f}')
        return epoch_f1/total
            # self.scheduler.step(float(correct[0]/total))
            # self.mask_scheduler.step(float(correct[1]/total))

            # grid = torchvision.utils.make_grid(inputs)
            # self.writer.add_image('images', grid, 0)
# #create image object and log
#                 img = wandb.Image(image, boxes = 
#                                   {"predictions": 
#                                    {"box_data": predicted_boxes, 
#                                     "class_labels" : class_id_to_label},"ground_truth": {"box_data": target_boxes}})
                
# #                 wandb.log({"bounding_boxes": img})                
#             # wandb.log({'val_acc': 100.*correct[1]/total, 'epoch': epoch})
#             self.writer.add_scalars('Loss/val', {'age': val_loss[0]/(batch_idx+1)}, epoch)
#             self.writer.add_scalars('Accuracy/val', {'age': 100.*correct[0]/total}, epoch)


    def train(self, epochs):
        seed_everything(42)
        self.scaler_1 = GradScaler()
        
        best_f1 = 0.0
        for epoch in range(epochs):
            if epoch == 5:
                set_parameter_requires_grad(self.model.model, True)
            self._train_epoch(epoch)
            epoch_f1 = self._vaild_epoch(epoch)
            if epoch_f1 >= best_f1:
                best_f1 = epoch_f1
                best_model_wts = copy.deepcopy(self.model.state_dict())
        save_name = f'F1_Fold{self.fold}_ef_final.pth'
        
        save_path = os.path.join(self.save_path, save_name)
        torch.save(best_model_wts, save_path) 

        #Clear memory
        del self.model
        del self.train_loader
        del self.valid_loader
        import gc
        gc.collect()
        torch.cuda.empty_cache()

        return best_f1
        # if epoch % self.config.TRAIN.PERIOD == 0:
            # self._save_checkpoint(epoch, self.model, self.optimizer, name='DenseNet')
        