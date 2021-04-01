import os
import torch
import torch.nn as nn 
from tqdm import tqdm
import torchvision
# import wandb
from torch.utils.tensorboard import SummaryWriter
from utils import plot_classes_preds

class Trainer:
    def __init__(self, model_set_1, model_set_2 , config, train_loader, val_loader):
        self.config = config 
        self.device = config.SYSTEM.DEVICE
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.save_path = config.PATH.SAVEDIR
        self.start_epoch = 0
        self.targets = ['gender', 'age', 'mask']

        self.gender_age_model = model_set_1['model'].to(self.device)
        self.gender_age_optimizer = model_set_1['optimizer']
        self.gender_age_scheduler = model_set_1['scheduler']
        self.gender_age_criterion = model_set_1['criterion']

        self.mask_model = model_set_2['model'].to(self.device)
        self.mask_optimizer = model_set_2['optimizer']
        self.mask_scheduler = model_set_2['scheduler']
        self.mask_criterion = model_set_2['criterion']
        

        self.total_step = 0
        self.training_time = 0 
        
        if config.TRAIN.RESUME:
            self._resume_checkpoint(config.PATH.RESUME_1, config.PATH.RESUME_2)

        self.writer = SummaryWriter('runs/DenseNet_EfficientNet_1')

    def _train_epoch(self, epoch, data_loader):
        """
        Train one epoch
        """
        print(f'Epoch: {epoch}')
        self.gender_age_model.train()
        self.mask_model.train()

        train_loss = [0.0] * len(self.targets)
        correct = [0.0] * len(self.targets)
        total = 0

        for batch_idx, (inputs, targets) in enumerate(tqdm(data_loader)):
            inputs = inputs.float().to(self.device)
            total += inputs.size(0)
            targets = [target.to(self.device) for target in targets]

            self.gender_age_optimizer.zero_grad()
            self.mask_optimizer.zero_grad()
            
            gender_age = self.gender_age_model(inputs)
            mask = self.mask_model(inputs)

            gender_age_loss = self.gender_age_criterion(gender_age, targets[0])
            mask_loss = self.mask_criterion(mask, targets[1])
            
            gender_age_loss.backward()
            mask_loss.backward()
            self.gender_age_optimizer.step()
            self.mask_optimizer.step()

            train_loss[0] += gender_age_loss.item()
            train_loss[1] += mask_loss.item()

            _, gender_age_predicted = gender_age.max(1)
            _, mask_predicted = mask.max(1)

            correct[0] += gender_age_predicted.eq(targets[0]).sum().item()
            correct[1] += mask_predicted.eq(targets[1]).sum().item()

        print(f'[TRAIN][GENDER/AGE] Loss: {train_loss[0]/(batch_idx+1):.3f} | Acc: {100.*correct[0]/total:.3f}')
        print(f'[TRAIN][MASK] Loss: {train_loss[1]/(batch_idx+1):.3f} | Acc: {100.*correct[1]/total:.3f}\n')
        # wandb.log({'train_acc': 100.*correct[1]/total, 'epoch': epoch})
        self.writer.add_scalars('Loss/train', {'age': train_loss[0]/(batch_idx+1),
                                               'mask': train_loss[1]/(batch_idx+1)}, epoch)
        self.writer.add_scalars('Accuracy/train', {'age': 100.*correct[0]/total,
                                                   'mask': 100.*correct[1]/total}, epoch)

    def _vaild_epoch(self, epoch):
        self.gender_age_model.eval()
        self.mask_model.eval()

        val_loss = [0.0] * len(self.targets)
        correct = [0.0] * len(self.targets)
        total = 0

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.val_loader):
                inputs = inputs.to(self.device)
                total += inputs.size(0)

                targets = [target.to(self.device) for target in targets]

                gender_age = self.gender_age_model(inputs)
                mask = self.mask_model(inputs)
                gender_age_loss = self.gender_age_criterion(gender_age, targets[0])
                mask_loss = self.mask_criterion(mask, targets[1])
                
                val_loss[0] += gender_age_loss.item()
                val_loss[1] += mask_loss.item()

                _, gender_age_predicted = gender_age.max(1)
                _, mask_predicted = mask.max(1)

                correct[0] += gender_age_predicted.eq(targets[0]).sum().item()
                correct[1] += mask_predicted.eq(targets[1]).sum().item()
                
                self.writer.add_figure('predictions VS ground truths',
                                    plot_classes_preds(self.gender_age_model, inputs, targets[0]),
                                    global_step=epoch*len(self.val_loader)+batch_idx)

            self.gender_age_scheduler.step(float(val_loss[0]))
            self.mask_scheduler.step(float(val_loss[1]))

            # grid = torchvision.utils.make_grid(inputs)
            # self.writer.add_image('images', grid, 0)
# #create image object and log
#                 img = wandb.Image(image, boxes = 
#                                   {"predictions": 
#                                    {"box_data": predicted_boxes, 
#                                     "class_labels" : class_id_to_label},"ground_truth": {"box_data": target_boxes}})
                
#                 wandb.log({"bounding_boxes": img})                
            print(f'[VALIDARION][GENDER/AGE] Loss: {val_loss[0]/(batch_idx+1):.3f} | Acc: {100.*correct[0]/total:.3f}')
            print(f'[VALIDARION][MASK] Loss: {val_loss[1]/(batch_idx+1):.3f} | Acc: {100.*correct[1]/total:.3f}')
            # wandb.log({'val_acc': 100.*correct[1]/total, 'epoch': epoch})
            self.writer.add_scalars('Loss/val', {'age': val_loss[0]/(batch_idx+1),
                                               'mask': val_loss[1]/(batch_idx+1)}, epoch)
            self.writer.add_scalars('Accuracy/val', {'age': 100.*correct[0]/total,
                                                   'mask': 100.*correct[1]/total}, epoch)

    def _save_checkpoint(self, epoch, model, optimizer, name):
        save_path = os.path.join(self.save_path, name+'_'+str(epoch))
        print(f'Saveing checkpoint {save_path}..')
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)
        file_name = f'{name}_{epoch}'
        file_path = os.path.join(self.save_path, file_name)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, file_path)
        # wandb.save(file_path)

    def _resume_checkpoint(self, resume_path_1, resume_path_2):
        print('Loading previous models..')
        resume_path_1 = str(resume_path_1)
        resume_path_2 = str(resume_path_2)
        checkpoint_1 = torch.load(resume_path_1)
        checkpoint_2 = torch.load(resume_path_2)

        self.start_epoch = checkpoint_1['epoch'] + 1

        self.gender_age_model.load_state_dict(checkpoint_1['model_state_dict'])
        self.mask_model.load_state_dict(checkpoint_2['model_state_dict'])
        self.gender_age_optimizer.load_state_dict(checkpoint_1['optimizer_state_dict'])
        self.mask_optimizer.load_state_dict(checkpoint_2['optimizer_state_dict'])
        print('Done!!\n')

    def train(self, epochs):
        print('Trianing Start!!\n')
        for epoch in range(self.start_epoch, epochs):
            self._train_epoch(epoch, self.train_loader)
            self._vaild_epoch(epoch)

            if epoch == 10:
                self._train_epoch(epoch, self.val_loader)
                self._train_epoch(epoch, self.val_loader)


            if epoch % self.config.TRAIN.PERIOD == 0:
                self._save_checkpoint(epoch, self.gender_age_model, self.gender_age_optimizer, name='DenseNet')
                self._save_checkpoint(epoch, self.mask_model, self.mask_optimizer, name='EfficientNet')
        
        self.writer.close()
        #Clear memory
        del self.model
        del self.train_loader
        del self.val_loader
        import gc
        gc.collect()
        torch.cuda.empty_cache()