import os
import torch
import torch.nn as nn 
from tqdm import tqdm
# import wandb

class Trainer:
    def __init__(self, model, optimizer, scheduler, criterion, config, train_loader, val_loader):
        self.config = config 
        self.device = config.SYSTEM.DEVICE
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.save_path = config.PATH.SAVEDIR

        self.targets = ['gender', 'age', 'mask']
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion

        self.total_step = 0
        self.training_time = 0 
        
        
    def _train_epoch(self, epoch, data_loader):
        """
        Train one epoch
        """
        print(f'Epoch: {epoch}')
        self.model.train()

        train_loss = [0.0] * len(self.targets)
        correct = [0.0] * len(self.targets)
        total = 0

        for batch_idx, (inputs, targets) in enumerate(tqdm(data_loader)):
            inputs = inputs.float().to(self.device)
            total += inputs.size(0)
            targets = [target.to(self.device) for target in targets]
            self.optimizer.zero_grad()
            gender, age, mask = self.model(inputs)
            gender_loss = self.criterion(gender, targets[0])
            age_loss = self.criterion(age, targets[1])
            mask_loss = self.criterion(mask, targets[2])
            loss = gender_loss + age_loss * 1.2 + mask_loss   # age에 가중치를 줘서 학습시키면 어떨까??
            if epoch < 5:
                loss = age_loss
            loss.backward()
            self.optimizer.step()

            train_loss[0] += gender_loss.item()
            train_loss[1] += age_loss.item()
            train_loss[2] += mask_loss.item()

            _, gender_predicted = gender.max(1)
            _, age_predicted = age.max(1)
            _, mask_predicted = mask.max(1)

            

            correct[0] += gender_predicted.eq(targets[0]).sum().item()
            correct[1] += age_predicted.eq(targets[1]).sum().item()
            correct[2] += mask_predicted.eq(targets[2]).sum().item()

        print(f'[GENDER][TRAIN] Loss: {train_loss[0]/(batch_idx+1):.3f} | Acc: {100.*correct[0]/total:.3f}')
        print(f'[AGE][TRAIN] Loss: {train_loss[1]/(batch_idx+1):.3f} | Acc: {100.*correct[1]/total:.3f}')
        print(f'[MASK][TRAIN] Loss: {train_loss[2]/(batch_idx+1):.3f} | Acc: {100.*correct[2]/total:.3f}')
        # wandb.log({'train_acc': 100.*correct[1]/total, 'epoch': epoch})

    def _vaild_epoch(self, epoch):
        self.model.eval()

        val_loss = [0.0] * len(self.targets)
        correct = [0.0] * len(self.targets)
        total = 0

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.val_loader):
                inputs = inputs.to(self.device)
                total += inputs.size(0)

                targets = [target.to(self.device) for target in targets]
                gender, age, mask = self.model(inputs)
                gender_loss = self.criterion(gender, targets[0])
                age_loss = self.criterion(age, targets[1])
                mask_loss = self.criterion(mask, targets[2])
                

                val_loss[0] += gender_loss.item()
                val_loss[1] += age_loss.item()
                val_loss[2] += mask_loss.item()

                _, gender_predicted = gender.max(1)
                _, age_predicted = age.max(1)
                _, mask_predicted = mask.max(1)

                correct[0] += gender_predicted.eq(targets[0]).sum().item()
                correct[1] += age_predicted.eq(targets[1]).sum().item()
                correct[2] += mask_predicted.eq(targets[2]).sum().item()

# #create image object and log
#                 img = wandb.Image(image, boxes = 
#                                   {"predictions": 
#                                    {"box_data": predicted_boxes, 
#                                     "class_labels" : class_id_to_label},"ground_truth": {"box_data": target_boxes}})
                
#                 wandb.log({"bounding_boxes": img})                
            print(f'[GENDER][VALIDATION] Loss: {val_loss[0]/(batch_idx+1):.3f} | Acc: {100.*correct[0]/total:.3f}')
            print(f'[AGE][VALIDATION] Loss: {val_loss[1]/(batch_idx+1):.3f} | Acc: {100.*correct[1]/total:.3f}')
            print(f'[MASK][VALIDATION] Loss: {val_loss[2]/(batch_idx+1):.3f} | Acc: {100.*correct[2]/total:.3f}')
            # wandb.log({'val_acc': 100.*correct[1]/total, 'epoch': epoch})

            self.scheduler.step(float(val_loss[1]))

    def _save_checkpoint(self, epoch):
        print(f'Saveing checkpoint {self.save_path}..')
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)
        file_name = f'Efficient_{epoch}'
        file_path = os.path.join(self.save_path, file_name)
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': self.config,
            }, file_path)
        # wandb.save(file_path)

    def _resume_checkpoint(self, resume_path):
        resume_path = str(resume_path)
        checkpoint = torch.load(resume_path)


    def train(self, epochs):
        print('Trianing Start!!\n')
        for epoch in range(epochs):
            self._train_epoch(epoch, self.train_loader)
            self._vaild_epoch(epoch)
            
            if epoch == 5:
                self._train_epoch(epoch, self.val_loader)
                self._train_epoch(epoch, self.val_loader)

            if epoch % self.config.TRAIN.PERIOD == 0:
                self._save_checkpoint(epoch)

        #Clear memory
        del self.model
        del self.train_loader
        del self.val_loader
        import gc
        gc.collect()
        torch.cuda.empty_cache()