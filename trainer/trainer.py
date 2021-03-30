import os
import torch
import torch.nn as nn 
from tqdm import tqdm

class Trainer:
    def __init__(self, model, optimizer, criterion, config, train_loader, val_loader):
        self.config = config 
        self.device = config.SYSTEM.DEVICE
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.save_path = config.SAVEDIR

        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.criterion = criterion

        self.total_step = 0
        self.training_time = 0 
    
    def _train_epoch(self, epoch):
        """
        Train one epoch
        """
        print(f'Epoch: {epoch}')
        self.model.train()

        train_loss = 0
        correct = 0
        total = 0

        for batch_idx, (inputs, targets) in tqdm(enumerate(self.train_loader)):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        print(f'[TRAIN] Loss: {train_loss/(batch_idx+1):.3f} | Acc: {100.*correct/total:.3f}')
    
    def _vaild_epoch(self, epoch):
        self.model.eval()

        val_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.val_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
            print(f'[VALIDATION] Loss: {val_loss/(batch_idx+1):.3f} | Acc: {100.*correct/total:.3f}\n')

        

    def _save_model(self, epoch):
        print(f'Save model {self.save_path}')
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)
        file_name = f'MLP_{epoch}'
        file_path = os.path.join(self.save_path, file_name)
        torch.save(self.model.state_dict(), file_path)


    def train(self):
        print('Trianing Start!!\n')
        for epoch in range(self.config.TRAIN.EPOCH):
            self._train_epoch(epoch)
            self._vaild_epoch(epoch)
            
            if epoch & self.config.TRAIN.PERIOD == 0:
                self._save_model(epoch)