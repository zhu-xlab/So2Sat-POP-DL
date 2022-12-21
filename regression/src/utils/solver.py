#  Solver File for Training

import time
import os

from sklearn import model_selection
import torch
from torch.utils.data import DataLoader
from torch import nn
from torchvision import transforms
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter

from utils.constants import img_rows, img_cols, all_patches_mixed_train_part1
from models.regression import EO2ResNet_OSM
from utils.dataset import PopulationDataset_Reg
from utils.metrics import AverageMeter
from utils.transform import RandomRotationTransform, RandomBrightness, RandomGamma
from utils.utils import get_fnames_labs_reg
from utils.evaluation import evaluate


class Solver(object):
    '''
    Class for Training of Models on Dataset So2Sat Pop
    ----------
    :Args:
        hparams_dict: dictionary containing the hyperparameters
        model: Model to train
        num_classes: how many predictions should be made (usually =1 for Regression)
        loss_fct: Loss Function to use
        PytorchDataset: Pytorch Dataset Class to use
        exp_dir: Experiment Directoy - Where to save the results
        osm: (bool) if osm data should be used or not
    '''
    def __init__(self, hparams_dict, model, num_classes, loss_fct, PytorchDataset, exp_dir = None, osm=True, continue_train=False, model_name=None):
        super().__init__()
        self.exp_dir = exp_dir
        self.osm = osm
        self.continue_train = continue_train
        if self.continue_train == False:
            self.hparams_dict = hparams_dict
        else:
            if model_name == None:
                raise AssertionError('If Training should be continued, model name has to be specified')
            map_location = "cuda:0" if torch.cuda.is_available() else "cpu"
            self.model_dict = torch.load(os.path.join(exp_dir, 'models', model_name+'.pth'), map_location=torch.device(map_location))
            self.hparams_dict = self.model_dict['hyperparams']
            if self.hparams_dict['num_epochs'] == hparams_dict['num_epochs']:
                raise AssertionError('If Training should be continued, higher number of epochs must be specified')
            self.hparams_dict['num_epochs'] = hparams_dict['num_epochs']
        self.epochs = self.hparams_dict['num_epochs']
        self.title = self.hparams_dict['title']
        self.title = time.strftime("%Y%m%d-%H%M%S_") + self.title
        self.lr = self.hparams_dict['learning_rate']
        self.lr_scheduler_factor = self.hparams_dict['lr_scheduler_factor']
        self.lr_scheduler_steps = self.hparams_dict['lr_scheduler_steps']
        self.hparams_dict['lr_scheduler_steps'] = torch.tensor(self.hparams_dict['lr_scheduler_steps'])
        self.batch_size_tr = self.hparams_dict['train_batch_size']
        self.batch_size_val = self.hparams_dict['val_batch_size']
        self.log_steps = self.hparams_dict['log_steps']
        self.weight_decay = self.hparams_dict['weight_decay']
        self.model_scale = self.hparams_dict['model_scale']

        ## Data Loading
        self.PytorchDataset = PytorchDataset
        self._load_data()

        ## configure pipeline
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = model
        self.num_classes = num_classes
        self._init_model()
        if self.continue_train == True:
            self.model.load_state_dict(self.model_dict['model_state_dict'])
            self.optimizer.load_state_dict(self.model_dict['optimizer_state_dict'])
            self.scheduler.load_state_dict(self.model_dict['scheduler_state_dict'])
            self._set_histories(self.model_dict)
        else:
            self._reset_histories()

        self.scaler =torch.cuda.amp.GradScaler()
        self.criterion = loss_fct()

    def _load_data(self):
        data_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            RandomRotationTransform(angles=[90, 180, 270], p=0.5),
            RandomGamma(),
            RandomBrightness()
        ])
        params = {'dim': (img_rows, img_cols)}

        val_size = 0.2
        self.data_dir = all_patches_mixed_train_part1
        f_names, labels = get_fnames_labs_reg(self.data_dir)
        f_names_train, f_names_val, labels_train, labels_val = model_selection.train_test_split(
             f_names, labels, test_size=val_size, random_state=42)

        self.train_dataset = self.PytorchDataset(f_names_train, labels_train, transform=data_transform, **params)
        self.val_dataset = self.PytorchDataset(f_names_val, labels_val, **params)

        self.train_loader = DataLoader(self.train_dataset,
                                        batch_size=self.batch_size_tr,
                                        shuffle=True,
                                        num_workers=4,
                                        pin_memory=True,
                                        drop_last=True)
        self.val_loader = DataLoader(self.val_dataset,
                                    batch_size=self.batch_size_val,
                                    shuffle=False,
                                    num_workers=4,
                                    pin_memory=True,
                                    drop_last=True)

    def _init_model(self):
        if self.osm is True:
            self.model = self.model(input_channels=10, num_classes=self.num_classes, scale_factor=self.model_scale)
        else:
            self.model = self.model(input_channels=10, num_classes=self.num_classes)
        self.model.to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(),
                                            lr=self.lr,
                                            weight_decay=self.weight_decay)
        self.scheduler = MultiStepLR(self.optimizer, milestones=self.lr_scheduler_steps, gamma=self.lr_scheduler_factor)

    def _reset_histories(self):
        """
        Resets train and val histories for the accuracy and the loss.
        """
        self.train_metric_history = []
        self.train_loss_history = []
        self.val_metric_history = []
        self.val_loss_history = []

    def _set_histories(self, history_dict):
        """
        Resets train and val histories for the accuracy and the loss.
        """
        self.train_metric_history = history_dict['train_loss_history']
        self.train_loss_history = history_dict['train_metric_history']
        self.val_metric_history = history_dict['val_loss_history']
        self.val_loss_history = history_dict['val_metric_history']


class Regression_Solver(Solver):
    '''
    Solver for Regression Task
    '''
    def __init__(self, hparams_dict, model=EO2ResNet_OSM, exp_dir=None, loss_fct=nn.MSELoss, osm=True,
                 continue_train=False, model_name=None):
        super(Regression_Solver, self).__init__(hparams_dict=hparams_dict,
                                                model=model,
                                                num_classes=1,
                                                loss_fct=loss_fct,
                                                PytorchDataset=PopulationDataset_Reg,
                                                exp_dir=exp_dir,
                                                osm=osm,
                                                continue_train=continue_train,
                                                model_name=model_name)

        self.metric = self.criterion

    def train(self):
        if self.continue_train is False:
            best_metric_epoch = -1
            best_metric_val = 10000000000000.0
            start_epoch = 0
        else:
            best_metric_epoch = self.model_dict['epoch']
            best_metric_val = self.model_dict['best_metric']
            start_epoch = self.model_dict['epoch']
        self.iteration_loss = AverageMeter('iteration_loss')
        self.epoch_loss = AverageMeter('train_loss')
        self.val_epoch_loss = AverageMeter('val_loss')
        self.train_metric = AverageMeter('train_metric')
        self.val_metric = AverageMeter('val_metric')
        for epoch in range(start_epoch, self.epochs):
            if epoch == 2:
                start_time = time.time()
            self.iteration_loss.reset()
            self.epoch_loss.reset()
            self.val_epoch_loss.reset()
            self.train_metric.reset()
            self.val_metric.reset()
            writer = SummaryWriter(os.path.join(self.exp_dir,'log',self.title))
            for i, data in enumerate(self.train_loader):
                self.model.train()
                self.step(data)
                if i % self.log_steps == self.log_steps-1:       # print every x mini-batches
                    print(f'{i+1}/{len(self.train_loader)}, train_loss: {self.iteration_loss.avg}'
                        f', scaled_loss: {self.iteration_loss.avg*53119.0}')
                    self.iteration_loss.reset()
            self.train_loss_history.append(self.epoch_loss.avg)
            self.train_metric_history.append(self.train_metric.avg)
            writer.add_scalar('train_epoch_loss', self.epoch_loss.avg, epoch + 1)
            writer.add_scalar('train_epoch_metric', self.train_metric.avg, epoch + 1)
            print(f'epoch {epoch + 1}, Training Loss {self.epoch_loss.avg}, MAE: {self.train_metric.avg}'
                f', scaled MAE: {self.train_metric.avg * 53119.0}')
            ## validation
            for j, val_data in enumerate(self.val_loader):
                self.model.eval()
                self.val_step(val_data)
            self.val_loss_history.append(self.val_epoch_loss.avg)
            self.val_metric_history.append(self.val_metric.avg)
            if self.val_metric.avg < best_metric_val:
                best_metric_val = self.val_metric.avg
                best_metric_epoch = epoch+1
                self.save(best_metric_val, best_metric_epoch)
                print('saved model with new best MAE: {}'.format(best_metric_val))
            print(f'epoch {epoch + 1}, Validation Loss {self.val_epoch_loss.avg}, MAE: {self.val_metric.avg}'
                f', scaled MAE: {self.val_metric.avg * 53119.0}')
            writer.add_scalar('val_epoch_loss', self.val_epoch_loss.avg, epoch + 1)
            writer.add_scalar('val_mae', self.val_metric.avg, epoch + 1)
            if epoch == 2:
                end_time = time.time()
                print('1 epoch takes ', end_time-start_time,'seconds to finish.')
            self.scheduler.step()

        self.save(best_metric_val, epoch+1, path_addon='continuity')
        print(f'Finished Training. Best MAE: {best_metric_val} on val at epoch: {best_metric_epoch}')
        writer.add_hparams(self.hparams_dict, {'best Accuracy on Validation Set': best_metric_val})
        writer.close()
        evaluate(model=EO2ResNet_OSM, model_name=self.title + '_model', exp_dir=self.exp_dir, osm_flag=self.osm)
        return best_metric_val * 53119.0

    def step(self, data):
        # zero the parameter gradients
        self.optimizer.zero_grad()
        if self.osm:
            inputs, targets, osm = data['input'].to(self.device), \
                data['label'].to(self.device), data['osm'].to(self.device)
            preds = self.model(inputs, osm)
        else:
            inputs, targets = data['input'].to(self.device), data['label'].to(self.device)
            preds = self.model(inputs)
        preds = preds.view(-1)
        with torch.cuda.amp.autocast():
            loss = self.criterion(preds, targets)
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        self.iteration_loss.update(loss.item())
        self.epoch_loss.update(loss.item())
        self.train_metric.update(self.metric(preds,targets).item())
    
    def val_step(self, val_data):
        with torch.no_grad():
            if self.osm:
                inputs, targets, osm = val_data['input'].to(self.device), \
                    val_data['label'].to(self.device), val_data['osm'].to(self.device)
                preds = self.model(inputs, osm)
            else:
                inputs, targets = val_data['input'].to(self.device), val_data['label'].to(self.device)
                preds = self.model(inputs)
            preds = preds.view(-1)
            val_loss = self.criterion(preds, targets)
            self.val_epoch_loss.update(val_loss.item())
            self.val_metric.update(self.metric(preds, targets).item())

    def save(self, best_metric, epoch, path_addon = None):
        state = {'best_metric': best_metric,
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(), 
                'hyperparams':self.hparams_dict,
                'optimizer_state_dict':self.optimizer.state_dict(),
                'train_loss_history': self.train_loss_history,
                'train_metric_history':self.train_metric_history,
                'val_loss_history': self.val_loss_history,
                'val_metric_history':self.val_metric_history,
                'scheduler_state_dict':self.scheduler.state_dict()
                } 
        if path_addon is None:
            name = self.title + '_model'
        else:
            name = self.title + '_model' + '_' + path_addon

        model_folder = os.path.join(self.exp_dir, 'models')
        if not os.path.exists(model_folder):
            os.mkdir((model_folder))
        torch.save(state, os.path.join(model_folder, name +'.pth'))
