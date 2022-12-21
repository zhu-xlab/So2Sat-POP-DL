'''
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
'''
import time
import os

import numpy as np
import torch
from sklearn import model_selection
from torch.utils.data import DataLoader
from torchvision import transforms
# import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter

from utils.constants import config_path, img_rows, img_cols, all_patches_mixed_train_part1, all_patches_mixed_test_part1
from models.classification import EO2ResNet_OSM
from utils.utils import get_fnames_labs_clf
from utils.dataset import PopulationDataset_Clf
from utils.metrics import AverageMeter, save_clf_metrics
from utils.transform import RandomRotationTransform, RandomGamma, RandomBrightness
from sklearn.metrics import balanced_accuracy_score, accuracy_score
from utils.loss import FocalLoss


class Solver(object):
    '''
    Class for Training of Models on Dataset So2Sat Pop
    ----------
    :Args:
        hparams_dict: dictionary containing the hyperparameters
        model: Model to train
        num_classes: how many predictions should be made (17 for Classification)
        loss_fct: Loss Function to use
        PytorchDataset: Pytorch Dataset Class to use
        exp_dir: Experiment Directoy - Where to save the results
        osm: (bool) if osm data should be used or not
    '''

    def __init__(self, hparams_dict, PytorchModel, num_classes, loss_fct, PytorchDataset, get_files_fct, exp_dir=None,
                 osm=True, bal_acc=None):
        super().__init__()
        self.exp_dir = exp_dir
        self.osm = osm
        self.bal_acc = bal_acc
        self.hparams_dict = hparams_dict
        self.epochs = hparams_dict['num_epochs']
        self.title = self.hparams_dict['title']
        self.title = time.strftime("%Y%m%d-%H%M%S_") + self.title
        self.lr = self.hparams_dict['learning_rate']
        lr_scheduler_factor = self.hparams_dict['lr_scheduler_factor']
        lr_scheduler_steps = self.hparams_dict['lr_scheduler_steps']
        self.hparams_dict['lr_scheduler_steps'] = torch.tensor(self.hparams_dict['lr_scheduler_steps'])
        self.batch_size_tr = self.hparams_dict['train_batch_size']
        self.batch_size_val = self.hparams_dict['val_batch_size']
        self.log_steps = self.hparams_dict['log_steps']
        self.weight_decay = self.hparams_dict['weight_decay']
        self.mode = self.hparams_dict['mode']
        self.model_scale = self.hparams_dict['model_scale']
        assert self.mode in ['train', 'test']

        if self.mode == 'train':
            data_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                RandomRotationTransform(angles=[90, 180, 270], p=0.5),
                RandomGamma(),
                RandomBrightness()
            ])
            params = {'dim': (img_rows, img_cols), 'n_classes': num_classes}

            #  manually loading
            val_size = 0.2
            self.data_dir = all_patches_mixed_train_part1
            f_names, labels = get_files_fct(self.data_dir)
            f_names_train, f_names_val, labels_train, labels_val = model_selection.train_test_split(
                 f_names, labels, test_size=val_size, random_state=42)

            self.train_dataset = PytorchDataset(f_names_train, labels_train, transform=data_transform, **params)
            self.val_dataset = PytorchDataset(f_names_val, labels_val, **params)

            self.train_loader = DataLoader(self.train_dataset,
                                           batch_size=self.batch_size_tr,
                                           shuffle=True, num_workers=2)
            self.val_loader = DataLoader(self.val_dataset,
                                         batch_size=self.batch_size_val,
                                         shuffle=False, num_workers=2)
        elif self.mode == 'test':
            raise NotImplementedError
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if osm == True:
            self.model = PytorchModel(input_channels=10, num_classes=num_classes, scale_factor=self.model_scale)
        else:
            self.model = PytorchModel(input_channels=10, num_classes=num_classes)
        self.model.to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(),
                                           lr=self.lr,
                                           weight_decay=self.weight_decay)
        self.scheduler = MultiStepLR(self.optimizer, milestones=lr_scheduler_steps, gamma=lr_scheduler_factor)
        self.scaler = torch.cuda.amp.GradScaler()
        self.criterion = loss_fct()
        self._reset_histories()

    def _reset_histories(self):
        """
        Resets train and val histories for the accuracy and the loss.
        """
        self.train_metric_history = []
        self.train_loss_history = []
        self.val_metric_history = []
        self.val_loss_history = []
        self.train_acc_history = []
        self.test_acc_history = []
        self.bal_acc_history = []


class Classification_Solver(Solver):
    '''
    Solver for Classification Task
    '''

    def __init__(self, hparams_dict, model=EO2ResNet_OSM, exp_dir=None, loss_fct=FocalLoss,
                 osm=True, bal_acc=None):
        super(Classification_Solver, self).__init__(hparams_dict=hparams_dict,
                                                    PytorchModel=model,
                                                    num_classes=17,
                                                    loss_fct=loss_fct,
                                                    PytorchDataset=PopulationDataset_Clf,
                                                    get_files_fct=get_fnames_labs_clf,
                                                    exp_dir=exp_dir,
                                                    osm=osm,
                                                    bal_acc=bal_acc)

        self.metric = self.criterion

    def train(self):
        best_metric_epoch = -1
        best_metric_val = 0
        self.iteration_loss = AverageMeter('iteration_loss')
        self.epoch_loss = AverageMeter('train_loss')
        self.epoch_accuracy = AverageMeter('train_accuracy')
        self.iteration_accuracy = AverageMeter('iteration_accuracy')
        self.val_epoch_loss = AverageMeter('val_loss')
        self.val_epoch_acc = AverageMeter('val_acc')
        self.train_metric = AverageMeter('train_metric')
        self.train_metric_acc = AverageMeter('train_metric_acc')
        self.val_metric = AverageMeter('val_metric')
        self.val_metric_acc = AverageMeter('val_metric_acc')
        self.test_epoch_acc = AverageMeter('test_acc')
        self.test_metric_acc = AverageMeter('test_metric_acc')

        for epoch in range(self.epochs):
            if epoch == 0:
                start_time = time.time()
            self.iteration_loss.reset()
            self.epoch_loss.reset()
            self.epoch_accuracy.reset()
            self.iteration_accuracy.reset()
            self.val_epoch_loss.reset()
            self.train_metric.reset()
            self.train_metric_acc.reset()
            self.val_metric.reset()
            self.test_epoch_acc.reset()

            writer = SummaryWriter(os.path.join(self.exp_dir, 'log', self.title))
            for i, data in enumerate(self.train_loader):
                self.model.train()
                self.step(data)
                if i % self.log_steps == self.log_steps - 1:  # print every x mini-batches
                    self.iteration_loss.reset()
                    self.iteration_accuracy.reset()
            self.train_loss_history.append(self.epoch_loss.avg)
            self.train_metric_history.append(self.train_metric.avg)
            self.train_acc_history.append(self.train_metric_acc.avg)

            writer.add_scalar('train_epoch_loss', self.epoch_loss.avg, epoch + 1)
            writer.add_scalar('train_epoch_acc', self.epoch_accuracy.avg, epoch + 1)
            writer.add_scalar('train_epoch_metric', self.train_metric.avg, epoch + 1)
            writer.add_scalar('train_epoch_metric_acc', self.train_metric_acc.avg, epoch + 1)
            print(f'epoch {epoch + 1}, Training Loss {self.epoch_loss.avg}, Accuracy: {self.epoch_accuracy.avg}')
            ## validation
            for j, val_data in enumerate(self.val_loader):
                self.model.eval()
                self.val_step(val_data)
            self.val_loss_history.append(self.val_epoch_loss.avg)
            self.val_metric_history.append(self.val_metric.avg)
            if self.val_metric_acc.avg > best_metric_val:
                best_metric_val = self.val_metric_acc.avg
                best_metric_epoch = epoch + 1
                self.save(best_metric_val, best_metric_epoch)
                print('saved model with new best Accuracy: {}'.format(best_metric_val))
            print(f'epoch {epoch + 1}, Validation Loss {self.val_epoch_loss.avg}, Accuracy: {self.val_epoch_acc.avg}')
            writer.add_scalar('val_epoch_loss', self.val_epoch_loss.avg, epoch + 1)
            writer.add_scalar('val_epoch_acc', self.val_epoch_acc.avg, epoch + 1)
            # computing on test data set
            accuracy, balanced_accuracy = evaluate_epoch(self.model, self.title, self.exp_dir, self.osm)
            self.test_epoch_acc.update(accuracy)
            self.test_acc_history.append(self.test_epoch_acc.avg)
            self.bal_acc_history.append(balanced_accuracy)

            writer.add_scalar('test_epoch_acc', self.test_epoch_acc.avg, epoch + 1)
            print(f'epoch {epoch + 1}, Test Accuracy: {accuracy}, Bal acc: {balanced_accuracy}')
            if epoch == 0:
                end_time = time.time()
                print('1 epoch takes ', end_time - start_time, 'seconds to finish.')
            self.scheduler.step()
        self.save(best_metric_val, epoch + 1, path_addon='continuity')
        print(f'Finished Training. Best Accuracy: {best_metric_val} on val at epoch: {best_metric_epoch}')
        writer.add_hparams(self.hparams_dict, {'best Accuracy on Validation Set': best_metric_val})
        writer.close()
        evaluate(self.model, self.title, self.exp_dir, self.osm)
        return

    def step(self, data):
        # zero the parameter gradients
        self.optimizer.zero_grad()
        correct = 0
        total = 0
        if self.osm:
            inputs, targets, osm = data['input'].to(self.device), data['label'].to(self.device), data['osm'].to(
                self.device)
            preds = self.model(inputs, osm)
        else:
            inputs, targets = data['input'].to(self.device), data['label'].to(self.device)
            preds = self.model(inputs)
        # preds = preds.view(-1)
        labels = targets.argmax(1)
        predicted = preds.argmax(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
        if self.bal_acc:
            labels_det = labels.cpu().numpy()
            predicted_det = predicted.cpu().numpy()
            accu = 100. * balanced_accuracy_score(labels_det, predicted_det)
        else:
            accu = 100. * correct / total
        with torch.cuda.amp.autocast():
            loss = self.criterion(preds, targets)
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        self.iteration_loss.update(loss.item())
        self.iteration_accuracy.update(accu)
        self.epoch_loss.update(loss.item())
        self.epoch_accuracy.update(accu)
        self.train_metric.update(self.metric(preds, targets).item())

    def val_step(self, val_data):
        correct = 0
        total = 0
        with torch.no_grad():
            if self.osm:
                inputs, targets, osm = val_data['input'].to(self.device), val_data['label'].to(self.device), val_data[
                    'osm'].to(self.device)
                preds = self.model(inputs, osm)
            else:
                inputs, targets = val_data['input'].to(self.device), val_data['label'].to(self.device)
                preds = self.model(inputs)
            # preds = preds.view(-1)
            labels = targets.argmax(1)
            predicted = preds.argmax(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
            if self.bal_acc:
                labels_det = labels.cpu().numpy()
                predicted_det = predicted.cpu().numpy()
                accu = 100. * balanced_accuracy_score(labels_det, predicted_det)
            else:
                accu = 100. * correct / total
            val_loss = self.criterion(preds, targets)
            self.val_epoch_loss.update(val_loss.item())
            self.val_epoch_acc.update(accu)
            self.val_metric.update(self.metric(preds, targets).item())
            self.val_metric_acc.update(accu)

    def save(self, best_metric, epoch, path_addon=None):
        state = {'best_metric': best_metric,
                 'epoch': epoch,
                 'model_state_dict': self.model.state_dict(),
                 'hyperparams': self.hparams_dict,
                 'optimizer_state_dict': self.optimizer.state_dict(),
                 'train_loss_history': self.train_loss_history,
                 'train_metric_history': self.train_metric_history,
                 'val_loss_history': self.val_loss_history,
                 'val_metric_history': self.val_metric_history,
                 'scheduler_state_dict': self.scheduler.state_dict()
                 }
        if path_addon == None:
            name = self.title + '_model'
        else:
            name = self.title + '_model' + '_' + path_addon
        model_folder = os.path.join(self.exp_dir, 'models')
        if not os.path.exists(model_folder):
            os.mkdir((model_folder))
        model_path = os.path.join(model_folder, name + '.pth')
        torch.save(state, model_path)


def evaluate(model, model_name, exp_dir, osm_set):
    params = {'dim': (img_rows, img_cols), 'n_classes': 17}
    batch_size = 32
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    title = model_name
    map_location = "cuda:0" if torch.cuda.is_available() else "cpu"
    model_dict = torch.load(os.path.join(exp_dir, 'models', model_name + '_model.pth'),
                            map_location=torch.device(map_location))
    model.to(device)
    model.load_state_dict(model_dict['model_state_dict'])
    ## manually loading
    data_dir = all_patches_mixed_test_part1
    f_names_test, labels_test = get_fnames_labs_clf(data_dir)
    np.save(os.path.join(config_path, 'f_lists', 'f_names_test.npy'), f_names_test)
    np.save(os.path.join(config_path, 'f_lists', 'labels_test.npy'), labels_test)
    f_names_test = np.load(os.path.join(config_path, 'f_lists', 'f_names_test.npy'))
    labels_test = np.load(os.path.join(config_path, 'f_lists', 'labels_test.npy'))
    test_dataset = PopulationDataset_Clf(f_names_test, labels_test, test=True, **params)
    test_loader = DataLoader(test_dataset,
                             batch_size=batch_size,
                             shuffle=False, num_workers=2)

    all_preds = torch.zeros((len(test_loader.dataset)))
    all_targets = torch.zeros((len(test_loader.dataset)))
    for i, data in enumerate(test_loader):  ## change to val or test
        model.eval()
        if osm_set == True:
            inputs, targets, osm = data['input'].to(device), \
                                   data['label'].to(device), data['osm'].to(device)
            with torch.no_grad():
                preds = model(inputs, osm)
        else:
            inputs, targets = data['input'].to(device), data['label'].to(device)
            with torch.no_grad():
                preds = model(inputs)
        # preds = preds.view(-1)
        _, predicted = preds.max(1)
        _, labels = targets.max(1)
        all_preds[i * batch_size:i * batch_size + preds.shape[0]] = predicted
        all_targets[i * batch_size:i * batch_size + preds.shape[0]] = labels
    save_clf_metrics(all_targets, all_preds, os.path.join(exp_dir, 'log', title), dataset='test')
    return all_targets, all_preds


def evaluate_epoch(model, model_name, exp_dir, osm_set):
    params = {'dim': (img_rows, img_cols), 'n_classes': 17}
    batch_size = 1
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    title = model_name
    map_location = "cuda:0" if torch.cuda.is_available() else "cpu"
    model_dict = torch.load(os.path.join(exp_dir, 'models', model_name + '_model.pth'),
                            map_location=torch.device(map_location))
    model.to(device)
    model.load_state_dict(model_dict['model_state_dict'])
    ## manually loading
    data_dir = all_patches_mixed_test_part1
    f_names_test, labels_test = get_fnames_labs_clf(data_dir)
    np.save(os.path.join(config_path, 'f_lists', 'f_names_test.npy'), f_names_test)
    np.save(os.path.join(config_path, 'f_lists', 'labels_test.npy'), labels_test)
    f_names_test = np.load(os.path.join(config_path, 'f_lists', 'f_names_test.npy'))
    labels_test = np.load(os.path.join(config_path, 'f_lists', 'labels_test.npy'))
    test_dataset = PopulationDataset_Clf(f_names_test, labels_test, test=True, **params)
    test_loader = DataLoader(test_dataset,
                             batch_size=batch_size,
                             shuffle=False, num_workers=2)

    all_preds = torch.zeros((len(test_loader.dataset)))
    all_targets = torch.zeros((len(test_loader.dataset)))
    for i, data in enumerate(test_loader):  ## change to val or test
        model.eval()
        if osm_set == True:
            inputs, targets, osm = data['input'].to(device), \
                                   data['label'].to(device), data['osm'].to(device)
            with torch.no_grad():
                preds = model(inputs, osm)
        else:
            inputs, targets = data['input'].to(device), data['label'].to(device)
            with torch.no_grad():
                preds = model(inputs)
        # preds = preds.view(-1)
        _, predicted = preds.max(1)
        _, labels = targets.max(1)
        all_preds[i * batch_size:i * batch_size + preds.shape[0]] = predicted
        all_targets[i * batch_size:i * batch_size + preds.shape[0]] = labels
    y_targ = all_targets.detach().numpy()
    y_pred = all_preds.detach().numpy()
    accuracy = accuracy_score(y_targ, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_targ, y_pred)

    return accuracy, balanced_accuracy

