import json
import os
import time

import torch
import torchvision.datasets
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import transforms

from Module import xNetCIFAR


class xCIFAR10:
    def __init__(self):
        self.config = None
        self.log_writer = None
        self.device_type = None

        self.cifar_loader_train = None
        self.cifar_loader_test = None

        self.net_cifar = None
        self.criterion = None
        self.optimizer = None

        self.tarin_step = 0

    def load_config(self):
        with open("./config.json", encoding="utf-8") as data:
            self.config = json.load(data)
        if self.config is None:
            assert Exception("配置文件错误或不存在")
        print("[INFO] [配置文件加载成功:{}]".format(self.config))

    def load_dataset(self):
        print(self.config['dataset_path'])
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomGrayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        cifar_data_train = torchvision.datasets.CIFAR10(self.config['dataset_path'],
                                                        train=True,
                                                        download=True,
                                                        transform=transform)

        cifar_data_test = torchvision.datasets.CIFAR10(self.config['dataset_path'],
                                                       train=False,
                                                       download=True,
                                                       transform=transform)

        self.cifar_loader_train = DataLoader(dataset=cifar_data_train, batch_size=self.config['batch_size'])
        self.cifar_loader_test = DataLoader(dataset=cifar_data_test, batch_size=self.config['batch_size'])

    def init_log_writer(self, date_str):
        log_path = "{}{}/".format(self.config['log_dir'], date_str)
        if os.path.exists(log_path) is False:
            os.makedirs(log_path)
        self.log_writer = SummaryWriter(log_dir=log_path, max_queue=256, flush_secs=5)
        print("[INFO] [日志保存目录:{}]".format(log_path))

    def init_check_checkpoint_path(self, date_str):
        checkpoint_path = "{}{}/".format(self.config['checkpoint_path'], date_str)
        if os.path.exists(checkpoint_path) is False:
            os.makedirs(checkpoint_path)
        self.config['checkpoint_path'] = checkpoint_path
        print("[INFO] [checkpoint 保存目录:{}]".format(checkpoint_path))

    def init_all(self):
        date_str = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
        self.load_config()
        self.load_dataset()
        self.init_log_writer(date_str)
        self.init_check_checkpoint_path(date_str)

        self.device_type = torch.device(self.config['device'])
        self.net_cifar = xNetCIFAR(self.config['class_num']).to(device=self.device_type)
        self.criterion = nn.CrossEntropyLoss().to(device=self.device_type)
        self.optimizer = torch.optim.Adam(self.net_cifar.parameters(), lr=self.config['lr'])
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.1, last_epoch=-1)

    def train_function(self, epoch_step):
        print("[INFO]--------第{}轮训练开始-----------".format(epoch_step))
        for images, targets in self.cifar_loader_train:
            images = images.to(device=self.device_type)
            targets = targets.to(device=self.device_type)

            output = self.net_cifar(images)
            train_loss = self.criterion(output, targets)
            self.optimizer.zero_grad()
            train_loss.backward()
            self.optimizer.step()

            self.tarin_step += 1

            self.config

            if self.tarin_step % 20 == 0:
                print("[INFO] [训练次数:{}-{}] [训练loss:{:.4f}]".format(epoch_step, self.tarin_step, train_loss.item()))
                self.log_writer.add_scalar("train_loss".format(epoch_step), train_loss.item(), self.tarin_step)
        self.scheduler.step()

    def test_function(self, epoch_step):
        # test
        correct = 0
        total_test_loss = 0.0
        total_test = 0
        with torch.no_grad():
            for images, targets in self.cifar_loader_test:
                images = images.to(device=self.device_type)
                targets = targets.to(device=self.device_type)

                output = self.net_cifar(images)

                test_loss = self.criterion(output, targets)
                total_test_loss += test_loss.item()
                _, predicted = torch.max(output.data, 1)
                total_test += targets.size(0)
                correct += (predicted == targets).sum().item()
            avg_test_loss = total_test_loss / len(self.cifar_loader_test)
            print("[INFO] [测试loss:{}-{:.4f}] [total {} images: {:.4f}%]".format(epoch_step,
                                                                                avg_test_loss,
                                                                                total_test,
                                                                                100 * correct / total_test))
            self.log_writer.add_scalar("test_loss", avg_test_loss, epoch_step)
            self.log_writer.add_scalar("accuracy", correct / total_test, epoch_step)

    def run_epoch(self):
        self.init_all()
        for epoch in range(self.config['epoch']):
            self.train_function(epoch + 1)
            self.test_function(epoch + 1)
            if (epoch + 1) % int(self.config['checkpoint_svae']) == 0:
                torch.save(self.net_cifar, "{}x_net_{}.pth".format(self.config['checkpoint_path'], epoch + 1))
        print("[INFO] [训练结束]")

    def __del__(self):
        self.log_writer.close()


if __name__ == "__main__":
    work = xCIFAR10()
    work.run_epoch()
    del work
