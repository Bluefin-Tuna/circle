import argparse
import pandas as pd
import numpy as np
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import torch.optim as optim
import torch
import torch.nn as nn
from circle.regression import ConvNet
from circle.utils import iou
from circle.dataset import NoisyCircles


parser = argparse.ArgumentParser(description='')

parser.add_argument('--name',default='v1', type=str,
                    help='Name of the experiment.')
parser.add_argument('--out_file', default='new_out.txt',
                    help='Path to output features file.')
parser.add_argument('-j', '--workers', default=10, type=int, metavar='N',
                    help='Number of data loaders. Default is 10.')
parser.add_argument('-b', '--batch_size', default=256, type=int,
                    metavar='N',
                    help='Mini-Batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='Initial learning rate.', dest='lr')
parser.add_argument('--resume',
                    default='', type=str, metavar='PATH',
                    help='Path to latest checkpoint.')
parser.add_argument('--data', default='./data', metavar='DIR',
                    help='Path to directory of data binaries for training and testing.')
parser.add_argument('--print_freq', default=1, type=int,
                    metavar='N', help='Print frequency.')
parser.add_argument('--epochs', default=51, type=int, metavar='N',
                    help='Number of epochs to run for.')
parser.add_argument('--save_freq', default=5, type=int,
                    help='Number of epochs to save after.')


def main():
    args = parser.parse_args()
    print(args)

    print("=> Creating model")
    model = ConvNet()

    if args.resume:
        print("=> Loading checkpoint: " + args.resume)
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint)
        args.start_epoch = int(args.resume.split('/')[1].split('_')[0])
        print("=> Checkpoint loaded. Epoch : " + str(args.start_epoch))

    else:
        print("=> Starting from scratch ")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    model = nn.DataParallel(model)
    model.to(device)

    criteria = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), args.lr)

    cudnn.benchmark = True
    normalize = transforms.Normalize(mean=[0.5], std=[0.5])

    trainset = NoisyCircles(
        f"{args.data}/train/images.npy",
        f"{args.data}/train/labels.npy",
        transforms.Compose([
            normalize,
        ]))

    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers)

    output = open(args.out_file, "w")
    for epoch in range(args.epochs):
        adjust_learning_rate(optimizer, epoch, args)
        train(train_loader, model, criteria, optimizer, epoch, args, device, len(trainset), output)


def train(train_loader, model, criteria, optimizer, epoch, args, device, len, file):

    # switch to train mode
    model.train()
    running_loss = 0.0

    for i, (images, target) in enumerate(train_loader):

        images = images.to(device)
        target = target.to(device)

        output = model(images)

        loss = criteria(output, target/100)

        # compute gradient and do optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % args.print_freq == args.print_freq - 1 or i == int(len/args.batch_size):    # print every 50 mini-batches
            new_line = 'Epoch: [%d][%d/%d] loss: %f' % \
                       (epoch + 1, i + 1, int(len/args.batch_size) + 1, running_loss / args.print_freq)
            file.write(new_line + '\n')
            print(new_line)
            running_loss = 0.0

        if epoch % args.save_freq == 0:
            torch.save(model.module.state_dict(), f'./checkpoints/{str(epoch)}_epoch_{args.name}_checkpoint.pth.tar')


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate"""
    lr = args.lr
    if 20 < epoch <= 30:
        lr = 0.0001
    elif 30 < epoch :
        lr = 0.00001
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    print("Learning Rate -> {}\n".format(lr))


if __name__ == '__main__':
    main()