from __future__ import print_function
import argparse
import os
import random
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from dataset_loader.modelnet import ModelNetDataLoader
from dataset_loader.shapenet_part import PartNormalDataset
from models.pointnet import PointNetCls, feature_transform_regularizer
from models.pointnet2 import PointNet2Cls
from models.pointcnn import PointCNN_cls
from models.dgcnn import DGCNN


def set_random_seed(seed=2048):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def set_deterministic():
    """Switch pytorch into a deterministic computation mode."""
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train(model, trainset, testset, optimizer, scheduler, device, feature_transform=False):

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch, shuffle=True, num_workers=4)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch, shuffle=True, num_workers=4)

    model.to(device)
    # train
    print('start training-------------------------------------')
    start_epoch = 0 
    best_ACC = 0
    for epoch in range(start_epoch, args.epoch):
        print("epoch: {}".format(epoch))
        # Training
        train_correct = 0
        total_trainset = 0
        total_loss = 0
        model.train()
        for i, (points, targets) in enumerate(trainloader):
            if args.model != 'pointcnn':
                points = points.transpose(2, 1)
            points, targets = points.to(device), targets.to(device)
            optimizer.zero_grad()
            pred, _, trans_feat = model(points)
            loss = F.nll_loss(pred, targets.long())
            if args.model == 'pointnet' and feature_transform:
                loss += feature_transform_regularizer(trans_feat) * 0.001
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(targets).cpu().sum()
            train_correct += correct.item()
            total_trainset += points.size()[0]
        train_accuracy = train_correct / float(total_trainset)
        scheduler.step()

        # Test accuracy on test samples
        total_correct = 0
        total_testset = 0
        model.eval()
        with torch.no_grad():
            for i, (points, targets) in enumerate(testloader):
                if args.model != 'pointcnn':
                    points = points.transpose(2, 1)
                points, targets = points.to(device), targets.to(device)
                pred, _, trans_feat = model(points)
                pred_choice = pred.data.max(1)[1]
                correct = pred_choice.eq(targets).cpu().sum()
                total_correct += correct.item()
                total_testset += points.size()[0]
        test_accuracy = total_correct / float(total_testset)

        print("training epoch ", epoch, ", train ACC = ", train_accuracy, " , test ACC = ", test_accuracy, ", loss = ", total_loss/args.batch)
        
        if test_accuracy > best_ACC:
            torch.save(model.state_dict(), os.path.join(args.savedir, args.model+'_'+args.dataset+'.pth'))
            best_ACC = test_accuracy


def get_model(model_name, num_classes):
    
    if model_name == 'pointnet':
        model = PointNetCls(k=num_classes)
    elif model_name == 'pointnet2':
        model = PointNet2Cls(num_class=num_classes)
    elif model_name == 'dgcnn':
        model = DGCNN(output_channels=num_classes)
    elif model_name == 'pointcnn':
        model = PointCNN_cls(num_class=num_classes)
    else:
        raise NotImplementedError("model does not exist")
    return model


def prepare_dataset(args):

    if args.dataset == 'modelnet40':
        num_classes = 40
        trainset = ModelNetDataLoader(root='dataset/modelnet40_normal_resampled/', args=args, split='train')
        testset = ModelNetDataLoader(root='dataset/modelnet40_normal_resampled/', args=args, split='test')
    elif args.dataset == 'shapenetpart':
        num_classes = 16
        trainset = PartNormalDataset(root='dataset/shapenetcore_partanno_segmentation_benchmark_v0_normal/', split='train')
        testset = PartNormalDataset(root='dataset/shapenetcore_partanno_segmentation_benchmark_v0_normal/', split='test')
    else:
        raise NotImplementedError("dataset not implemented")
    
    return trainset, testset, num_classes


def prepare_train_settings(args, model):

    if args.model == 'pointnet' or args.model == 'pointnet2':
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
    elif args.model == 'dgcnn':
        optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
        scheduler = CosineAnnealingLR(optimizer, args.epoch, eta_min=0.001)
    elif args.model == 'pointcnn':
        optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
        scheduler = scheduler = StepLR(optimizer, step_size=20, gamma=0.5)

    return optimizer, scheduler


def prepare_settings(args):

    trainset, testset, num_classes = prepare_dataset(args)
    model = get_model(args.model, num_classes)
    optimizer, scheduler = prepare_train_settings(args)

    return model, trainset, testset, optimizer, scheduler


def main(args):
    ##
    ## seed 2048
    set_random_seed()
    set_deterministic()
    
    model, trainset, testset, optimizer, scheduler = prepare_settings(args)

    train(model, trainset, testset, device=args.device, epoch_num=args.epoch, feature_transform=False, saved=True)
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='cuda:0', help='specify gpu device')
    parser.add_argument('--model', default='pointnet', help='model name [default: pointnet]')
    parser.add_argument('--dataset', default='modelnet40', type=str, help='name of dataset [modelnet40 or shapenetpart]')
    parser.add_argument('--epoch', default=200, type=int, help='num of epoches')
    parser.add_argument('--batch', default=24, type=int, help='size of one batch')
    parser.add_argument('--num_points', type=int, default=1024, help='Point Number')
    parser.add_argument('--feature_transform', action='store_true', help='use feature tranform in pointNet')
    parser.add_argument('--savedir', default='pretrain', help='save directory of trained model')
    parser.add_argument('--seed', type=int, default=2048, help='seed')
    args = parser.parse_args()

    main(args)
    
    
