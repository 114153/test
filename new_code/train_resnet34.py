from __future__ import print_function
import os
import yaml
import time
import random
import shutil
# import argparse
import numpy as np
import pandas as pd
# # # os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
# from sklearn.model_selection import KFold

from sklearn.metrics import f1_score

from libs.utils import LiverDataset
# # from libs.utils import get_augumentor
from libs.utils import accuracy
from libs.utils import AverageMeter
from libs.utils import adjust_learning_rate
# # from libs.utils import save_checkpoint
# # from libs.utils import balance_accuracy

from libs.set_models import *
# # from tensorboardX import SummaryWriter
# # from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

# set random seed
random.seed(666)
np.random.seed(666)
torch.manual_seed(666)
torch.cuda.manual_seed_all(666)


def start(cfg):

    BASE_PATH = cfg['dataset']['data']

    # TRAIN1_LABEL_PATH = os.path.join(BASE_PATH, "train_label.csv")
    # TRAIN2_LABEL_PATH = os.path.join(BASE_PATH, "train2_label.csv")
    # train1_df = pd.read_csv(TRAIN1_LABEL_PATH)
    # train2_df = pd.read_csv(TRAIN2_LABEL_PATH)
    # train1_df['id'] = BASE_PATH + "/train_dataset/" + train1_df['id']
    # train2_df['id'] = BASE_PATH + "/train_dataset/" + train2_df['id']
    # train1_df['suffix'] = '.npy'
    # train2_df['suffix'] = '.npy'
    # train_df = train1_df.append(train2_df).reset_index(drop=True)


    TRAIN_LABEL_PATH = os.path.join(BASE_PATH, "train_choose_label.csv")
    val_LABEL_PATH = os.path.join(BASE_PATH, "val_choose_label.csv")
    train_df = pd.read_csv(TRAIN_LABEL_PATH)
    val_df = pd.read_csv(val_LABEL_PATH)
    train_df['id'] = BASE_PATH + "/train_dataset/" + train_df['id']
    val_df['id'] = BASE_PATH + "/train_dataset/" + val_df['id']
    train_df['suffix'] = '.npy'
    val_df['suffix'] = '.npy'


    print(len(train_df), len(val_df))

    # target = train_df['ret'].tolist()
    # # print(len(target))

    # folds = KFold(cfg['training']['nfold'],shuffle=True,random_state=114153)
    # # folds = MultilabelStratifiedKFold(cfg['training']['nfold'],shuffle=True,random_state=66666)

    # for n_fold, (tr_idx, val_idx) in enumerate(folds.split(train_df['id'], target)):
        
    #     # print(val_idx)

    #     print("###################################################################################")
    #     print("fold:", n_fold)
    #     # print("tr_idx:", tr_idx)
    #     # print("val_idx:", val_idx)
    #     tr_data = train_df.iloc[tr_idx]
    #     val_data = train_df.iloc[val_idx]
    #     # print(val_data.head())
    #     print(len(tr_data), len(val_data))

    #     main(cfg, n_fold,tr_data,val_data)

    main(cfg, 0, train_df, val_df)



def main(cfg, fold, tr_data, val_data):

    # global cfg  # , writer

    # images_id = val_df['id'].values
    # suffix = val_df['suffix'].values

    # a = np.load(images_id[0] + suffix[0])
    # print(a.shape)
    # print(val_df.head())

    # if cfg['model']['pretrained']:
    #     print("=> using pre-trained model and fintune '{}'".format(cfg['model']['arch']))
    #     model = models.__dict__[cfg['model']['arch']](pretrained='imagenet')
    # else:
    #     print("=> creating model '{}'".format(cfg['model']['arch']))
    #     model = models.__dict__[cfg['model']['arch']](pretrained=None)

    # model.last_linear = nn.Linear(model.last_linear.in_features, cfg['training']['nb_classes'])

    model, opt = get_model_opt(cfg)
    # print(model)


    if cfg['model']['arch'].startswith('alexnet') or cfg['model']['arch'].startswith('vgg'):
        model._features = torch.nn.DataParallel(model._features)
        model.cuda()
    else:
        model = torch.nn.DataParallel(model).cuda()

    cudnn.benchmark = True


    # define loss function and optimizer
    criterion = nn.CrossEntropyLoss()

    # opt = torch.optim.SGD(model.parameters(), cfg['training']['optimizer']['lr'],
    #                             momentum=cfg['training']['optimizer']['momentum'],
    #                             weight_decay=cfg['training']['optimizer']['weight_decay'],
    #                             nesterov=cfg['training']['optimizer']['nesterov'])
    # print("Using optimizer {}".format(opt))

    # # resume
    # if cfg['checkpoint']['resume']:
    #     if os.path.isfile(cfg['checkpoint']['resume']):
    #         print("=> loading checkpoint '{}'".format(cfg['checkpoint']['resume']))
    #         checkpoint = torch.load(cfg['checkpoint']['resume'])
    #         cfg['training']['start_epoch'] = checkpoint['epoch']
    #         best_Macro_F1 = checkpoint['best_Macro_F1']
    #         model.load_state_dict(checkpoint['state_dict'])
    #         optimizer.load_state_dict(checkpoint['optimizer'])
    #         print("=> loaded checkpoint '{}' (epoch {})"
    #               .format(cfg['checkpoint']['resume'], checkpoint['epoch']))
    #     else:
    #         print("=> no checkpoint found at '{}'".format(cfg['checkpoint']['resume']))
    # else:
    #     print("=> no using checkpoint")

    

    # Data loading code
    # traindir = os.path.join(cfg['dataset']['data'], 'train_dataset')
    # valdir = os.path.join(cfg['dataset']['data'], 'val_dataset')

    # Data augmentations
    # normalize = transforms.Normalize(mean=[0.2012, 0.2073, 0.2122, 0.2156, 0.2187, 0.2159, 0.2131, 0.2100, 0.2068],
    #                                  std=[0.2495, 0.2509, 0.2521, 0.2530, 0.2546, 0.2530, 0.2517, 0.2505, 0.2494])


    train_transforms = transforms.Compose([
        # transforms.ColorJitter(brightness=0, contrast=0.2, saturation=0.5, hue=0.5),
        # transforms.RandomSharp(0.3),
        # transforms.RandomCrop(1500),
        # transforms.RandomResizedCrop(299),
        # transforms.RandomFixedRotate([0, 90, 180, 270]),
        # transforms.ToTensor(),
        # transforms.ToPILImage(),
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.2209961, 0.22721613, 0.23213791, 0.23545924, 0.23665502, 0.23613228, 0.23374401, 0.23078922, 0.22764728],
                              std=[0.27426004, 0.27456304, 0.27475687, 0.27484286, 0.27380459, 0.27403751, 0.2726981,  0.27149808, 0.27040896])
        # normalize,
        ])

    test_transforms = transforms.Compose([
            # transforms.CenterCrop(1500),
            # transforms.RandomResizedCrop(299),
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.2209961, 0.22721613, 0.23213791, 0.23545924, 0.23665502, 0.23613228, 0.23374401, 0.23078922, 0.22764728],
                                  std=[0.27426004, 0.27456304, 0.27475687, 0.27484286, 0.27380459, 0.27403751, 0.2726981,  0.27149808, 0.27040896])
            # normalize,
        ])

    train_dataset = LiverDataset(tr_data, 'train', train_transforms)
    val_dataset = LiverDataset(val_data, 'val', test_transforms)

    print(cfg)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg['training']['batch_size'], shuffle=True,
        num_workers=cfg['training']['workers'], pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=32, shuffle=False,
        num_workers=cfg['training']['workers'], pin_memory=True)

    # for i, (input, target) in enumerate(train_loader):
    #     print(input.shape)
    #     print(target)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg['training']['cos_length']*len(train_loader))

    # training
    start_epoch = cfg['training']['start_epoch']
    epochs = cfg['training']['epochs']
    early_stop = 0
    best_epoch = 0
    best_Macro_F1 = np.float('-inf')
    for epoch in range(start_epoch, epochs):

        if epoch - best_epoch > 5 and epoch > 10:
            early_stop = 1
            break
        adjust_learning_rate(cfg['training']['scheduler'], opt, epoch, cfg['training']['optimizer']['lr'])
        print("Using optimizer {}".format(opt))

        # train for one epoch
        train(fold, train_loader, model, criterion, opt, epoch, scheduler)
        print("")

        # evaluate on validation set
        f1_sco = validate(fold, val_loader, model, criterion)
        print("")

        # remember best f1_sco and save checkpoint
        is_best = (f1_sco > best_Macro_F1)
        
        best_Macro_F1 = max(f1_sco, best_Macro_F1)

        if is_best:

            # Setup checkpoint_path
            model_path = os.path.join(cfg['checkpoint']['path'], cfg['model']['arch']+'_'+str(run_time)+'_'+str(cfg['training']['optimizer']['lr']))
            if not os.path.exists(model_path):
                os.makedirs(model_path)
            
            best_epoch = epoch
            print('fold:{}, f1_score:{}, improved save model.......'.format(fold, f1_sco))
            print("")

            torch.save(model.state_dict(), os.path.join(model_path, 'fold-'+str(fold)+'-'+str(cfg['model']['arch'])+'.pth'))

            # save_checkpoint({
            # 'epoch': epoch,
            # 'arch': cfg['model']['arch'],
            # 'state_dict': model.state_dict(),
            # 'best_Macro_F1': best_Macro_F1,
            # 'optimizer' : opt.state_dict(),
            # }, cfg['checkpoint']['num'], filename=os.path.join(model_path, 'fold-'+str(fold)+'-checkpoint-'+str(n_iter)+'.pth.tar'))
        else:
            print('fold:{}, f1_score:{}, best f1_score:{}'.format(fold, f1_sco, best_Macro_F1))
            print("")

        if cfg['training']['scheduler'] == '2' and epoch % 2 == 1:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg['training']['cos_length']*len(train_loader))
	    print('adjust optimzier!')
        if cfg['training']['scheduler'] == '1':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg['training']['cos_length']*len(train_loader))


    if early_stop:
        print("early_stop...")

    print("fold:", fold)
    print("best_epoch:", best_epoch)
    print("best_Macro_F1:", best_Macro_F1)
    print("good luck !")

    txt_file_name = os.path.join(model_path, 'fold-'+str(fold)+'-'+str(cfg['model']['arch'])+'.txt')

    finish_state = 'fold: ' + str(fold) + '\r\n' + \
        'best_epoch: ' + str(best_epoch) + '\r\n' + \
        'best_Macro_F1: ' + str(best_Macro_F1) + '\r\n' + \
        'good luck !'

    with open(txt_file_name, 'w') as f:     
        f.write(finish_state)
    # writer.export_scalars_to_json(os.path.join(model_path, cfg['model']['arch']+"_all_scalars.json"))
    # writer.close()


def train(fold, train_loader, model, criterion, optimizer, epoch, scheduler):
    # global n_iter # , writer
    # batch_time = AverageMeter()
    # data_time = AverageMeter()
    losses = AverageMeter()
    # f1_sco = AverageMeter()
    # top1 = AverageMeter()

    # switch to train mode
    model.train()

    # end = time.time()
    target = []
    predict = []
    for i, (inputs, targets) in enumerate(train_loader):
        # measure data loading time
        # data_time.update(time.time() - end)

        if torch.cuda.is_available():
            inputs = inputs.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)
        # compute output
        outputs = model(inputs)
        # print(outputs.size(), targets.size())
        loss = criterion(outputs, targets)

        # compute gradient and do SGD step
        # optimizer.zero_grad()
        loss.backward()
        
        # norm = nn.utils.clip_grad_norm(model.parameters(), 1.0)

        # optimizer.step()

        if cfg['training']['num_grad_acc'] is None:
            optimizer.step()
            optimizer.zero_grad()
        elif (i+1) % cfg['training']['num_grad_acc'] == 0:
            optimizer.step()
            optimizer.zero_grad()

        scheduler.step()

        # measure acc , F1 and record loss
        outputs = F.softmax(outputs, dim=1)
        prec1 = accuracy(outputs, targets)

        _, predicted = torch.max(outputs.data, 1)
        p_array = predicted.cpu().numpy().T
        t_array = targets.data.long().cpu().numpy().T

        # print(p_array)
        # print(t_array)
        # print(p_array.data)
        Now_F1_score = f1_score(t_array, p_array, average='macro')
        predict += p_array.tolist()
        target += t_array.tolist()
        # print(predict)
        # print(target)
        All_F1_score = f1_score(target, predict, average='macro')

        # print("F1_score:", F1_score)

        # f1_sco.update(F1_score.item(), inputs.size(0))
        losses.update(loss.item(), inputs.size(0))
        # top1.update(prec1[0], inputs.size(0))

        # measure elapsed time
        # batch_time.update(time.time() - end)
        # end = time.time()

        # n_iter += 1
        # writer.add_scalar('data/train_precision', top1.avg/100, n_iter)
        # writer.add_scalar('data/train_loss', losses.avg, n_iter)
        # writer.add_scalar('data/train_f1', f1_sco.avg, n_iter)
        # with torch.no_grad():
        #     y_true = target.cpu().numpy().reshape(1,-1)
        #     y_pred = np.argmax(output.cpu().numpy(), axis=1).reshape(1,-1)
        #     writer.add_scalar('metric/train_balance_accuracy', balance_accuracy(y_true, y_pred, classes=cfg['training']['nb_classes']), n_iter)

        # if n_iter % cfg['training']['print_freq'] == 1:
        #     writer.add_image('image/train', vutils.make_grid(input[:8], normalize=True, scale_each=True), n_iter)

        if i % cfg['training']['print_freq'] == 0 :
            print('fold {0}\t'
                  'Epoch: [{1}][{2}/{3}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'F1 {Now_F1_score:.3f} ({All_F1_score:.3f})\t'.format(
                   fold, epoch, i+1, len(train_loader), loss=losses, Now_F1_score=Now_F1_score, All_F1_score=All_F1_score))
    print('fold {0}\t'
          'Loss {loss.avg:.4f}\t'
          ' * F1 {All_F1_score:.3f}\t'.format(fold, loss=losses, All_F1_score=All_F1_score))



def validate(fold, val_loader, model, criterion):
    # global n_iter # , writer
    # batch_time = AverageMeter()
    losses = AverageMeter()
    # f1_sco = AverageMeter()
    # top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        # end = time.time()
        target = []
        predict = []
        for i, (inputs, targets) in enumerate(val_loader):

            if torch.cuda.is_available():
                inputs = inputs.cuda(non_blocking=True)
                targets = targets.cuda(non_blocking=True)

            # compute output
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # measure acc , F1 and record loss
            outputs = F.softmax(outputs, dim=1)
            prec1 = accuracy(outputs, targets)

            _, predicted = torch.max(outputs.data, 1)
            p_array = predicted.cpu().numpy().T
            t_array = targets.data.long().cpu().numpy().T

            Now_F1_score = f1_score(t_array, p_array, average='macro')
            predict += p_array.tolist()
            target += t_array.tolist()
            All_F1_score = f1_score(target, predict, average='macro')

            # f1_sco.update(F1_score.item(), inputs.size(0))
            losses.update(loss.item(), inputs.size(0))
            # top1.update(prec1[0], inputs.size(0))

            # # measure accuracy and record loss
            # prec1 = accuracy(output, target)
            # losses.update(loss.item(), input.size(0))
            # top1.update(prec1[0], input.size(0))

            # # measure elapsed time
            # batch_time.update(time.time() - end)
            # end = time.time()

            if i % cfg['training']['print_freq'] == 0 :
                print('fold {0}\t'
                      'Test: [{1}/{2}]\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'F1 {Now_F1_score:.3f} ({All_F1_score:.3f})\t'.format(
                       fold, i+1, len(val_loader), loss=losses, Now_F1_score=Now_F1_score, All_F1_score=All_F1_score))

        # writer.add_scalar('data/val_precision', top1.avg/100, n_iter)
        # writer.add_scalar('data/val_loss', losses.avg, n_iter)
        # writer.add_scalar('data/val_f1', f1_sco.avg, n_iter)
        # # writer.add_image('image/val', vutils.make_grid(input[:2], normalize=True, scale_each=True), n_iter)

        print('fold {0}\t'
              'Loss {loss.avg:.4f}\t'
              ' * F1 {All_F1_score:.3f}\t'.format(fold, loss=losses, All_F1_score=All_F1_score))

    return All_F1_score



if __name__ == "__main__":

    config_dir = "./configs/train_resnet34.yml"

    with open(config_dir) as fp:
        cfg = yaml.load(fp)

    run_time = time.strftime('%Y-%m-%d-%H-%M-%S',time.localtime(time.time()))
    logdir = os.path.join('runs', os.path.basename(config_dir[:-4]), str(run_time) + '_' + str(cfg['training']['optimizer']['lr']))
    # writer = SummaryWriter(log_dir=logdir)

    if not os.path.exists(logdir):
        os.makedirs(logdir)

    print('RUNDIR: {}'.format(logdir))
    shutil.copy(config_dir, logdir)
    start(cfg)
