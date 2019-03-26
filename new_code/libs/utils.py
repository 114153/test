import os
import cv2
import torch
import random
import pydicom
import numbers
import numpy as np
import pandas as pd 
from PIL import Image
import torch.nn.functional as F
from scipy.ndimage import rotate
from torch.utils.data import Dataset

import torchvision.transforms as transforms
# from imgaug import augmenters as iaa


# set random seed
random.seed(666)
np.random.seed(666)
torch.manual_seed(666)
torch.cuda.manual_seed_all(666)


class LiverDataset(Dataset):
    def __init__(self, df, mode=None, transforms=None):

        self.images_df = df.copy()
        self.images_id = df['id'].values
        self.suffix = df['suffix'].values
        self.transforms = transforms

        target = df['ret'].tolist()
        self.target = target
        self.mode = mode

        # print(len(target))

    def __len__(self):
        return len(self.images_df)

    def __getitem__(self, index):

        X = self.read_images(index)

        # print(type(X))
        # X = Image.fromarray(X[:,:,:5].astype('uint8'))
        # X = Image.fromarray(X.astype('uint8')).convert('RGB')

        # X = self.transforms(X)

        # Normalization
        # MIN_BOUND = -1000.0
        # MAX_BOUND = 400.0
        # X = (X - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
        # X[X>1] = 1.
        # X[X<0] = 0.

        # # Zero centering
        # PIXEL_MEAN = 0.25
        # X = X - PIXEL_MEAN

        # X = np.transpose(X, axes=(2, 0, 1))
        X = X.astype('uint8')

        if self.mode=='val':

            X = centerCrop(X, 460, 460)

        if self.mode=='train':

            # lr_num = random.randint(0,9)
            # ud_num = random.randint(10,19)
            # rot_num = random.randint(20,29)

            # if lr_num%2 == 0:
            #     X = np.fliplr(X).copy()

            # if ud_num%2 == 0:
            #     X = np.flipud(X).copy()

            # if rot_num%2 == 0:

            X = randomCrop(X, 460, 460)

            angle = random.randrange(-60, 65, 15)   # -60,-30,0,30,60
            # angle = random.uniform(-30, 30)
            # print(angle)
            X = rotate(X,angle,(0,1),reshape=False)

        if self.mode=='TTA1':

            # X = centerCrop(X, 460, 460)
            X = rotate(X,-30,(0,1),reshape=False)

        if self.mode=='TTA2':

            # X = centerCrop(X, 460, 460)
            X = rotate(X,30,(0,1),reshape=False)

        if self.mode=='TTA3':

            X = rotate(X,-60,(0,1),reshape=False)

        if self.mode=='TTA4':

            X = rotate(X,60,(0,1),reshape=False)

        if self.mode=='TTA5':

            X = np.fliplr(X).copy()

        if self.mode=='TTA6':

             X = np.flipud(X).copy()

        if self.mode=='TTA7':

            X = randomCrop(X, 460, 460)

        if self.mode=='TTA10':

            img1 = X[60:,:,]
            a = np.zeros((60,512,9), dtype=np.uint8)
            img2 = np.vstack((img1,a))
            X = img2

        if self.mode=='TTA11':

            img1 = X[:,:452,]
            a = np.zeros((512,60,9), dtype=np.uint8)
            img2 = np.hstack((a,img1))
            X = img2

        if self.mode=='TTA12':

            img1 = X[:452,:,]
            a = np.zeros((60,512,9), dtype=np.uint8)
            img2 = np.vstack((a,img1))
            X = img2

        if self.mode=='TTA13':

            img1 = X[:,60:,]
            a = np.zeros((512,60,9), dtype=np.uint8)
            img2 = np.hstack((img1,a))
            X = img2


        data = self.transforms(X)

        return data, self.target[index]

    def read_images(self, index):

        img_path = self.images_id[index]
        suffix = self.suffix[index]
        # print(img_path)

        image = np.load(img_path + suffix)

        return image

    # def get_max_sclice(self, img_path, file_list):

    #     file_num = len(file_list)
    #     lis = []
    #     imgs = []

    #     for i in range(file_num):

    #         file_dir = os.path.join(img_path, file_list[i])
    #         slice_dicom = pydicom.read_file(file_dir)
    #         slice_dicom = self.changeLevelWidth(slice_dicom, 60, 350)
    #         imgs.append(slice_dicom)
    #         lis.append(np.sum(slice_dicom))
    #         # print(file_dir, np.sum(slice_dicom))
    #     # print(lis.index(max(lis))+1)
    #     return imgs, lis.index(max(lis))

    # def changeLevelWidth(self, slice, level, width):

    #     image = slice.pixel_array
    #     # print(image)
    #     image = image.astype(np.int16)
    #     image[image == -2000] = 0

    #     intercept = slice.RescaleIntercept
    #     slope = slice.RescaleSlope

    #     if slope != 1:
    #         image = slope * image.astype(np.float64)
    #         image = image.astype(np.int16)
            
    #     image += np.int16(intercept)

    #     imageHU = np.array(image, dtype=np.int32)

    #     imageHU[imageHU<(level - width/2)] = level - width/2
    #     imageHU[imageHU>(level + width/2)] = level - width/2
    #     imageHU = (imageHU+width/2-level)*255/width
        
    #     return imageHU

def predict_mean(n,fold, model, data_loader):
    with torch.no_grad():  # evaulate
        model.eval()
        pred = []
        y = []
        num = len(data_loader)
        for i, (images, target) in enumerate(data_loader):
            # print(target)
            images_var = images.cuda(non_blocking=True)

            print("{} predicting fold {}\t{}/{}".format(n, fold, i, num-1))
            output = model(images_var)
            # print(output)
            output = F.softmax(output, dim=1)
            # print(output)
            # print(output)
            # print(output.cpu().data.tolist())
            pred += output.cpu().data.tolist()
            # print(pred)
            y += target.tolist()
    return y, pred

def predict_vote(fold, model, data_loader):
    with torch.no_grad():  # evaulate
        model.eval()
        pred = []
        y = []
        num = len(data_loader)
        for i, (images, target) in enumerate(data_loader):
            images_var = images.cuda(non_blocking=True)

            print("predicting fold {}\t{}/{}".format(fold, i, num-1))
            output = model(images_var)

            output = F.softmax(output, dim=1)
            _, predicted = torch.max(output.data, 1)
            p_array = predicted.cpu().numpy().T
            pred.append(p_array)
            y += target.tolist()
            
    return y, pred

def bayes(prob0,prob1,p_w0,p_w1):

    p_x = prob0*p_w0 + prob1*p_w1
    p_w0_x = prob0*p_w0/p_x
    p_w1_x = prob1*p_w1/p_x

    if p_w1_x > p_w0_x:
        return 1
    else:
        return 0

def anaylse_bayes(submit_csv,bayes_csv_path):
    bayes_csv = pd.read_csv(bayes_csv_path)
    result = []
    for row in range(len(submit_csv)):
        print(row)
        if submit_csv['sex'][row] is 'F':
            p_w1 = bayes_csv['F'][int(submit_csv['age'][row]/5)]
            p_w0 = 1-p_w1
        else:
            p_w1 = bayes_csv['M'][int(submit_csv['age'][row]/5)]
            p_w0 = 1-p_w1
        ret = bayes(submit_csv['0_prob'][row],submit_csv['1_prob'][row],p_w0,p_w1)
        result.append(ret)
    return result


def count_bayes(pred,p_w1=0.43517296,p_w0=0.56482704):

    pred_bayes = []
    for p_x_w in pred:
        # print(np.array(p_x_w).shape, type(p_x_w))
        p_x = p_x_w[1]*p_w1+p_x_w[0]*p_w0
        p_w1_x = p_w1*p_x_w[1]/p_x
        p_w0_x = p_x_w[0]*p_w0/p_x

        i = [p_w0_x, p_w1_x]

        pred_bayes.append(i)
        # print(np.array(pred_bayes).shape)
    return pred_bayes

# def get_augumentor(mode):
#     if mode =='train':
#         return iaa.OneOf([
#                     iaa.Affine(rotate=90),
#                     iaa.Affine(rotate=180),
#                     iaa.Affine(rotate=270),
#                     iaa.Fliplr(1),
#                     iaa.Flipud(1),
#                 ])
#     elif mode == 'TTA1':
#         return iaa.Flipud(1)
#     elif mode == 'TTA2':
#         return iaa.Fliplr(1)
#     elif mode == 'TTA3':
#         return iaa.Affine(rotate=90)
#     elif mode == 'TTA4':
#         return iaa.Affine(rotate=180)
#     elif mode == 'TTA5':
#         return iaa.Affine(rotate=270)
#     else:
#         raise ValueError("aug error")


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)    # (input, k, dim=None, largest=True, sorted=True, out=None)->  (values,indices)
        pred = pred.t()   # (k,n)
        correct = pred.eq(target.view(1, -1).expand_as(pred))  # (k,n)

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(float(correct_k.mul_(100.0 / batch_size)))     # batch_acc
        return res

def predict(fold, model, data_loader):
    with torch.no_grad():  # evaulate
        model.eval()
        pred = []
        y = []
        num = len(data_loader)
        for i, (images, target) in enumerate(data_loader):
            images_var = images.cuda(non_blocking=True)

            print("predicting fold {}\t{}/{}".format(fold, i, num-1))
            output = model(images_var)
            output = F.softmax(output, dim=1)
            pred += output.cpu().data.tolist()
            y += target.tolist()
    return y,pred

def balance_accuracy(y_true, y_pred, classes=None):
    """ calculate the average accuracy of each classes

    @param y_true: true label on classes
    @param y_pred: pred label on classes
    @param classes: a list contains all labels, or a number labels [0,1,...,classes]

    @return balance: average accuracy of each classes
    @return a: accuracy of each classes
    @return c: number of each lables on y_pred
    @return true: number of each labels predicted correctly
    """
    if classes is None:
        classes = np.array(range(np.max(y_true.max(), y_pred.max())))
    elif isinstance(classes, numbers.Number):
        classes = np.arange(classes)
    nb_class = len(classes)
    c = np.zeros(nb_class)
    a = np.zeros(nb_class)
    true = np.zeros(nb_class)
    for i, cla in enumerate(classes):
        c[i] = np.sum(y_pred==cla)                       # i_num
        true[i] = np.sum((y_pred==cla) & (y_true==cla))  # i_correct_num
        if c[i]>0:
            a[i] = true[i]*1.0/c[i]      # i_precision
    balance = np.mean(a)
    return balance

def save_checkpoint(state, nb_checkpoint, filename='./checkpoint/checkpoint.pth.tar'):
    path = os.path.dirname(filename)
    if not os.path.exists(path):
        os.makedirs(path)
    files = os.listdir(path)
    
    if len(files)>=nb_checkpoint:
        os.remove(os.path.join(path, files[np.argmin(np.array([os.stat(os.path.join(path, file)).st_ctime for file in files]))]))
    with open(filename, 'wb') as file:
        torch.save(state, file)


def adjust_learning_rate(mode, optimizer, epoch, lr):

    if mode == '1':
        end_lr = lr * (0.5 ** int(epoch>9)) * (0.25 ** int(epoch>14)) * (0.2 ** int(epoch>18))
        middle_lr = lr / 3 * (0.5 ** int(epoch>9)) * (0.25 ** int(epoch>14)) * (0.2 ** int(epoch>18))
        first_lr = lr / 10 * (0.5 ** int(epoch>9)) * (0.25 ** int(epoch>14)) * (0.2 ** int(epoch>18))
        num = 0
        for param_group in optimizer.param_groups:
            if num == 0:
                param_group['initial_lr'] = first_lr
            if num == 1:
                param_group['initial_lr'] = middle_lr
            if num == 2:
                param_group['initial_lr'] = end_lr
            num += 1

    if mode == '2':
	if epoch % 2 == 0:
            end_lr = lr * (0.5 ** int(epoch>9)) * (0.25 ** int(epoch>14)) * (0.2 ** int(epoch>18))
            middle_lr = lr / 3 * (0.5 ** int(epoch>9)) * (0.25 ** int(epoch>14)) * (0.2 ** int(epoch>18))
            first_lr = lr / 10 * (0.5 ** int(epoch>9)) * (0.25 ** int(epoch>14)) * (0.2 ** int(epoch>18))
            num = 0
            for param_group in optimizer.param_groups:
                if num == 0:
                    param_group['initial_lr'] = first_lr
                if num == 1:
                    param_group['initial_lr'] = middle_lr
                if num == 2:
                    param_group['initial_lr'] = end_lr
                num += 1           

    # if mode == '1':
        # end_lr = lr * (0.25 ** int(epoch>6)) * (0.25 ** int(epoch>12)) * (0.25 ** int(epoch>18))
        # middle_lr = lr / 3 * (0.25 ** int(epoch>6)) * (0.25 ** int(epoch>12)) * (0.25 ** int(epoch>18))
        # first_lr = lr / 10 * (0.25 ** int(epoch>6)) * (0.25 ** int(epoch>12)) * (0.25 ** int(epoch>18))

    # if mode == '2':
    #     end_lr = lr * (0.1 ** int(epoch>7)) * (0.1 ** int(epoch>14)) * (0.1 ** int(epoch>21))
    #     middle_lr = lr / 3 * (0.1 ** int(epoch>7)) * (0.1 ** int(epoch>14)) * (0.1 ** int(epoch>21))
    #     first_lr = lr / 10 * (0.1 ** int(epoch>7)) * (0.1 ** int(epoch>14)) * (0.1 ** int(epoch>21))

    # num = 0
    # for param_group in optimizer.param_groups:
    #     if num == 0:
    #         param_group['lr'] = first_lr
    #     if num == 1:
    #         param_group['lr'] = middle_lr
    #     if num == 2:
    #         param_group['lr'] = end_lr
    #     num += 1



def randomCrop(img, width, height):
    assert img.shape[0] >= height
    assert img.shape[1] >= width
    # assert img.shape[0] == mask.shape[0]
    # assert img.shape[1] == mask.shape[1]
    x = random.randint(0, img.shape[1] - width)
    y = random.randint(0, img.shape[0] - height)
    img = img[y:y+height, x:x+width]
    # mask = mask[y:y+height, x:x+width]
    return img


def centerCrop(img, width, height):
    assert img.shape[0] >= height
    assert img.shape[1] >= width
    x = (img.shape[0] - height) / 2
    y = (img.shape[1] - width) / 2
    img = img[y:y+height, x:x+width]
    return img
