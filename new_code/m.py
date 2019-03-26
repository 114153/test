import os
import yaml
import argparse
import numpy as np
import pandas as pd

import torch
# # import torch.nn as nn
# # import torch.nn.functional as F
# # from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms


from libs.set_models import *
from libs.utils import predict_mean
from libs.utils import LiverDataset
from libs.utils import count_bayes
from libs.utils import bayes


# # os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def get_predict_result(cfg):

    data_dir = '/home/dataset/'
    dataset = pd.read_csv(data_dir+'submit_example.csv')

    test = dataset.copy()
    test['id'] = data_dir + 'test_dataset/' + test['id']
    test['suffix'] = '.npy'

    sex_age_dir = '/home/dataset/test_count_age.csv'
    beyes_dir = '/home/dataset/all_age_3.csv'
    sex_age = pd.read_csv(sex_age_dir)
    beyes = pd.read_csv(beyes_dir)

    result_test = dataset.copy()
    #result_test_softmax = dataset.copy()
    # print(test.head())

    # normalize = transforms.Normalize(mean=[0.2012, 0.2073, 0.2122, 0.2156, 0.2187, 0.2159, 0.2131, 0.2100, 0.2068],
    #                                  std=[0.2495, 0.2509, 0.2521, 0.2530, 0.2546, 0.2530, 0.2517, 0.2505, 0.2494])

    test_transforms = transforms.Compose([
            # transforms.CenterCrop(1500),
            # transforms.RandomResizedCrop(299),
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.27713317, 0.28488823, 0.29159758, 0.29691843, 0.29974841, 0.29707527,0.29284262, 0.28850767, 0.28427105],
                            std=[0.28050878, 0.27913887, 0.27838972, 0.2785273, 0.27935466, 0.27726396, 0.27585287, 0.275116, 0.27461502])
        ])

    # test_dataset0 = LiverDataset(test, 'None', test_transforms)
    test_dataset0 = LiverDataset(test, 'TTA7', test_transforms)
    test_dataset3 = LiverDataset(test, 'TTA7', test_transforms)
    test_dataset4 = LiverDataset(test, 'TTA7', test_transforms)
    # test_dataset1 = LiverDataset(test, 'TTA7', test_transforms)
    # test_dataset2 = LiverDataset(test, 'TTA7', test_transforms)
    test_dataset1 = LiverDataset(test, 'TTA1', test_transforms)
    test_dataset2 = LiverDataset(test, 'TTA2', test_transforms)
    # test_dataset5 = LiverDataset(test, 'TTA')


    # test_dataset3 = LiverDataset(test, 'TTA3', test_transforms)
    # test_dataset4 = LiverDataset(test, 'TTA4', test_transforms)

    # test_dataset10 = LiverDataset(test, 'TTA10', test_transforms)
    # test_dataset11 = LiverDataset(test, 'TTA11', test_transforms)
    # test_dataset12 = LiverDataset(test, 'TTA12', test_transforms)
    # test_dataset13 = LiverDataset(test, 'TTA13', test_transforms)
    # test_dataset9 = LiverDataset(test, 'TTA9', test_transforms)

    # print(len(test_dataset))
    print(cfg)


    test_loader0 = torch.utils.data.DataLoader(
        test_dataset0, batch_size=cfg['training']['batch_size'], shuffle=False,
        num_workers=cfg['training']['workers'], pin_memory=True)

    test_loader1 = torch.utils.data.DataLoader(
        test_dataset1, batch_size=cfg['training']['batch_size'], shuffle=False,
        num_workers=cfg['training']['workers'], pin_memory=True)

    test_loader2 = torch.utils.data.DataLoader(
        test_dataset2, batch_size=cfg['training']['batch_size'], shuffle=False,
        num_workers=cfg['training']['workers'], pin_memory=True)

    test_loader3 = torch.utils.data.DataLoader(
        test_dataset3, batch_size=cfg['training']['batch_size'], shuffle=False,
        num_workers=cfg['training']['workers'], pin_memory=True)

    test_loader4 = torch.utils.data.DataLoader(
        test_dataset4, batch_size=cfg['training']['batch_size'], shuffle=False,
        num_workers=cfg['training']['workers'], pin_memory=True)


    # test_loader10 = torch.utils.data.DataLoader(
    #     test_dataset10, batch_size=cfg['training']['batch_size'], shuffle=False,
    #     num_workers=cfg['training']['workers'], pin_memory=True)

    # test_loader11 = torch.utils.data.DataLoader(
    #     test_dataset11, batch_size=cfg['training']['batch_size'], shuffle=False,
    #     num_workers=cfg['training']['workers'], pin_memory=True)

    # test_loader12 = torch.utils.data.DataLoader(
    #     test_dataset12, batch_size=cfg['training']['batch_size'], shuffle=False,
    #     num_workers=cfg['training']['workers'], pin_memory=True)

    # test_loader13 = torch.utils.data.DataLoader(
    #     test_dataset13, batch_size=cfg['training']['batch_size'], shuffle=False,
    #     num_workers=cfg['training']['workers'], pin_memory=True)

    # test_loader9 = torch.utils.data.DataLoader(
    #     test_dataset9, batch_size=cfg['training']['batch_size'], shuffle=False,
    #     num_workers=cfg['training']['workers'], pin_memory=True)

    # for i, (input, target) in enumerate(test_loader):
    #     print(input.shape)
    #     print(target)

    model, _ = get_model_opt(cfg)
    if cfg['model']['arch'].startswith('alexnet') or cfg['model']['arch'].startswith('vgg'):
        model._features = torch.nn.DataParallel(model._features)
        model.cuda()
    else:
        model = torch.nn.DataParallel(model).cuda()

    cudnn.benchmark = True

    result1 = []
    result2 = []
    for fold in range(cfg['training']['nfold']):

        if fold != 0:
            break

        print("#############################################")
        base_dir = '/home/mkq/dcic2019/checkpoint/'
        checkpoint_file_name = 'resnet34_2019-03-14-12-09-02_0.0004'
        weight_dir = base_dir + checkpoint_file_name + '/fold-' + str(fold) + '-' + str(cfg['model']['arch']) + '.pth'
        print(weight_dir)
        model.load_state_dict(torch.load(weight_dir))

        _, pred0 = predict_mean(0,fold, model, test_loader0)
        _, pred1 = predict_mean(1,fold, model, test_loader1)
        _, pred2 = predict_mean(2,fold, model, test_loader2)
        _, pred3 = predict_mean(3,fold, model, test_loader3)
       # _, pred4 = predict_mean(4,fold, model, test_loader4)

        # _, pred10 = predict_mean(10,fold, model, test_loader10)
        # _, pred11 = predict_mean(11,fold, model, test_loader11)
        # _, pred12 = predict_mean(12,fold, model, test_loader12)
        # _, pred13 = predict_mean(13,fold, model, test_loader13)
        # _, pred9 = predict_mean(9,fold, model, test_loader9)

        # pred0 = count_bayes(pred0)
        # pred1 = count_bayes(pred1)
        # pred2 = count_bayes(pred2)
        # pred3 = count_bayes(pred3)
        # pred4 = count_bayes(pred4)


        result1.append(np.average([pred0, pred1, pred2, pred3],axis=0))
        # result2.append(np.average([pred0,pred10,pred11,pred12,pred13],axis=0))
        # result.append(np.average([pred0,pred1,pred2,pred3,pred4,pred5,pred6,pred7,pred8,pred9],axis=0))
        # print(np.array(result).shape)

    result1 = np.average(result1, axis=0)
    #result2 = np.average(result2, axis=0)
    # print(result.shape)

    result1 = count_bayes(result1)
    # result2 = count_bayes(result2)

    # result = np.average([result1, result2],axis=0)

    result_test['0_prob'] = np.array(result1)[:,0]
    result_test['1_prob'] = np.array(result1)[:,1]
    # save soft_max
    #result_test_softmax['0_prob'] = np.array(result1)[:,0]
    #result_test_softmax['1_prob'] = np.array(result1)[:,1]

    # result = count_bayes(result)

    # result = np.argmax(result1, axis=1)


    final_result = []
    for row in range(len(result_test)):
        # print(row)
        if sex_age['sex'][row] is 'F':
            p_w1 = beyes['F'][int(sex_age['age'][row])]
            p_w0 = 1-p_w1
        else:
            p_w1 = beyes['M'][int(sex_age['age'][row])]
            p_w0 = 1-p_w1
        ret = bayes(result_test['0_prob'][row],result_test['1_prob'][row],p_w0,p_w1)
        final_result.append(ret)

    # print(result.shape)

    dataset['ret'] = final_result

    base_submit_path = '/home/mkq/dcic2019/res/' + checkpoint_file_name

    if not os.path.exists(base_submit_path):
        os.makedirs(base_submit_path)

    submit_path = base_submit_path + '/TTA1-4_submit_mean_' + cfg['model']['arch'] + '.csv' 
    dataset.to_csv(submit_path, index=False)
    # result_test_softmax.to_csv(os.path.join(base_submit_path,'soft_max.csv'),index=False)
    print("good luck !")

    
# #     base_checkpoint_path = cfg['checkpoint']['path']
# #     resume = os.path.join(base_checkpoint_path, cfg['file_path'], cfg['submit'])
# #     # resume = "./checkpoint/resnet50_2019-01-28-14-32-21/checkpoint-9752.pth.tar"
# #     print("=> loading checkpoint '{}'".format(resume))
# #     checkpoint = torch.load(resume)
# #     state = checkpoint['state_dict']
# #     model.load_state_dict(state)

# #     

# #     test_transforms = transforms.Compose([
# #             # transforms.CenterCrop(1500),
# #             # transforms.RandomResizedCrop(299),
# #             transforms.ToTensor(),
# #             # normalize,
# #         ])

# #     BASE_PATH = cfg['dataset']['data']

# #     TEST_LABEL_PATH = os.path.join(BASE_PATH, "submit_example.csv")
# #     test_df = pd.read_csv(TEST_LABEL_PATH)
# #     test_df['id'] = BASE_PATH + cfg['test_file_name'] + test_df['id']
# #     test_df['suffix'] = '.npy'

# #     test_dataset = LiverDataset(test_df, test_transforms)

# #     test_loader = torch.utils.data.DataLoader(
# #         test_dataset, batch_size=cfg['training']['batch_size'], shuffle=False,
# #         num_workers=cfg['training']['workers'], pin_memory=True)

# #     # print(len(test_loader))

# #     model.eval()

# #     pred_list = []
# #     for batch_idx, (inputs, target) in enumerate(test_loader):
# #         if torch.cuda.is_available():
# #             print(batch_idx)
# #             inputs = inputs.cuda()
# #             # inputs = Variable(inputs, volatile=True)
# #             outputs = model(inputs)
# #             outputs = F.softmax(outputs, dim=1)
# #             # print(outputs)
# #             _, predicted = torch.max(outputs.data, 1)
# #             p_array = predicted.cpu().numpy().T
# #             pred_list.append(p_array)
# #     # break

# #     pred = np.concatenate(pred_list, axis=-1)
# #     print(pred.shape)
# #     print(pred)
# #     # filename = args.dir_logs+'.predict'
# #     # ftest = file('./res/'+filename, 'w')
# #     # for i in range(pred_.shape[0]):
# #     #     if pred_[i] == 0:
# #     #         pred_[i] = 1
# #     #     else:
# #     #         pred_[i] = 0
# #     #     if i == (pred_.shape[0] - 1):
# #     #         ftest.write(str(pred_[i]))
# #     #     else:
# #     #         ftest.write(str(pred_[i]) + '\r\n')
# #     # ftest.close()

# #     test_df = pd.read_csv(TEST_LABEL_PATH)
# #     test_df['ret'] = pred

# #     base_submit_path = './results'
# #     file_name = cfg['submit'].split('.')[0]
# #     submit_path = base_submit_path + '/submit_' + cfg['model']['arch'] + '_' + file_name + '.csv' 
# #     test_df.to_csv(submit_path,index=False)



if __name__ == "__main__":

    # parser = argparse.ArgumentParser(description="config")
    # parser.add_argument(
    #     "--config",
    #     nargs="?",
    #     type=str,
    #     default="./configs/train_resnet34.yml",   #
    #     help="Configuration file to use"
    # )

    config_dir = "/home/mkq/dcic2019/configs/train_resnet34.yml"

    # args = parser.parse_args()

    with open(config_dir) as fp:
        cfg = yaml.load(fp)

    get_predict_result(cfg)
