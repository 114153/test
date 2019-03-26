import torch
import torch.nn as nn
import models.resnet as resnet
import models.densenet as densenet
import models.inception as inception
from Nadam import Nadam
import torch.optim as optim

def get_model_opt(cfg):

    if cfg['model']['arch'] == 'resnet34':
        return resnet34_model(cfg)

    if cfg['model']['arch'] == 'resnet50':
        return resnet50_model(cfg)

    if cfg['model']['arch'] == 'se_resnet50':
        return se_resnet50_model(cfg)

    if cfg['model']['arch'] == 'densenet121':
        return densenet121_model(cfg)

    if cfg['model']['arch'] == 'inception_v3':
        return inceptionv3_model(cfg)


def inceptionv3_model(cfg):

    model = inception.inception_v3(num_classes=2)

    pretrained_state = torch.load('/home/mkq/dcic2019/libs/models/weights/inception_v3.pth')
    model_dict = model.state_dict()
    pretrained_state = {k: v for k, v in pretrained_state.items() if k in model_dict}
    model_dict.update(pretrained_state)
    model.load_state_dict(model_dict)

    model.Conv2d_1a_3x3.conv = nn.Conv2d(9, 32, bias=False, kernel_size=3, stride=2)
    # model.fc = nn.Linear(2048, 2)
    # model.AuxLogits.fc = nn.Linear(768, 2)

    w = model.Conv2d_1a_3x3.conv.weight
    model.Conv2d_1a_3x3.weight = torch.nn.Parameter(torch.cat((w, w, w), dim=1))

    conv1_params = list(map(id, model.Conv2d_1a_3x3.parameters()))
    # bn1_params = list(map(id, model.bn1.parameters()))
    fc_bn1_params = list(map(id, model.fc_bn1.parameters()))
    fc_bn2_params = list(map(id, model.fc_bn2.parameters()))
    fc_1_params = list(map(id, model.fc_1.parameters()))
    fc_2_params = list(map(id, model.fc_2.parameters()))
    add_params = conv1_params+fc_bn1_params+fc_bn2_params+fc_1_params+fc_2_params

    first_conv_params = filter(lambda p:id(p) in conv1_params, model.parameters())
    classifier_params = filter(lambda p:id(p) in fc_1_params+fc_2_params+fc_bn1_params+fc_bn2_params, model.parameters())
    middle_params = filter(lambda p:id(p) not in add_params, model.parameters())

    opt = Nadam([
        {'params':filter(lambda p:p.requires_grad, first_conv_params), 'lr':cfg['training']['optimizer']['lr']/10},
        {'params':filter(lambda p:p.requires_grad, middle_params), 'lr':cfg['training']['optimizer']['lr']/3},
        {'params':filter(lambda p:p.requires_grad, classifier_params), 'lr':cfg['training']['optimizer']['lr']}],
        )

    # conv0_params = list(map(id, model.Conv2d_1a_3x3.parameters()))
    # fc_params = list(map(id, model.fc.parameters()))
    # main_params = conv0_params+fc_params

    # first_conv_params = filter(lambda p:id(p) in conv0_params, model.parameters())
    # classifier_params = filter(lambda p:id(p) in fc_params, model.parameters())
    # base_params = filter(lambda p:id(p) not in main_params, model.parameters())

    # opt = Nadam([
    # {'params':filter(lambda p:p.requires_grad, first_conv_params), 'lr':cfg['training']['optimizer']['lr']/10},
    # {'params':filter(lambda p:p.requires_grad, base_params), 'lr':cfg['training']['optimizer']['lr']/3},
    # {'params':filter(lambda p:p.requires_grad, classifier_params), 'lr':cfg['training']['optimizer']['lr']}],
    #               )

    return model, opt

def densenet121_model(cfg):

    model = densenet.densenet121(num_classes=1000)

    pretrained_state = torch.load('/home/mkq/dcic2019/libs/models/weights/densenet121.pth')
    model_dict = model.state_dict()
    pretrained_state = {k: v for k, v in pretrained_state.items() if k in model_dict}
    model_dict.update(pretrained_state)
    model.load_state_dict(model_dict)

    model.classifier = nn.Linear(model.classifier.in_features, 2)
    w = model.features.conv0.weight
    model.features.conv0 = nn.Conv2d(9,64,kernel_size=(7,7),stride=(2,2),padding=(3, 3), bias=False)
    model.features.conv0.weight = torch.nn.Parameter(torch.cat((w, w, w), dim=1))

    conv0_params = list(map(id, model.features.conv0.parameters()))
    bn0_params = list(map(id, model.features.norm0.parameters()))
    # fc_bn1_params = list(map(id, model.fc_bn1.parameters()))
    # fc_bn2_params = list(map(id, model.fc_bn2.parameters()))
    # fc_1_params = list(map(id, model.fc_1.parameters()))
    # fc_2_params = list(map(id, model.fc_2.parameters()))
    fc_params = list(map(id, model.classifier.parameters()))
    main_params = conv0_params+bn0_params+fc_params

    first_conv_params = filter(lambda p:id(p) in conv0_params+bn0_params, model.parameters())
    classifier_params = filter(lambda p:id(p) in fc_params, model.parameters())
    base_params = filter(lambda p:id(p) not in main_params, model.parameters())

    opt = Nadam([
        {'params':filter(lambda p:p.requires_grad, first_conv_params), 'lr':cfg['training']['optimizer']['lr']/10},
        {'params':filter(lambda p:p.requires_grad, base_params), 'lr':cfg['training']['optimizer']['lr']/3},
        {'params':filter(lambda p:p.requires_grad, classifier_params), 'lr':cfg['training']['optimizer']['lr']}],
                      )
    # opt = optim.Adam([
        # {'params':filter(lambda p:p.requires_grad, first_conv_params), 'lr':cfg['training']['optimizer']['lr']/3},
        # {'params':filter(lambda p:p.requires_grad, base_params), 'lr':cfg['training']['optimizer']['lr']/3},
        # {'params':filter(lambda p:p.requires_grad, classifier_params), 'lr':cfg['training']['optimizer']['lr']}],
                      # )
    return model, opt


def resnet34_model(cfg):

    model = resnet.resnet34(num_classes=2)

    # pretrained_state = torch.load('/home/mkq/dcic2019/libs/models/weights/resnet34-333f7ec4.pth')
    # model_dict = model.state_dict()
    # pretrained_state = {k: v for k, v in pretrained_state.items() if k in model_dict}
    # pop vgg 16_bn classifier.6
    # pretrained_state.pop('classifier.6.weight')
    # pretrained_state.pop('classifier.6.bias')
    # model_dict.update(pretrained_state)
    # model.load_state_dict(model_dict)

    # w = model.conv1.weight
    model.conv1 = nn.Conv2d(9,64,kernel_size=(7,7),stride=(2,2),padding=(3, 3), bias=False)
    # model.conv1.weight = torch.nn.Parameter(torch.cat((w, w, w), dim=1))

    conv1_params = list(map(id, model.conv1.parameters()))
    bn1_params = list(map(id, model.bn1.parameters()))
    fc_bn1_params = list(map(id, model.fc_bn1.parameters()))
    fc_bn2_params = list(map(id, model.fc_bn2.parameters()))
    fc_1_params = list(map(id, model.fc_1.parameters()))
    fc_2_params = list(map(id, model.fc_2.parameters()))
    add_params = conv1_params+bn1_params+fc_bn1_params+fc_bn2_params+fc_1_params+fc_2_params

    first_conv_params = filter(lambda p:id(p) in conv1_params+bn1_params, model.parameters())
    classifier_params = filter(lambda p:id(p) in fc_1_params+fc_2_params+fc_bn1_params+fc_bn2_params, model.parameters())
    middle_params = filter(lambda p:id(p) not in add_params, model.parameters())

    opt = Nadam([
        {'params':filter(lambda p:p.requires_grad, first_conv_params), 'lr':cfg['training']['optimizer']['lr']/10},
        {'params':filter(lambda p:p.requires_grad, middle_params), 'lr':cfg['training']['optimizer']['lr']/3},
        {'params':filter(lambda p:p.requires_grad, classifier_params), 'lr':cfg['training']['optimizer']['lr']}],
                      )
    return model, opt


# a = torch.ones((2,2))
# b = torch.zeros((3,2))
# c = torch.ones((2,2))
# y = torch.cat((a,b,c),0)
# print(y)

# if __name__ == "__main__":
