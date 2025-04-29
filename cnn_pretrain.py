import os
import argparse
import torch
import torchvision

from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import time
import random
# from utils.loss_functions.dice_loss import DC_and_BCE_loss, DC_and_CE_loss, SoftDiceLoss
from tqdm import tqdm
import warnings

from Module.AttentiveLayers import CNNTransformer

warnings.filterwarnings("ignore")

# from Models.models import get_model
# from SETR import CNNTransformer
# from utils.evaluation import get_eval
# from utils.loss_functions.dice_loss import DC_and_CE_loss
from utils.config import get_config

from utils.evaluation import get_eval
from utils.loss_functions.dice_loss import DC_and_CE_loss

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class Config:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():

    #  =========================================== parameters setting ==================================================

    parser = argparse.ArgumentParser(description='Networks')
    parser.add_argument('--modelname', default='SETR_ConvFormer', type=str, help='type of model')       # 模型名称
    parser.add_argument('--task', default='mol_img', help='task or dataset name')      # 数据集类别

    args = parser.parse_args()                  # 加载参数
    # 配置超参数，将数据集类型传入get_config函数
    opt = get_config(args.task)  # please configure your hyper-parameter
    opt.save_path_code = "_"

    #

    # 灰度图
    if opt.gray == "yes":
        from utils.utils_gray import JointTransform2D, ImageToImage2D
    else:
        # rgb图
        from utils.utils_rgb import JointTransform2D, ImageToImage2D

    # example:timestr = 11151419(月日时分）
    timestr = time.strftime('%m%d%H%M')  # initialize the tensorboard for record the training process
    # 设置tensorboard的路径
    boardpath = opt.tensorboard_path + args.modelname + opt.save_path_code + timestr
    # 如果路径不存在，创建路径
    if not os.path.isdir(boardpath):
        os.makedirs(boardpath)
    # tensorwriter配置完成
    # TensorWriter = SummaryWriter(boardpath)

    # torch.backends.cudnn.enabled = True # Whether to use nondeterministic algorithms to optimize operating efficiency
    # torch.backends.cudnn.benchmark = True

    '''
    在Pytorch中设置随机数种子，保证实验可重复
    '''
    #  ============================= add the seed to make sure the results are reproducible ============================
    # # 配置seed保证结果可复现
    # seed_value = 30  # the number of seed
    # # 保证每次生成的随机数一样
    # np.random.seed(seed_value)  # set random seed for numpy
    # # 同上，保证每次生成的随机数一样
    # random.seed(seed_value)  # set random seed for python
    # os.environ['PYTHONHASHSEED'] = str(seed_value)  # avoid hash random
    # torch.manual_seed(seed_value)  # set random seed for CPU
    # torch.cuda.manual_seed(seed_value)  # set random seed for one GPU
    # torch.cuda.manual_seed_all(seed_value)  # set random seed for all GPU
    # torch.backends.cudnn.deterministic = True  # set random seed for convolution

    #  ============================================= model initialization =============     =================================
    # 训练模型的变换
    tf_train = JointTransform2D(img_size=opt.img_size, crop=opt.crop, p_flip=0.0, p_rota=0.5, p_scale=0.5, p_gaussn=0.0,
                                p_contr=0.5, p_gama=0.5, p_distor=0.0, 
                                color_jitter_params=None, long_mask=False)  # image reprocessing
    # 评估模型的变换
    tf_val = JointTransform2D(img_size=opt.img_size, crop=opt.crop, p_flip=0, color_jitter_params=None, long_mask=False)

    train_dataset = ImageToImage2D(opt.data_path, opt.train_split, tf_train, opt.classes)
    val_dataset = ImageToImage2D(opt.data_path, opt.val_split, tf_val, opt.classes)  # return image, mask, and filename
    trainloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True)
    valloader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    # model = get_model(modelname=args.modelname, img
    # ._size=opt.img_size, img_channel=opt.img_channel, classes=opt.classes)
    model=CNNTransformer(n_channels=3, imgsize=224)
    model.to(Config.device)
    # if opt.pre_trained:
    # model.load_state_dict(torch.load("./checkpoints/mol_img/SETR_ConvFormer_04121145_10_0.0171802899248953.pth"))

    # 定义损失函数
    criterion = nn.MSELoss(reduction='mean')
    # ADAM优化器
    optimizer = torch.optim.Adam(list(model.parameters()), lr=opt.learning_rate, weight_decay=1e-5)

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total_params: {}".format(pytorch_total_params))

    # if torch.cuda.device_count() > 1:  # distributed parallel training
    #     print("Let's use", torch.cuda.device_count(), "GPUs!")
    #     dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
    #     model = nn.DataParallel(model, device_ids = [0,1]).cuda()

    #  ========================================== begin to train the model =============================================

    best_loss, loss_log = 10.0, np.zeros(opt.epochs+1)

    for epoch in range(opt.epochs):
        #  ------------------------------------ training ------------------------------------
        # 切换训练模式
        model.train()
        train_losses = 0
        # enumerate(trainloader)返回的是索引和训练批次
        # trainloader return image, mask, id_ + '.png'
        # loop = tqdm((trainloader), total=len(trainloader))
        loop = tqdm(enumerate(trainloader),total=len(trainloader),desc=f'Epoch [{epoch+1}/{opt.epochs}]',position=0)
        for batch_idx, (input_image, ground_truth, *rest) in loop:
        # for batch_idx, (input_image, ground_truth, *rest) in loop:
            input_image = Variable(input_image.to(device=opt.device))
            # print(input_image.shape)
            ground_truth = Variable(ground_truth.to(device=opt.device))
            # print(ground_truth.shape)
            # ---------------------------------- forward ----------------------------------
            output,latent,loss = model(input_image)
            # print(output.shape)
            # print(latent)
            # train_loss = criterion(output, ground_truth)
            # ---------------------------------- backward ---------------------------------
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses += loss.item()

            loop.set_postfix(loss=train_losses / (batch_idx + 1))
            # 这部分的打印感觉有点问题
            loop.set_description(f'Epoch [{epoch}/{opt.epochs}]')
            # loop.set_postfix(loss=train_loss.item())

            # print("train_loss:",train_losses)
        #  ---------------------------- log the train progress ----------------------------
        print('epoch [{}/{}], train loss:{:.4f}'.format(epoch+1, opt.epochs, train_losses / (batch_idx + 1)))
        TensorWriter.add_scalar('train_loss', train_losses / (batch_idx + 1), epoch)
        loss_log[epoch] = train_losses / (batch_idx + 1)
        #  ----------------------------------- evaluate -----------------------------------
        # if epoch % opt.eval_freq == 0:
        #     dices, mean_dice, _, val_losses = get_eval(valloader, model, criterion, opt)
        #     print('epoch [{}/{}], val loss:{:.4f}'.format(epoch, opt.epochs, val_losses))
        #     print('epoch [{}/{}], val dice:{:.4f}'.format(epoch, opt.epochs, mean_dice))
        #     print("dice of each class:", dices)
        #     TensorWriter.add_scalar('val_loss', val_losses, epoch)
        #     TensorWriter.add_scalar('dices', mean_dice, epoch)
        #     if mean_dice > best_dice:
        #         best_dice = mean_dice
        #         timestr = time.strftime('%m%d%H%M')
        #         if not os.path.isdir(opt.save_path):
        #             os.makedirs(opt.save_path)
        #         save_path = opt.save_path + args.modelname + opt.save_path_code + '%s' % timestr + '_' + str(epoch) + '_' + str(best_dice)
        #         torch.save(model.state_dict(), save_path + ".pth", _use_new_zipfile_serialization=False)
        # if epoch % opt.save_freq == 0 or epoch == (opt.epochs-1):
        #     if not os.path.isdir(opt.save_path):
        #         os.makedirs(opt.save_path)
        #     save_path = opt.save_path + args.modeln   ame + opt.save_path_code + '_' + str(epoch)
        #     torch.save(model.state_dict(), save_path + ".pth", _use_new_zipfile_serialization=False)


        #  ----------------------------------- evaluate -----------------------------------
        if epoch % opt.eval_freq == 0:
            val_losses = get_eval(valloader, model, criterion, opt)

            print('epoch [{}/{}], val loss:{:.4f}'.format(epoch, opt.epochs, val_losses))
            TensorWriter.add_scalar('val_loss', val_losses, epoch)
            if val_losses < best_loss:
                best_loss = val_losses
                timestr = time.strftime('%m%d%H%M')
                if not os.path.isdir(opt.save_path):
                    os.makedirs(opt.save_path)
                save_path = opt.save_path + args.modelname + opt.save_path_code + '%s' % timestr + '_' + str(epoch) + '_' + str(best_loss)
                torch.save(model.state_dict(), save_path + ".pth", _use_new_zipfile_serialization=False)
        if epoch % opt.save_freq == 0 or epoch == (opt.epochs-1):
            if not os.path.isdir(opt.save_path):
                os.makedirs(opt.save_path)
            save_path = opt.save_path + args.modelname + opt.save_path_code + '_' + str(epoch)
            torch.save(model.state_dict(), save_path + ".pth", _use_new_zipfile_serialization=False)


if __name__ == '__main__':
    main()
            


