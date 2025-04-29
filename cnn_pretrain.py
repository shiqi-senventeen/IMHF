import os
import argparse
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import numpy as np
import torch
import time
from tqdm import tqdm
import warnings
from Module import CNNTransformer
from utils import get_eval

warnings.filterwarnings("ignore")

class Config_mol_img:
    # Dataset path
    data_path = "./dataset/mol_img/"
    # Save path
    save_path = "./checkpoints/mol_img/"
    # Result path
    result_path = "./result/mol_img/"
    tensorboard_path = "./tensorboard/mol_img/"
    visual_result_path = "./Visualization/SEGACDC"

    save_path_code = "_"

    workers = 8  # number of data loading workers (default: 8)
    epochs = 400  # number of total epochs to run (default: 400)
    batch_size = 4  # batch size (default: 4)
    learning_rate = 1e-4  # initial learning rate (default: 0.001)
    momentum = 0.9  # momentum
    classes = 4  # the number of classes
    img_size = 224  # the input size of model
    train_split = "train"  # the file name of training set

    val_split = "val"
    test_split = "test"
    crop = (224, 224)  # the cropped image size
    eval_freq = 1  # the frequency of evaluate the model,default=1
    save_freq = 2000  # the frequency of saving the model,default =2000
    device = "cuda"  # training device, cpu or cuda
    cuda = "on"  # switch on/off cuda option (default: off)
    gray = "no"  # the type of input image
    img_channel = 3  # the channel of input image

    eval_mode = "mol_graph"  # the mode when evaluate the model, slice level or patient level
    pre_trained = False
    mode = "train"
    visual = False
    modelname = "CSA"


def main():
    #  =========================================== parameters setting ==================================================

    parser = argparse.ArgumentParser(description='Networks')
    parser.add_argument('--modelname', default='CSA', type=str, help='type of model')
    parser.add_argument('--task', default='mol_img', help='task or dataset name')

    args = parser.parse_args()

    # opt = get_config(args.task)  # please configure your hyper-parameter
    opt = Config_mol_img()
    opt.save_path_code = "_"
    # grey
    if opt.gray == "yes":
        from utils.utils_gray import JointTransform2D, ImageToImage2D
    else:
        # rgb
        from utils.utils_rgb import JointTransform2D, ImageToImage2D

    timestr = time.strftime('%m%d%H%M')  # initialize the tensorboard for record the training process
    # Result path
    boardpath = opt.tensorboard_path + args.modelname + opt.save_path_code + timestr
    # Create path if it does not exist
    if not os.path.isdir(boardpath):
        os.makedirs(boardpath)

    #  ============================================= model initialization ==============================================

    tf_train = JointTransform2D(img_size=opt.img_size, crop=opt.crop, p_flip=0.0, p_rota=0.5, p_scale=0.5, p_gaussn=0.0,
                                p_contr=0.5, p_gama=0.5, p_distor=0.0,
                                color_jitter_params=None, long_mask=False)  # image reprocessing

    tf_val = JointTransform2D(img_size=opt.img_size, crop=opt.crop, p_flip=0, color_jitter_params=None, long_mask=False)

    train_dataset = ImageToImage2D(opt.data_path, opt.train_split, tf_train, opt.classes)
    val_dataset = ImageToImage2D(opt.data_path, opt.val_split, tf_val, opt.classes)  # return image, mask, and filename
    trainloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True)
    valloader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    model = CNNTransformer(n_channels=3, imgsize=224)
    model.to(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    criterion = nn.MSELoss(reduction='mean')
    # ADAM
    optimizer = torch.optim.Adam(list(model.parameters()), lr=opt.learning_rate, weight_decay=1e-5)

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total_params: {}".format(pytorch_total_params))

    #  ========================================== begin to train the model =============================================

    best_loss, loss_log = 10.0, np.zeros(opt.epochs + 1)

    for epoch in range(opt.epochs):
        #  ------------------------------------ training ------------------------------------

        model.train()
        train_losses = 0

        loop = tqdm(enumerate(trainloader), total=len(trainloader), desc=f'Epoch [{epoch + 1}/{opt.epochs}]',
                    position=0)
        for batch_idx, (input_image, ground_truth, *rest) in loop:
            input_image = Variable(input_image.to(device=opt.device))

            # ---------------------------------- forward ----------------------------------
            output, latent, loss = model(input_image)

            # ---------------------------------- backward ---------------------------------
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses += loss.item()

            loop.set_postfix(loss=train_losses / (batch_idx + 1))

            loop.set_description(f'Epoch [{epoch}/{opt.epochs}]')

        #  ---------------------------- log the train progress ----------------------------
        print('epoch [{}/{}], train loss:{:.4f}'.format(epoch + 1, opt.epochs, train_losses / (batch_idx + 1)))

        loss_log[epoch] = train_losses / (batch_idx + 1)

        #  ----------------------------------- evaluate -----------------------------------
        if epoch % opt.eval_freq == 0:
            val_losses = get_eval(valloader, model, criterion, opt)

            print('epoch [{}/{}], val loss:{:.4f}'.format(epoch, opt.epochs, val_losses))
            # TensorWriter.add_scalar('val_loss', val_losses, epoch)
            if val_losses < best_loss:
                best_loss = val_losses
                timestr = time.strftime('%m%d%H%M')
                if not os.path.isdir(opt.save_path):
                    os.makedirs(opt.save_path)
                save_path = opt.save_path + args.modelname + opt.save_path_code + '%s' % timestr + '_' + str(
                    epoch) + '_' + str(best_loss)
                torch.save(model.state_dict(), save_path + ".pth", _use_new_zipfile_serialization=False)
        if epoch % opt.save_freq == 0 or epoch == (opt.epochs - 1):
            if not os.path.isdir(opt.save_path):
                os.makedirs(opt.save_path)
            save_path = opt.save_path + args.modelname + opt.save_path_code + '_' + str(epoch)
            torch.save(model.state_dict(), save_path + ".pth", _use_new_zipfile_serialization=False)


if __name__ == '__main__':
    main()
