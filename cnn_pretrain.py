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
    """
    Configuration class for molecular image segmentation task.
    Contains all hyperparameters and paths for training and evaluation.
    """
    # Dataset path
    data_path = "./dataset/mol_img/"
    # Save path for model checkpoints
    save_path = "./checkpoints/mol_img/"
    # Path for results and outputs
    result_path = "./result/mol_img/"
    # TensorBoard logging path
    tensorboard_path = "./tensorboard/mol_img/"
    # Path for visualization results
    visual_result_path = "./Visualization/SEGACDC"

    # Identifier for saved models
    save_path_code = "_"

    # Training parameters
    workers = 8  # Number of data loading workers
    epochs = 400  # Total number of training epochs
    batch_size = 4  # Batch size for training
    learning_rate = 1e-4  # Initial learning rate
    momentum = 0.9  # Momentum for optimizer
    classes = 4  # Number of segmentation classes
    img_size = 224  # Input image size
    train_split = "train"  # Name of training dataset split

    # Evaluation parameters
    val_split = "val"  # Name of validation dataset split
    test_split = "test"  # Name of test dataset split
    crop = (224, 224)  # Size for image crop
    eval_freq = 1  # Frequency for model evaluation (epochs)
    save_freq = 2000  # Frequency for saving model checkpoints (epochs)
    device = "cuda"  # Training device (cuda/cpu)
    cuda = "on"  # Switch for CUDA usage
    gray = "no"  # Whether to use grayscale images
    img_channel = 3  # Number of input image channels

    # Model settings
    eval_mode = "mol_graph"  # Evaluation mode (slice or patient level)
    pre_trained = False  # Whether to use pre-trained weights
    mode = "train"  # Run mode (train/test/etc.)
    visual = False  # Enable visualization
    modelname = "CSA"  # Model architecture name


def main():
    """
    Main function to run the training pipeline.
    Handles argument parsing, data loading, model creation, and training loop.
    """
    #  =========================================== Parameters Setting ==================================================

    parser = argparse.ArgumentParser(description='Networks')
    parser.add_argument('--modelname', default='CSA', type=str, help='type of model')
    parser.add_argument('--task', default='mol_img', help='task or dataset name')

    args = parser.parse_args()

    # Get configuration for the specified task
    # opt = get_config(args.task)  # please configure your hyper-parameter
    opt = Config_mol_img()
    opt.save_path_code = "_"

    # Import appropriate transformation utilities based on image type (grayscale/RGB)
    if opt.gray == "yes":
        from utils.utils_gray import JointTransform2D, ImageToImage2D
    else:
        # RGB images
        from utils.utils_rgb import JointTransform2D, ImageToImage2D

    # Create timestamp for logging
    timestr = time.strftime('%m%d%H%M')
    # Initialize the tensorboard for recording the training process
    boardpath = opt.tensorboard_path + args.modelname + opt.save_path_code + timestr
    # Create directory if it doesn't exist
    if not os.path.isdir(boardpath):
        os.makedirs(boardpath)

    #  ============================================= Model Initialization ==============================================

    # Define image transformations for training
    tf_train = JointTransform2D(img_size=opt.img_size, crop=opt.crop, p_flip=0.0, p_rota=0.5, p_scale=0.5, p_gaussn=0.0,
                                p_contr=0.5, p_gama=0.5, p_distor=0.0,
                                color_jitter_params=None, long_mask=False)  # Image preprocessing

    # Define image transformations for validation (minimal augmentation)
    tf_val = JointTransform2D(img_size=opt.img_size, crop=opt.crop, p_flip=0, color_jitter_params=None, long_mask=False)

    # Create datasets for training and validation
    train_dataset = ImageToImage2D(opt.data_path, opt.train_split, tf_train, opt.classes)
    val_dataset = ImageToImage2D(opt.data_path, opt.val_split, tf_val, opt.classes)  # Returns image, mask, and filename

    # Create data loaders
    trainloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True)
    valloader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    # Initialize the model
    model = CNNTransformer(n_channels=3, imgsize=224)
    model.to(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    # Define loss function
    criterion = nn.MSELoss(reduction='mean')

    # Initialize optimizer (ADAM)
    optimizer = torch.optim.Adam(list(model.parameters()), lr=opt.learning_rate, weight_decay=1e-5)

    # Count and display trainable parameters
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total_params: {}".format(pytorch_total_params))

    #  ========================================== Begin Training Process =============================================

    best_loss, loss_log = 10.0, np.zeros(opt.epochs + 1)

    for epoch in range(opt.epochs):
        #  ------------------------------------ Training Phase ------------------------------------

        model.train()
        train_losses = 0

        # Create progress bar for tracking
        loop = tqdm(enumerate(trainloader), total=len(trainloader), desc=f'Epoch [{epoch + 1}/{opt.epochs}]',
                    position=0)

        for batch_idx, (input_image, ground_truth, *rest) in loop:
            # Move input to device (GPU/CPU)
            input_image = Variable(input_image.to(device=opt.device))

            # ---------------------------------- Forward Pass ----------------------------------
            output, latent, loss = model(input_image)

            # ---------------------------------- Backward Pass ---------------------------------
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses += loss.item()

            # Update progress bar with current loss
            loop.set_postfix(loss=train_losses / (batch_idx + 1))
            loop.set_description(f'Epoch [{epoch}/{opt.epochs}]')

        #  ---------------------------- Log Training Progress ----------------------------
        print('epoch [{}/{}], train loss:{:.4f}'.format(epoch + 1, opt.epochs, train_losses / (batch_idx + 1)))

        # Store loss for this epoch
        loss_log[epoch] = train_losses / (batch_idx + 1)

        #  ----------------------------------- Evaluation Phase -----------------------------------
        if epoch % opt.eval_freq == 0:
            # Calculate validation loss
            val_losses = get_eval(valloader, model, criterion, opt)

            print('epoch [{}/{}], val loss:{:.4f}'.format(epoch, opt.epochs, val_losses))
            # TensorWriter.add_scalar('val_loss', val_losses, epoch)

            # Save model if validation loss improves
            if val_losses < best_loss:
                best_loss = val_losses
                timestr = time.strftime('%m%d%H%M')

                # Create directory if it doesn't exist
                if not os.path.isdir(opt.save_path):
                    os.makedirs(opt.save_path)

                # Construct save path with model name, timestamp, epoch, and best loss
                save_path = opt.save_path + args.modelname + opt.save_path_code + '%s' % timestr + '_' + str(
                    epoch) + '_' + str(best_loss)

                # Save model weights
                torch.save(model.state_dict(), save_path + ".pth", _use_new_zipfile_serialization=False)

        # Periodic model saving based on save frequency
        if epoch % opt.save_freq == 0 or epoch == (opt.epochs - 1):
            if not os.path.isdir(opt.save_path):
                os.makedirs(opt.save_path)

            save_path = opt.save_path + args.modelname + opt.save_path_code + '_' + str(epoch)
            torch.save(model.state_dict(), save_path + ".pth", _use_new_zipfile_serialization=False)


if __name__ == '__main__':
    main()