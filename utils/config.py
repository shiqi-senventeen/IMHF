# This file is used to configure the training parameters for each task

class Config_ACDC:
    # 数据集路径
    data_path = "./dataset/ACDC/"
    # 保存路径
    save_path = "./checkpoints/ACDC/"
    # 结果路径
    result_path = "./result/ACDC/"
    tensorboard_path = "./tensorboard/ACDC/"
    visual_result_path = "./Visualization/SEGACDC"
    load_path = "./checkpoints/ACDC/SETR_ConvFormer_11161451_9_0.860175890046231.pth"
    save_path_code = "_"

    workers = 8                  # number of data loading workers (default: 8)
    epochs = 2                 # number of total epochs to run (default: 400)
    batch_size = 3               # batch size (default: 4)
    learning_rate = 1e-4         # initial learning rate (default: 0.001)
    momentum = 0.9               # momentum
    classes = 4                  # the number of classes
    img_size = 224                # the input size of model
    train_split = "train"        # the file name of training set
    # val_split = "valofficial"
    # test_split = "testofficial"           the file name of testing set
    val_split = "val"
    test_split = "test"
    crop = (224, 224)            # the cropped image size
    eval_freq = 1                # the frequency of evaluate the model,default=1
    save_freq = 2000               # the frequency of saving the model,default =2000
    device = "cuda"              # training device, cpu or cuda
    cuda = "on"                  # switch on/off cuda option (default: off)
    gray = "no"                 # the type of input image
    img_channel = 3              # the channel of input image
    # 设置为patient_record即可在eval的时候写入文件，对应get_eval函数
    eval_mode = "mol_graph"      # the mode when evaluate the model, slice level or patient level
    pre_trained = False
    mode = "train"
    visual = False
    modelname = "ConvFormer"


class Config_ICH:
    data_path = "../../dataset/ICH/"
    save_path = "./checkpoints/ICH/"
    result_path = "./result/ICH/"
    tensorboard_path = "./tensorboard/ICH/"
    visual_result_path = "./Visualization/SEGICH"
    load_path = "./xxxx"
    save_path_code = "_"

    workers = 1                  # number of data loading workers (default: 8)
    epochs = 400                 # number of total epochs to run (default: 400)
    batch_size = 4               # batch size (default: 4)
    learning_rate = 1e-4         # initial learning rate (default: 0.001)
    momentum = 0.9               # momentum
    classes = 2                  # the number of classes
    img_size = 256                # the input size of model
    train_split = "train"        # the file name of training set
    val_split = "val"
    test_split = "test"           # the file name of testing set
    crop = (256, 256)            # the cropped image size
    eval_freq = 1                # the frequency of evaluate the model
    save_freq = 2000               # the frequency of saving the model
    device = "cuda"              # training device, cpu or cuda
    cuda = "on"                  # switch on/off cuda option (default: off)
    gray = "yes"                 # the type of input image
    img_channel = 1              # the channel of input image
    # 设置为patient_record即可在eval的时候写入文件，对应get_eval函数
    eval_mode = "patient_record"        # the mode when evaluate the model, slice level or patient level
    pre_trained = False
    mode = "train"
    visual = True
    modelname = "ConvFormer"


class Config_ISIC:
    data_path = "../../dataset/ISIC/"
    save_path = "./checkpoints/ISIC/"
    result_path = "./result/ISIC/"
    tensorboard_path = "./tensorboard/ISIC/"
    visual_result_path = "./Visualization/SEGISIC"
    load_path = "./xxx"
    save_path_code = "_"

    workers = 16                  # number of data loading workers (default: 8)
    epochs = 40                 # number of total epochs to run (default: 400)
    batch_size = 4               # batch size (default: 4)
    learning_rate = 1e-4         # initial learning rate (default: 0.001)
    momentum = 0.9               # momentum
    classes = 2                  # the number of classes (background + foreground)
    img_size = 256               # the input size of model
    train_split = "train"  # the file name of training set
    val_split = "test"     # the file name of testing set
    test_split = "test"     # the file name of testing set
    crop = None                  # the cropped image size
    eval_freq = 1                # the frequency of evaluate the model
    save_freq = 2000               # the frequency of saving the model
    device = "cuda"              # training device, cpu or cuda
    cuda = "on"                  # switch on/off cuda option (default: off)
    gray = "no"                 # the type of input image
    img_channel = 3              # the channel of input image
    eval_mode = "slice"        # the mode when evaluate the model, slice level or patient level
    pre_trained = True
    mode = "train"
    visual = False

class Config_mol_img:
    # 数据集路径
    data_path = "./dataset/mol_img/"
    # 保存路径
    save_path = "./checkpoints/mol_img/"
    # 结果路径
    result_path = "./result/mol_img/"
    tensorboard_path = "./tensorboard/mol_img/"
    visual_result_path = "./Visualization/SEGACDC"
    load_path = "./checkpoints/ACDC/SETR_ConvFormer_11161451_9_0.860175890046231.pth"
    save_path_code = "_"

    workers = 8                  # number of data loading workers (default: 8)
    epochs = 400                 # number of total epochs to run (default: 400)
    batch_size = 6               # batch size (default: 4)
    learning_rate = 1e-4         # initial learning rate (default: 0.001)
    momentum = 0.9               # momentum
    classes = 4                  # the number of classes
    img_size = 224                # the input size of model
    train_split = "train"        # the file name of training set
    # val_split = "valofficial"
    # test_split = "testofficial"           the file name of testing set
    val_split = "val"
    test_split = "test"
    crop = (224, 224)            # the cropped image size
    eval_freq = 1                # the frequency of evaluate the model,default=1
    save_freq = 2000               # the frequency of saving the model,default =2000
    device = "cuda"              # training device, cpu or cuda
    cuda = "on"                  # switch on/off cuda option (default: off)
    gray = "no"                 # the type of input image
    img_channel = 3              # the channel of input image
    # 设置为patient_record即可在eval的时候写入文件，对应get_eval函数
    eval_mode = "mol_graph"      # the mode when evaluate the model, slice level or patient level
    pre_trained = False
    mode = "train"
    visual = False
    modelname = "ConvFormer"

# ==================================================================================================
def get_config(task="Synapse"):
    return Config_mol_img()
