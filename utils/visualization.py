import torchvision
import os
import torch
import cv2
import numpy as np
import matplotlib

matplotlib.use('Agg')  # Use Agg backend for non-interactive plotting
import matplotlib.pyplot as plt
from utils.imgname import read_img_name
import seaborn as sns


def network_inputs_visual(center_input, assist_input,
                          out_dir='./utils/visualization',  # Directory path to save feature maps
                          save_feature=False,  # Whether to save feature maps as images
                          slice_number=5,
                          show_feature=True,  # Whether to display feature maps using plt
                          ):
    """
    Visualize network inputs including center and assistant inputs.

    Args:
        center_input: Center input tensor of shape [B, C, H, W]
        assist_input: Assistant input tensor
        out_dir: Output directory for visualizations
        save_feature: Whether to save visualization as image files
        slice_number: Number of slices to visualize
        show_feature: Whether to show the feature maps
    """
    # feature = feature.detach().cpu()
    b, c, h, w = center_input.shape
    over_input = assist_input[:, :slice_number, :, :, :]
    under_input = assist_input[:, slice_number:, :, :, :]

    if b > 6:
        b = 6  # Limit to displaying maximum 6 samples

    for i in range(b):
        figure = np.zeros(((h + 30) * 2, (w + 30) * (slice_number + 1) + 30), dtype=np.uint8) + 255
        # Place center input in the visualization
        figure[10:h + 10, 10 + (w + 20) * 0: 10 + (w + 20) * 0 + w] = center_input[i, 0, :, :] * 255

        # Place over inputs in the visualization
        for j in range(1, (slice_number + 1)):
            overj = over_input[:, j - 1, :, :, :]
            figure[10:h + 10, 10 + (w + 20) * j: 10 + (w + 20) * j + w] = overj[i, 0, :, :] * 255

        # Place under inputs in the visualization
        for j in range(1, (slice_number + 1)):
            underj = under_input[:, j - 1, :, :, :]
            figure[30 + h:30 + h + h, 10 + (w + 20) * j: 10 + (w + 20) * j + w] = underj[i, 0, :, :] * 255

        if save_feature:
            cv2.imwrite(out_dir + '/' + 'input' + '.png', figure)

        cv2.imshow("attention-" + str(c), figure)
        cv2.waitKey(0)


# Global layer counter for tracking position in network
global layer
layer = 0


def attentionheatmap_visual(features,
                            out_dir='./Visualization/attention_af3/',  # Directory to save feature maps
                            save_feature=True,  # Whether to save feature maps as images
                            show_feature=True,  # Whether to display feature maps using plt
                            feature_title=None,  # Title for feature maps (defaults to shape)
                            channel=None,
                            ):
    """
    Visualize attention heatmaps.

    Args:
        features: Feature tensor of shape [B, C, H, W]
        out_dir: Output directory for visualizations
        save_feature: Whether to save visualization as image files
        show_feature: Whether to show the feature maps
        feature_title: Title for the visualization
        channel: Specific channel to visualize (if None, visualize all)
    """
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    global layer
    b, c, h, w = features.shape

    if b > 1:
        b = 1  # Limit to displaying only the first sample

    for i in range(b):
        for j in range(c):
            featureij = features[i, j, :, :]
            featureij = featureij.cpu().detach().numpy()

            # Create heatmap visualization with coolwarm colormap
            # vmin=-0.01, vmax=0.01 sets the color scale range
            fig = sns.heatmap(featureij, cmap="coolwarm", vmin=-0.01, vmax=0.01)

            # Remove axis ticks for cleaner visualization
            fig.set_xticks(range(0))
            fig.set_yticks(range(0))

            plt.show()
            plt.close()

            fig_heatmap = fig.get_figure()
            imgpath = read_img_name()
            filename = os.path.basename(imgpath)
            filename = filename.split('.')[0]

            # Save heatmap with layer and channel information in filename
            fig_heatmap.savefig(os.path.join(out_dir, filename + '_l' + str(layer) + '_' + str(j) + '.png'))

    # Increment layer counter and wrap around at 12
    layer = (layer + 1) % (12)


def attentionheatmap_visual3(features,
                             out_dir='./Visualization/attention_af3/',  # Directory to save feature maps
                             save_feature=True,  # Whether to save feature maps as images
                             show_feature=True,  # Whether to display feature maps using plt
                             feature_title=None,  # Title for feature maps (defaults to shape)
                             channel=None,
                             ):
    """
    Alternative attention heatmap visualization with 16 layers cycling.

    Args:
        features: Feature tensor of shape [B, C, H, W]
        out_dir: Output directory for visualizations
        save_feature: Whether to save visualization as image files
        show_feature: Whether to show the feature maps
        feature_title: Title for the visualization
        channel: Specific channel to visualize (if None, visualize all)
    """
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    global layer
    b, c, h, w = features.shape

    if b > 1:
        b = 1  # Limit to displaying only the first sample

    for i in range(b):
        for j in range(c):
            featureij = features[i, j, :, :]
            featureij = featureij.cpu().detach().numpy()

            # Create heatmap visualization with coolwarm colormap
            fig = sns.heatmap(featureij, cmap="coolwarm", vmin=-0.01, vmax=0.01)

            # Remove axis ticks for cleaner visualization
            fig.set_xticks(range(0))
            fig.set_yticks(range(0))

            plt.show()
            plt.close()

            fig_heatmap = fig.get_figure()
            imgpath = read_img_name()
            filename = os.path.basename(imgpath)
            filename = filename.split('.')[0]

            # Save heatmap with layer and channel information in filename
            fig_heatmap.savefig(os.path.join(out_dir, filename + '_l' + str(layer) + '_' + str(j) + '.png'))

    # Increment layer counter and wrap around at 16
    layer = (layer + 1) % (16)


def attentionheatmap_visual2(features, sita,
                             out_dir='./Visualization/attention_af3/',  # Directory to save feature maps
                             value=0.05,  # Value range for heatmap visualization
                             save_feature=True,  # Whether to save feature maps as images
                             show_feature=True,  # Whether to display feature maps using plt
                             feature_title=None,  # Title for feature maps (defaults to shape)
                             channel=None,
                             ):
    """
    Attention heatmap visualization with custom value range and sita parameter.

    Args:
        features: Feature tensor of shape [B, C, H, W]
        sita: Parameter values to include in filenames
        out_dir: Output directory for visualizations
        value: Range value for visualization scaling
        save_feature: Whether to save visualization as image files
        show_feature: Whether to show the feature maps
        feature_title: Title for the visualization
        channel: Specific channel to visualize (if None, visualize all)
    """
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    global layer
    b, c, h, w = features.shape

    if b > 1:
        b = 1  # Limit to displaying only the first sample

    for i in range(b):
        for j in range(c):
            featureij = features[i, j, :, :]
            featureij = featureij.cpu().detach().numpy()

            # Create heatmap visualization with coolwarm colormap and custom value range
            fig = sns.heatmap(featureij, cmap="coolwarm", vmin=-value, vmax=value)

            # Remove axis ticks for cleaner visualization
            fig.set_xticks(range(0))
            fig.set_yticks(range(0))

            plt.show()
            plt.close()

            fig_heatmap = fig.get_figure()
            imgpath = read_img_name()
            filename = os.path.basename(imgpath)
            filename = filename.split('.')[0]

            # Save heatmap with layer, channel, and sita value in filename
            fig_heatmap.savefig(
                os.path.join(out_dir, filename + '_l' + str(layer) + '_' + str(j) + '_' + str(sita[j].item()) + '.png'))

    # Increment layer counter and wrap around at 12
    layer = (layer + 1) % (12)


def visual_segmentation(seg, image_filename, opt):
    """
    Visualize segmentation results by overlaying colored segments on original image.

    Args:
        seg: Segmentation tensor/array
        image_filename: Name of the image file
        opt: Options containing paths and parameters
    """
    # Load original image
    img_ori = cv2.imread(os.path.join(opt.data_path + '/img', image_filename))
    img_ori0 = cv2.imread(os.path.join(opt.data_path + '/img', image_filename))

    # Initialize overlay
    overlay = img_ori * 0
    img_r = img_ori[:, :, 0]
    img_g = img_ori[:, :, 1]
    img_b = img_ori[:, :, 2]

    # Color table for different segments
    table = np.array([[193, 182, 255], [219, 112, 147], [237, 149, 100], [211, 85, 186], [204, 209, 72],
                      [144, 255, 144], [0, 215, 255], [96, 164, 244], [128, 128, 240], [250, 206, 135]])

    seg0 = seg[0, :, :]

    # Apply colors to different segments
    for i in range(1, opt.classes):
        # img_r[seg0 == i] = table[i - 1, 0]
        # img_g[seg0 == i] = table[i - 1, 1]
        # img_b[seg0 == i] = table[i - 1, 2]
        img_r[seg0 == i] = table[i + 1 - 1, 0]
        img_g[seg0 == i] = table[i + 1 - 1, 1]
        img_b[seg0 == i] = table[i + 1 - 1, 2]

    # Combine channels to create overlay
    overlay[:, :, 0] = img_r
    overlay[:, :, 1] = img_g
    overlay[:, :, 2] = img_b
    overlay = np.uint8(overlay)

    # Blend original image with overlay
    # img = cv2.addWeighted(img_ori0, 0.6, overlay, 0.4, 0)  # ACDC
    img = cv2.addWeighted(img_ori0, 0.5, overlay, 0.5, 0)  # ISIC
    # img = np.uint8(0.3 * overlay + 0.7 * img_ori)

    # Save visualization result
    fulldir = opt.visual_result_path + "/" + opt.modelname + "/"
    if not os.path.isdir(fulldir):
        os.makedirs(fulldir)
    cv2.imwrite(fulldir + image_filename, img)


def visual_segmentation_binary(seg, image_filename, opt):
    """
    Visualize binary segmentation results by creating a white mask on black background.

    Args:
        seg: Binary segmentation tensor/array
        image_filename: Name of the image file
        opt: Options containing paths and parameters
    """
    # Load original image
    img_ori = cv2.imread(os.path.join(opt.data_path + '/img', image_filename))
    img_ori0 = cv2.imread(os.path.join(opt.data_path + '/img', image_filename))

    # Initialize overlay
    overlay = img_ori * 0
    img_r = img_ori[:, :, 0]
    img_g = img_ori[:, :, 1]
    img_b = img_ori[:, :, 2]

    seg0 = seg[0, :, :]

    # Create white overlay for all segments
    for i in range(1, opt.classes):
        img_r[seg0 == i] = 255
        img_g[seg0 == i] = 255
        img_b[seg0 == i] = 255

    # Combine channels to create overlay
    overlay[:, :, 0] = img_r
    overlay[:, :, 1] = img_g
    overlay[:, :, 2] = img_b
    overlay = np.uint8(overlay)

    # Save visualization result
    fulldir = opt.visual_result_path + "/" + opt.modelname + "/"
    if not os.path.isdir(fulldir):
        os.makedirs(fulldir)
    cv2.imwrite(fulldir + image_filename, overlay)