print("Importing Modules")

import time
import os
import socket
import numpy as np
from   scipy.ndimage import zoom
import matplotlib.pyplot as plt
import docx
from   docx.shared import Inches

from   monai.transforms import (Spacingd, Compose, LoadImaged, EnsureChannelFirstd, RandSpatialCropd, CenterSpatialCropd,\
                               RandRotate90d, ToTensord, ScaleIntensityRanged, SpatialPadd)
from   monai.networks.nets import UNet
from   monai.losses import DiceLoss
from   monai.data import Dataset, DataLoader, pad_list_data_collate
print("Monai imported. Importing Torch")

from   torch.optim import Adam

import pytorch_lightning as pl
from   pytorch_lightning.loggers import TensorBoardLogger
import torch
from torch.utils.tensorboard import SummaryWriter
# import tensorflow as tf
# from mpl_toolkits.mplot3d.art3d import Poly3DCollection
# from skimage import measure
# import nrrd
# import torchvision 
# from   docx.enum.text import WD_PARAGRAPH_ALIGNMENT


from monai.transforms import (
    Compose, LoadImaged, ScaleIntensityRanged,
    RandSpatialCropd, EnsureTyped, RandRotate90d
)
from monai.data import DataLoader, CacheDataset
from pytorch_lightning import Trainer, callbacks
from pytorch_lightning.callbacks import ModelCheckpoint

print("Creating Dataset and DataLoader Functions")

def create_dataset(data_dir):
    print(f"Creating dataset from directory: {data_dir}")
    data_dicts = []
    for filename in os.listdir(data_dir):
        if filename.endswith("_Vx3.nrrd"):
            image_path = os.path.join(data_dir, filename)
            label_filename = filename.replace("_Vx3.nrrd", "_Label.nrrd")
            label_path = os.path.join(data_dir, label_filename)
            data_dicts.append({'image': image_path, 'label': label_path})
    return data_dicts

def select_images(image_paths, num_select=5):
    selected_indices = np.linspace(0, len(image_paths) - 1, num_select, dtype=int)
    return [image_paths[i] for i in selected_indices]

def plot_losses(train_losses, val_losses):
    train_epochs, train_loss_values = zip(*train_losses)

    # Extract the validation loss values and their corresponding epochs
    val_epochs, val_loss_values = zip(*val_losses)

    plt.figure(figsize=(10, 5))
    plt.plot(train_epochs, train_loss_values, label='Train Loss')
    plt.plot(val_epochs, val_loss_values, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
    plt.legend()
    plt.grid(True, which='both', linestyle='-', linewidth=0.5, color="black")
    plt.minorticks_on()
    plt.grid(True, which='minor', linestyle='-', linewidth=0.3, color='gray')
    plt.savefig("./PNG/lossfunction.png")
    plt.close()

def plot_losses_log(train_losses, val_losses):
    train_epochs, train_loss_values = zip(*train_losses)

    # Extract the validation loss values and their corresponding epochs
    val_epochs, val_loss_values = zip(*val_losses)

    plt.figure(figsize=(10, 5))
    plt.plot(train_epochs, train_loss_values, label='Train Loss')
    plt.plot(val_epochs, val_loss_values, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.title('Training and Validation Losses (Log Scale)')
    plt.legend()
    plt.grid(True, which='both', linestyle='-', linewidth=0.5, color="black")
    plt.minorticks_on()
    plt.grid(True, which='minor', linestyle='-', linewidth=0.3, color='gray')
    plt.savefig("./PNG/log-lossfunction.png")
    plt.close()

def plot_learning_rate(learning_rate):
    # epochs = range(1, len(train_losses) + 1) 


    epochs, learning_rate = zip(*learning_rate)

    # Extract the validation loss values and their corresponding epochs
    
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, learning_rate, label='Learning Rate')
    plt.xlabel('Epochs')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Change Vs Epoch')
    plt.legend()
    plt.grid(True, which='both', linestyle='-', linewidth=0.5, color="black")
    plt.minorticks_on()
    plt.grid(True, which='minor', linestyle='-', linewidth=0.3, color='gray')
    plt.draw()
    plt.pause(0.001)
    plt.savefig("./PNG/learning_rate.png")
    plt.close()

def plot_log_learning_rate(learning_rate):
    # epochs = range(1, len(learning_rate) + 1) 
    # plt.figure(figsize=(10, 5))
    # plt.plot(learning_rate, label='Learning Rate')
    epochs, learning_rate = zip(*learning_rate)
    # Extract the validation loss values and their corresponding epochs
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, learning_rate, label='Learning Rate')
    plt.xlabel('Epochs')
    plt.ylabel('Learning Rate')
    plt.yscale('log')
    plt.title('Learning Rate Change Vs Epoch')
    plt.legend()
    plt.grid(True, which='both', linestyle='-', linewidth=0.5, color="black")
    plt.minorticks_on()
    plt.grid(True, which='minor', linestyle='-', linewidth=0.3, color='gray')
    plt.draw()
    plt.pause(0.001)
    plt.savefig("./PNG/log_learning_rate.png")
    plt.close()

def plot_3d_volume(ax, volume, color_map):
    colors = np.take(color_map, volume, axis=0) / 255.0
    ax.voxels(volume, facecolors=colors, edgecolor=None)

def visualize_and_save_train(inputs, outputs, labels, batch_indext, epoch, global_step, trainer):
    color_map = np.array([
        [255, 255, 255],  # White for background (0)
        [0, 151, 206],      # Blue for arteries (1)
        [216, 101, 79]       # Red for veins (2)
    ], dtype=np.uint8)
    saved_train_image_paths = []
    

    with open(md_file_path, "a") as md_file:
        for i in range(outputs.shape[0]):  # Iterate over the batch dimension
            if i < inputs.shape[0]:
                fig, ax = plt.subplots(1, 3, figsize=(15, 5))
                # Get the input image and normalize it for visualization
                input_slice = inputs[i, 0, :, :, :].detach().cpu().numpy()
                input_slice_mid = input_slice[input_slice.shape[0] // 2]  # Middle slice of the input volume
                # input_slice_mid_normalized = (input_slice_mid - clippingmin) / \
                #                             (clippingmax - clippingmin)
                # Get the label and prediction volumes
                label_volume = labels[i, 0].cpu().numpy().astype(np.uint8)
                output_volume = outputs[i].detach().cpu().numpy()
                predicted_labels_volume = np.argmax(output_volume, axis=0).astype(np.uint8)
            
                # Select the middle slice index after argmax reduction
                slice_idx = predicted_labels_volume.shape[0] // 2
                label_slice = label_volume[slice_idx]
                predicted_labels_slice = predicted_labels_volume[slice_idx]

                # Apply the color map to the label and prediction slices
                label_rgb_slice = color_map[label_slice]
                prediction_rgb_slice = color_map[predicted_labels_slice]

                ax[0].imshow(input_slice_mid, cmap='gray')
                ax[0].set_title("Input")
                ax[1].imshow(label_rgb_slice)
                ax[1].set_title("Ground Truth")
                ax[2].imshow(prediction_rgb_slice)
                ax[2].set_title("Prediction")
                
                fig.suptitle(f'Training Image: Epoch-{epoch}-Batch-{batch_indext}-Image-{global_step+i}', fontsize=16)
                img_path = f"./PNG/T-Epoch-{epoch}-Batch-{batch_indext}-Image-{global_step+i}.png"
                print(f"V-Epoch-{epoch}-Batch-{batch_indext}-Image-{i}.png")
                print(f"Saving Image to:  {img_path}")
                

                # Save the image
                plt.savefig(img_path)
                plt.close()
                saved_train_image_paths.append(img_path)
                md_file.write(f"![T-Epoch-{epoch}-Batch-{batch_indext}-Image-{global_step+i}.png]({img_path})\n\n")
                # global_image_counter += 1

    print(f"Markdown file updated at {md_file_path}")
    return saved_train_image_paths
def visualize_and_save_valid(inputs, outputs, labels, batch_indexv, epoch, global_step, trainer):
    color_map = np.array([
        [255, 255, 255],  # White for background (0)
        [0, 151, 206],      # Blue for arteries (1)
        [216, 101, 79]       # Red for veins (2)
    ], dtype=np.uint8)
    saved_valid_image_paths = []

    with open(md_file_path, "a") as md_file:
        for i in range(outputs.shape[0]):  # Iterate over the batch dimension
            if i < inputs.shape[0]:
                fig, ax = plt.subplots(1, 3, figsize=(15, 5))
           
                # Get the input image and normalize it for visualization
                input_slice = inputs[i, 0, :, :, :].detach().cpu().numpy()
                input_slice_mid = input_slice[input_slice.shape[0] // 2]  # Middle slice of the input volume
                label_volume = labels[i, 0].cpu().numpy().astype(np.uint8)
                output_volume = outputs[i].detach().cpu().numpy()
                predicted_labels_volume = np.argmax(output_volume, axis=0).astype(np.uint8)


                #Select the middle slice index after argmax reduction
                slice_idx = predicted_labels_volume.shape[0] // 2
                label_slice = label_volume[slice_idx]
                predicted_labels_slice = predicted_labels_volume[slice_idx]

                # Apply the color map to the label and prediction slices
                label_rgb_slice = color_map[label_slice]
                prediction_rgb_slice = color_map[predicted_labels_slice]

                ax[0].imshow(input_slice_mid, cmap='gray')
                ax[0].set_title("Input")
                ax[1].imshow(label_rgb_slice)
                ax[1].set_title("Ground Truth")
                ax[2].imshow(prediction_rgb_slice)
                ax[2].set_title("Prediction")
                
                fig.suptitle(f'Validation Image: Epoch-{epoch}-Batch-{batch_indexv}-Image-{global_step+i}', fontsize=16)
                img_path = f"./PNG/V-Epoch-{epoch}-Batch-{batch_indexv}-Image-{global_step+i}.png"
                print(f"V-Epoch-{epoch}-Batch-{batch_indexv}-Image-{i}.png")
                print(f"Saving Image to:  {img_path}")
                # Save the image
                plt.savefig(img_path)
                plt.close()
                saved_valid_image_paths.append(img_path)
                # Write to the markdown file
                md_file.write(f"![V-Epoch-{epoch}-Batch-{batch_indexv}-Image-{global_step+i}.png]({img_path})\n\n")

    print(f"Markdown file updated at {md_file_path}")

    return saved_valid_image_paths
def visualize_and_save_train_3d(inputs, outputs, labels, batch_indext, epoch, global_step, trainer):
    color_map = np.array([
        [255, 255, 255],  # White for background (0)
        [0, 151, 206],      # Blue for arteries (1)
        [216, 101, 79]       # Red for veins (2)
    ], dtype=np.uint8)
    saved_train_image_paths = []

    with open(md_file_path, "a") as md_file:
        for i in range(outputs.shape[0]):  # Iterate over the batch dimension
            if i < inputs.shape[0]:
                fig, ax = plt.subplots(1, 5, figsize=(25, 5))
                # Get the input image and normalize it for visualization
                input_slice = inputs[i, 0, :, :, :].detach().cpu().numpy()
                input_slice_mid = input_slice[input_slice.shape[0] // 2]  # Middle slice of the input volume
                # input_slice_mid_normalized = (input_slice_mid - clippingmin) / \
                #                             (clippingmax - clippingmin)
                zoom_factor = 0.5
                # Get the label and prediction volumes
                label_volume = labels[i, 0].cpu().numpy().astype(np.uint8)
                output_volume = outputs[i].detach().cpu().numpy()
                predicted_labels_volume = np.argmax(output_volume, axis=0).astype(np.uint8)
            
                # Select the middle slice index after argmax reduction
                slice_idx = predicted_labels_volume.shape[0] // 2
                label_slice = label_volume[slice_idx]
                predicted_labels_slice = predicted_labels_volume[slice_idx]

                downsampled_label = zoom(label_volume, zoom_factor, order=0)
                ax[3] = fig.add_subplot(1, 5, 4, projection='3d')
                

                ax[3].set_title("3D Ground Truth")
                plot_3d_volume(ax[3], downsampled_label, color_map)
                downsampled_prediction = zoom(predicted_labels_volume, zoom_factor, order=0)
                ax[4] = fig.add_subplot(1, 5, 5, projection='3d')
                
                ax[4].set_title("3D Prediction")
                plot_3d_volume(ax[4], downsampled_prediction, color_map)
                # non_zero_coords = np.argwhere(downsampled_label)
                # min_coords = non_zero_coords.min(axis=0) * 1/zoom_factor
                # max_coords = non_zero_coords.max(axis=0) * 1/zoom_factor
                # ax[3].set_xlim(min_coords[0], max_coords[0])
                # ax[3].set_ylim(min_coords[1], max_coords[1])
                # ax[3].set_zlim(min_coords[2], max_coords[2])
                # ax[4].set_xlim(min_coords[0], max_coords[0])
                # ax[4].set_ylim(min_coords[1], max_coords[1])
                # ax[4].set_zlim(min_coords[2], max_coords[2])
                # Remove the labels and ticks for the 3D ground truth plot
                ax[3].set_xticklabels([])
                ax[3].set_yticklabels([])
                ax[3].set_zticklabels([])
                # Remove the labels and ticks for the 3D prediction plot
                ax[4].set_xticklabels([])
                ax[4].set_yticklabels([])
                ax[4].set_zticklabels([])

                # Apply the color map to the label and prediction slices
                label_rgb_slice = color_map[label_slice]
                prediction_rgb_slice = color_map[predicted_labels_slice]

                ax[0].imshow(input_slice_mid, cmap='gray')
                ax[0].set_title("Input")
                ax[1].imshow(label_rgb_slice)
                ax[1].set_title("Ground Truth")
                ax[2].imshow(prediction_rgb_slice)
                ax[2].set_title("Prediction")
                
                fig.suptitle(f'Training Image: Epoch-{epoch}-Batch-{batch_indext}-Image-{i}', fontsize=16)
                # img_path = f"./PNG/T-Epoch-{epoch}-Batch-{batch_indext}-Image-{global_step+i}.png"
                img_path = f"./PNG/T-Epoch-{epoch}-Batch-{batch_indext}-Image-{i}.png"
                print(f"T-Epoch-{epoch}-Batch-{batch_indext}-Image-{i}.png SAVED!")

                # Save the image
                plt.savefig(img_path)
                plt.close()
                saved_train_image_paths.append(img_path)
                md_file.write(f"![T-Epoch-{epoch}-Batch-{batch_indext}-Image-{global_step+i}.png]({img_path})\n\n")

    print(f"Markdown file updated at {md_file_path}")
    return saved_train_image_paths
def visualize_and_save_valid_3d(inputs, outputs, labels, batch_indexv, epoch, global_step, trainer):
    color_map = np.array([
        [255, 255, 255],  # White for background (0)
        [0, 151, 206],      # Blue for arteries (1)
        [216, 101, 79]       # Red for veins (2)
    ], dtype=np.uint8)
    saved_valid_image_paths = []

    with open(md_file_path, "a") as md_file:
        for i in range(outputs.shape[0]):  # Iterate over the batch dimension
            if i < inputs.shape[0]:
                fig, ax = plt.subplots(1, 5, figsize=(25, 5))
                # Get the input image and normalize it for visualization
                input_slice = inputs[i, 0, :, :, :].detach().cpu().numpy()
                input_slice_mid = input_slice[input_slice.shape[0] // 2]  # Middle slice of the input volume
                # input_slice_mid_normalized = (input_slice_mid - clippingmin) / \
                #                             (clippingmax - clippingmin)
                zoom_factor = 0.5
                # Get the label and prediction volumes
                label_volume = labels[i, 0].cpu().numpy().astype(np.uint8)
                output_volume = outputs[i].detach().cpu().numpy()
                predicted_labels_volume = np.argmax(output_volume, axis=0).astype(np.uint8)
            
                # Select the middle slice index after argmax reduction
                slice_idx = predicted_labels_volume.shape[0] // 2
                label_slice = label_volume[slice_idx]
                predicted_labels_slice = predicted_labels_volume[slice_idx]

                downsampled_label = zoom(label_volume, zoom_factor, order=0)
                ax[3] = fig.add_subplot(1, 5, 4, projection='3d')
                

                ax[3].set_title("3D Ground Truth")
                plot_3d_volume(ax[3], downsampled_label, color_map)
                downsampled_prediction = zoom(predicted_labels_volume, zoom_factor, order=0)
                ax[4] = fig.add_subplot(1, 5, 5, projection='3d')
                
                ax[4].set_title("3D Prediction")
                plot_3d_volume(ax[4], downsampled_prediction, color_map)
                # non_zero_coords = np.argwhere(downsampled_label)
                # min_coords = non_zero_coords.min(axis=0) * 1/zoom_factor
                # max_coords = non_zero_coords.max(axis=0) * 1/zoom_factor
                # ax[3].set_xlim(min_coords[0], max_coords[0])
                # ax[3].set_ylim(min_coords[1], max_coords[1])
                # ax[3].set_zlim(min_coords[2], max_coords[2])
                # ax[4].set_xlim(min_coords[0], max_coords[0])
                # ax[4].set_ylim(min_coords[1], max_coords[1])
                # ax[4].set_zlim(min_coords[2], max_coords[2])
                # Remove the labels and ticks for the 3D ground truth plot
                ax[3].set_xticklabels([])
                ax[3].set_yticklabels([])
                ax[3].set_zticklabels([])
                # Remove the labels and ticks for the 3D prediction plot
                ax[4].set_xticklabels([])
                ax[4].set_yticklabels([])
                ax[4].set_zticklabels([])

                # Apply the color map to the label and prediction slices
                label_rgb_slice = color_map[label_slice]
                prediction_rgb_slice = color_map[predicted_labels_slice]

                ax[0].imshow(input_slice_mid, cmap='gray')
                ax[0].set_title("Input")
                ax[1].imshow(label_rgb_slice)
                ax[1].set_title("Ground Truth")
                ax[2].imshow(prediction_rgb_slice)
                ax[2].set_title("Prediction")

                fig.suptitle(f'Validation Image: Epoch-{epoch}-Batch-{batch_indexv}-Image-{i}', fontsize=16)
                # img_path = f"./PNG/V-Epoch-{epoch}-Batch-{batch_indexv}-Image-{global_step+i}.png"
                img_path = f"./PNG/V-Epoch-{epoch}-Batch-{batch_indexv}-Image-{i}.png"
                print(f"V-Epoch-{epoch}-Batch-{batch_indexv}-Image-{i} SAVED!.png")
                print(f"Saving Image to:  {img_path}")
                # Save the image
                plt.savefig(img_path)
                plt.close()
                saved_valid_image_paths.append(img_path)
                # Write to the markdown file
                md_file.write(f"![V-Epoch-{epoch}-Batch-{batch_indexv}-Image-{global_step+i}.png]({img_path})\n\n")

    print(f"Markdown file updated at {md_file_path}")
    return saved_valid_image_paths
def create_word_document(expdesc, selected_train_images, selected_val_images, log_loss_path, learning_rate_path, log_learning_rate_path):
    ########################################################################
    parent_directory = os.path.dirname(os.getcwd())
    output_path = os.path.join(parent_directory, 'JobReport.docx')

    # Check if the file exists. If yes, open the existing file; if no, create a new one
    if os.path.exists(output_path):
        doc = docx.Document(output_path)
        print(f"Trying to append to existing document: {output_path}")
    else:
        doc = docx.Document()
        print(f"Creating new document: {output_path}")
    print("Appending Images Now")
    # Append content to the document
    doc.add_paragraph("Experiment Description:")
    doc.add_paragraph(expdesc)
    # doc.add_paragraph(final_time)
    doc.add_picture(log_loss_path, width=Inches(6))
    doc.add_picture(learning_rate_path, width=Inches(6))
    doc.add_picture(log_learning_rate_path, width=Inches(6))
    for img_path in selected_train_images + selected_val_images:
        doc.add_picture(img_path, width=Inches(6))
    # for img_path in selected_train_images:
        # doc.add_picture(img_path, width=Inches(6))

    # Save the document
    doc.save(output_path)

    # Print the path where the file was updated or created
    print(f"Document updated/created: {output_path}")
    ############################################################################



md_file_path = "./visualization.md"
hostname = socket.gethostname()

expdesc = (f"Job 10: \n Copy of Job 9 \n Train Images: 16 \n Validation Images: 16 \n Train Crop: RandSpatialCrop \n \
Validation Crop: RandSpatialCrop \n Epochs: 3500 \n Learning Rate: 6e-4 \n \
Spacing: (0.8, 0.8, 0.8) \n Batch Size: 4 \n Patch Size: (128, 128, 128) \n Compute Node: {hostname} /n NO DOWNSAMPLE TRANSFORM (NOT WORKING WITH LIGHTNING)") # REMEBER TO CHANGE THIS EACH JOB

with open('expdesc.txt', 'w') as file:
    file.write(expdesc)
log_loss_path          ="./PNG/log-lossfunction.png"
learning_rate_path     ="./PNG/learning_rate.png"
log_learning_rate_path ="./PNG/log_learning_rate.png"
train_data_dir = "../../TrainingData16V3/"
val_data_dir = "../../ValidationData16V3/"
model_save_path = "./model/Nate_Unet.pth"
optimizer_save_path = "./model/Nate_Unet_optimizer.pth"
train_image_paths = []
val_image_paths = []
train_losses = []
val_losses = []
train_image_batch_paths = []
val_image_batch_paths = []
learning_rates = [] 
starttime = time.time()
# word_output_path = "./JobReport.docx"
NGPUS = 1 ############################################## 1
# Main training loop
# num_epochs = 6
display_interval = 1
starttime = time.time()
print("Defining the create_dataset function...")
# Function to create dataset
print("Setting data paths...")
# Set data paths
print("Creating datasets...")

train_files = create_dataset(train_data_dir)
val_files   = create_dataset(val_data_dir)

# Define the size of the cropped region
# roi_size = (192, 192, 192) ############################################################### 3
# roi_size = (64, 64, 64)
roi_size = (128, 128, 128)
print("Defining transformations...")
# Transformations
print(f"Elapsed: {(time.time()-starttime)/60:.2f} Minutes")

downsampling_transform = Spacingd(
    keys=['image', 'label'], 
    pixdim=(0.8, 0.8, 0.8), 
    mode=('bilinear', 'nearest') ############################################################### 1
)

train_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    downsampling_transform, ############################################################### 2
    # AddChanneld(keys=["image", "label"]),
    RandSpatialCropd(keys=["image", "label"], roi_size=roi_size, random_size=False), ################################### 4
    RandRotate90d   (keys=["image", "label"], prob=0.5),
    ScaleIntensityRanged(keys=["image"], a_min=-1100, a_max=3000, b_min=-1.0, b_max=1.0, clip=True),
    EnsureTyped(keys=["image", "label"]),
])

val_transforms = Compose([
    LoadImaged(keys=['image', 'label']),
    EnsureChannelFirstd(keys=['image', 'label']),
    downsampling_transform,
    # SpatialPadd(keys=['image', 'label'], spatial_size=roi_size, method='symmetric'),  # Add padding here
    RandSpatialCropd(keys=['image', 'label'], roi_size=roi_size),
    ScaleIntensityRanged(keys=['image'], a_min=-1100, a_max=3000, b_min=-1.0, b_max=1.0, clip=True),
    EnsureTyped(keys=["image", "label"]),
])

print("Initializing data loaders...")

train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_rate=1.0, num_workers=4)
val_ds   = CacheDataset(data=val_files  , transform=  val_transforms, cache_rate=1.0, num_workers=4) ###################### 5
train_loader = DataLoader(train_ds, batch_size=4, shuffle=True , num_workers=4) ################################ 6&7
val_loader   = DataLoader(val_ds  , batch_size=4, shuffle=False, num_workers=4)

print("Initializing U-Net model...")
class UNetModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        # self.global_image_counter = 0
        # self.global_image_counter = 0
        self.model = UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=3,  # For two objects + background
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
        )

        self.loss_fn = DiceLoss(to_onehot_y=True, softmax=True) ############################################ 8
        self.all_epochs = []
        # self.loss_fn = FocalLoss(
        #     include_background=True, 
        #     to_onehot_y=True, 
        #     gamma=2.0, 
        #     reduction="mean",
        #     use_softmax=True
        # )

        # torch.nn.CrossEntropyLoss()
        # FocalLoss(
        #     include_background=True, 
        #     to_onehot_y=True, 
        #     gamma=2.0, 
        #     reduction="mean",
        #     use_softmax=True
        # )

    def forward(self, x):
        x = self.model(x)
        return x
    
    def configure_optimizers(self):
        return Adam(self.parameters(), lr=6e-4) ###################################### 9

    def training_step(self, batchT, batch_idxT):
        # x, y = batch
        imagesT, targetsT = batchT["image"], batchT["label"]
        outputsT = self(imagesT)
        loss = self.loss_fn(outputsT, targetsT)
        self.log("train_loss", loss)
        
        current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        current_epoch = self.current_epoch
        learning_rates.append((current_epoch, current_lr))
        train_losses.append((current_epoch, loss.item()))
        if self.current_epoch % 100 == 0 and self.current_epoch != 0:
            train_image_batch_paths = visualize_and_save_train_3d(imagesT, outputsT, targetsT, batch_idxT, self.current_epoch, self.global_step, self.trainer)
            train_image_paths.extend(train_image_batch_paths)
        print(f"Elapsed: {(time.time()-starttime)/60:.2f} Minutes")
        # train_image_batch_paths = visualize_and_save_train(images, outputs, targets, batch_idx, self.current_epoch, self.global_step, self.trainer)
        # train_image_paths.extend(train_image_batch_paths)
        # if batch_idx % 2 == 0:
        #     x=x[:5]
        #     grid = torchvision.utils.make_grid(x.view(-1, 1, 28, 28))
        #     self.logger.experiment.add_image("Train data", grid, self.global_step)
        # train_image_batch_paths, global_image_counter = visualize_and_save_train(images, outputs, targets, batch_idx, self.current_epoch, global_image_counter)
        # train_image_batch_paths = visualize_and_save_train_3d(images, outputs, targets, batch_idx, self.current_epoch)
        # train_image_paths.extend(train_image_batch_paths)
        return loss
    
    def validation_step(self, batchV, batch_idxV):
        imagesV, targetsV = batchV["image"], batchV["label"]
        outputsV = self(imagesV)
        loss = self.loss_fn(outputsV, targetsV)
        self.log("val_loss", loss, prog_bar=True)
        current_epoch = self.current_epoch
        val_losses.append((current_epoch, loss.item()))
        if self.current_epoch % 100 == 0 and self.current_epoch != 0:
            # val_image_batch_paths = visualize_and_save_valid(images, outputs, targets, batch_idx, self.current_epoch, self.global_step, self.trainer)
            val_image_batch_paths = visualize_and_save_valid_3d(imagesV, outputsV, targetsV, batch_idxV, self.current_epoch, self.global_step, self.trainer)
            val_image_paths.extend(val_image_batch_paths)
        print(f"Elapsed: {(time.time()-starttime)/60:.2f} Minutes")
        # val_image_batch_paths = visualize_and_save_valid(images, outputs, targets, batch_idx, self.current_epoch, self.global_step, self.trainer)
        # val_image_paths.extend(val_image_batch_paths)
        if self.current_epoch % 10 == 0 and self.current_epoch != 0:
            torch.save(model.state_dict(), "./model/Model.pth")
        return loss
    
    def on_epoch_start(self):
        # Print the current epoch number at the start of each epoch
        print(f"Epoch: {self.current_epoch}")
        # Append the current epoch to the list
        print(f"Elapsed: {(time.time()-starttime)/60:.2f} Minutes")
        self.all_epochs.append(self.current_epoch)

    def on_train_end(self):
        # Print the list of all epochs after training ends
        print(f"Elapsed: {(time.time()-starttime)/60:.2f} Minutes")
        print(f"Completed Epochs: {self.all_epochs}")


class PlotLossCallback(pl.Callback):
    def __init__(self, log_dir):
        super().__init__()
        self.log_dir = log_dir

    def on_train_start(self, trainer, pl_module):
        # Clear previous graphs
        # tf.compat.v1.reset_default_graph()
        self.writer = SummaryWriter(self.log_dir)

    def on_train_epoch_end(self, trainer, pl_module, **kwargs):
        epoch = trainer.current_epoch
        train_loss = trainer.callback_metrics['train_loss'].cpu().item()
        val_loss = trainer.callback_metrics['val_loss'].cpu().item()

        # Write both training and validation loss to the same graph with epoch as global step
        self.writer.add_scalars('Trial Six Tensorboard', {'training' : train_loss, 
                                                   'validation' : val_loss }, global_step=epoch +1 )
    def on_train_end(self, trainer, pl_module):
        self.writer.close()

trainer = Trainer(
    accelerator="gpu", devices=NGPUS,
    max_epochs=100,
    log_every_n_steps=1,
    callbacks=[
        callbacks.ModelCheckpoint(monitor="val_loss"),
        callbacks.ModelCheckpoint(monitor="train_loss"),
        PlotLossCallback("./logs/"),
        # Add additional callbacks for logging and visualization as needed
    ],
    logger=True,  # Enables TensorBoard and MLFlow logging by default
)
print(f"Elapsed: {(time.time()-starttime)/60:.2f} Minutes")
model = UNetModel()
trainer.fit(model, train_loader, val_loader)
print(f"Elapsed: {(time.time()-starttime)/60:.2f} Minutes")
# plot_losses(model.epoch_numbers, train_losses, val_losses)
# plot_losses_log(model.epoch_numbers, train_losses, val_losses)
plot_losses(train_losses, val_losses)
plot_losses_log(train_losses, val_losses)
plot_learning_rate(learning_rates)
plot_log_learning_rate(learning_rates)
# selected_train_images = select_images(train_image_paths)
# selected_val_images = select_images(val_image_paths)
# selected_train_images = select_images(train_image_paths)
# selected_val_images = select_images(val_image_paths)
# if trainer.global_rank == 0:
#     # Code here will only be executed by the process with rank 0
#     print("This is executed only once, by the master process.")
#     create_word_document(expdesc, selected_train_images, selected_val_images, log_loss_path, learning_rate_path, log_learning_rate_path)
#     print("gang gang gang gang gang gang gang gang gang gang")
# # create_word_document(expdesc, selected_train_images, selected_val_images, log_loss_path, learning_rate_path, log_learning_rate_path) #final_time)
# print("done")
print(f"Elapsed: {(time.time()-starttime)/60:.2f} Minutes")