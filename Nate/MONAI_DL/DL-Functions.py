


print("Importing Modules")
import os
from monai.transforms import (Spacingd, Compose, LoadImaged, EnsureChannelFirstd, RandSpatialCropd, CenterSpatialCropd,\
                               RandRotate90d, ToTensord, ScaleIntensityRanged, SpatialPadd)
from monai.networks.nets import UNet
from monai.losses import DiceLoss
from monai.data import Dataset, DataLoader, pad_list_data_collate
print("Monai imported. Importing Torch")
import matplotlib.pyplot as plt
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
import time 
import numpy as np
import docx
from docx.shared import Inches
from docx.enum.text import WD_PARAGRAPH_ALIGNMENT
import socket
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure
from scipy.ndimage import zoom
import nrrd



def create_dataset(data_dir):
    data_dicts = []
    for filename in os.listdir(data_dir):
        if filename.endswith("_Vx3.nrrd"):
            image_path = os.path.join(data_dir, filename)
            label_filename = filename.replace("_Vx3.nrrd", "_Label.nrrd")
            label_path = os.path.join(data_dir, label_filename)
            data_dicts.append({'image': image_path, 'label': label_path})
    return data_dicts

def plot_3d_object(ax, data, value, color, opacity):
    zoom_factor = 0.5
    mask = data == value
    try:
        verts, faces, _, _ = measure.marching_cubes(mask, level=0)
        verts *= 1 / zoom_factor  # Adjust for downsampling
        mesh = Poly3DCollection(verts[faces], alpha=opacity)
        mesh.set_facecolor(color)
        ax.add_collection3d(mesh)
    except RuntimeError as e:
        print(f"An error occurred: {e}")
        
def visualize_and_save_train(inputs, outputs, labels, batch_indext, epoch):
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
                
                fig.suptitle(f'Training Image: Epoch-{epoch}-Batch-{batch_indext}-Image-{i}', fontsize=16)
                img_path = f"./PNG/T-Epoch-{epoch}-Batch-{batch_indext}-Image-{i}.png"
                print(f"T-Epoch-{epoch}-Batch-{batch_indext}-Image-{i}.png")

                # Save the image
                plt.savefig(img_path)
                plt.close()
                saved_train_image_paths.append(img_path)
                md_file.write(f"![T-Epoch-{epoch}-Batch-{batch_indext}-Image-{i}.png]({img_path})\n\n")

    print(f"Markdown file updated at {md_file_path}")
    return saved_train_image_paths

def visualize_and_save_valid(inputs, outputs, labels, batch_indexv, epoch):
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
                
                fig.suptitle(f'Validation Image: Epoch-{epoch}-Batch-{batch_indexv}-Image-{i}', fontsize=16)
                img_path = f"./PNG/V-Epoch-{epoch}-Batch-{batch_indexv}-Image-{i}.png"
                print(f"V-Epoch-{epoch}-Batch-{batch_indexv}-Image-{i}.png")
                # Save the image
                plt.savefig(img_path)
                plt.close()
                saved_valid_image_paths.append(img_path)
                # Write to the markdown file
                md_file.write(f"![V-Epoch-{epoch}-Batch-{batch_indexv}-Image-{i}.png]({img_path})\n\n")

    print(f"Markdown file updated at {md_file_path}")

    return saved_valid_image_paths

def visualize_and_save_train_3d(inputs, outputs, labels, batch_indext, epoch):
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
                plot_3d_object(ax[3], downsampled_label, 1, (0, 0.592157, 0.807843), 1)  # Example values
                plot_3d_object(ax[3], downsampled_label, 2, (0.847059, 0.396078, 0.309804), 1)
                ax[3].set_title("3D Ground Truth")
                downsampled_prediction = zoom(predicted_labels_volume, zoom_factor, order=0)
                ax[4] = fig.add_subplot(1, 5, 5, projection='3d')
                plot_3d_object(ax[4], downsampled_prediction, 1, (0, 0.592157, 0.807843), 1)
                plot_3d_object(ax[4], downsampled_prediction, 2, (0.847059, 0.396078, 0.309804), 1)
                ax[4].set_title("3D Prediction")
                non_zero_coords = np.argwhere(downsampled_label)
                min_coords = non_zero_coords.min(axis=0) * 1/zoom_factor
                max_coords = non_zero_coords.max(axis=0) * 1/zoom_factor
                ax[3].set_xlim(min_coords[0], max_coords[0])
                ax[3].set_ylim(min_coords[1], max_coords[1])
                ax[3].set_zlim(min_coords[2], max_coords[2])
                ax[4].set_xlim(min_coords[0], max_coords[0])
                ax[4].set_ylim(min_coords[1], max_coords[1])
                ax[4].set_zlim(min_coords[2], max_coords[2])
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
                img_path = f"./PNG/T-Epoch-{epoch}-Batch-{batch_indext}-Image-{i}.png"
                print(f"T-Epoch-{epoch}-Batch-{batch_indext}-Image-{i}.png")

                # Save the image
                plt.savefig(img_path)
                plt.close()
                saved_train_image_paths.append(img_path)
                md_file.write(f"![T-Epoch-{epoch}-Batch-{batch_indext}-Image-{i}.png]({img_path})\n\n")

    print(f"Markdown file updated at {md_file_path}")
    return saved_train_image_paths

def visualize_and_save_valid_3d(inputs, outputs, labels, batch_indexv, epoch):
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
                zoom_factor = 0.5
               
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


                downsampled_label = zoom(label_volume, zoom_factor, order=0)
                ax[3] = fig.add_subplot(1, 5, 4, projection='3d')
                plot_3d_object(ax[3], downsampled_label, 1, (0, 0.592157, 0.807843), 1)  # Example values
                plot_3d_object(ax[3], downsampled_label, 2, (0.847059, 0.396078, 0.309804), 1)
                ax[3].set_title("3D Ground Truth")
                downsampled_prediction = zoom(predicted_labels_volume, zoom_factor, order=0)
                ax[4] = fig.add_subplot(1, 5, 5, projection='3d')
                plot_3d_object(ax[4], downsampled_prediction, 1, (0, 0.592157, 0.807843), 1)
                plot_3d_object(ax[4], downsampled_prediction, 2, (0.847059, 0.396078, 0.309804), 1)
                ax[4].set_title("3D Prediction")
                non_zero_coords = np.argwhere(downsampled_label)
                min_coords = non_zero_coords.min(axis=0) * 1/zoom_factor
                max_coords = non_zero_coords.max(axis=0) * 1/zoom_factor
                ax[3].set_xlim(min_coords[0], max_coords[0])
                ax[3].set_ylim(min_coords[1], max_coords[1])
                ax[3].set_zlim(min_coords[2], max_coords[2])
                ax[4].set_xlim(min_coords[0], max_coords[0])
                ax[4].set_ylim(min_coords[1], max_coords[1])
                ax[4].set_zlim(min_coords[2], max_coords[2])
                # Remove the labels and ticks for the 3D ground truth plot
                ax[3].set_xticklabels([])
                ax[3].set_yticklabels([])
                ax[3].set_zticklabels([])
                # Remove the labels and ticks for the 3D prediction plot
                ax[4].set_xticklabels([])
                ax[4].set_yticklabels([])
                ax[4].set_zticklabels([])



                ax[0].imshow(input_slice_mid, cmap='gray')
                ax[0].set_title("Input")
                ax[1].imshow(label_rgb_slice)
                ax[1].set_title("Ground Truth")
                ax[2].imshow(prediction_rgb_slice)
                ax[2].set_title("Prediction")
                
                fig.suptitle(f'Validation Image: Epoch-{epoch}-Batch-{batch_indexv}-Image-{i}', fontsize=16)
                img_path = f"./PNG/V-Epoch-{epoch}-Batch-{batch_indexv}-Image-{i}.png"
                print(f"V-Epoch-{epoch}-Batch-{batch_indexv}-Image-{i}.png")
                # Save the image
                plt.savefig(img_path)
                plt.close()
                saved_valid_image_paths.append(img_path)
                # Write to the markdown file
                md_file.write(f"![V-Epoch-{epoch}-Batch-{batch_indexv}-Image-{i}.png]({img_path})\n\n")

    print(f"Markdown file updated at {md_file_path}")

    return saved_valid_image_paths

def select_images(image_paths, num_select=5):
    selected_indices = np.linspace(0, len(image_paths) - 1, num_select, dtype=int)
    return [image_paths[i] for i in selected_indices]


def plot_losses(train_losses, val_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
    plt.legend()
    plt.grid(True, which='both', linestyle='-', linewidth=0.5, color="black")
    plt.minorticks_on()
    plt.grid(True, which='minor', linestyle='-', linewidth=0.3, color='gray')
    plt.draw()
    plt.pause(0.001)
    plt.savefig("./PNG/lossfunction.png")
    plt.close()

def plot_losses_log(train_losses, val_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.title('Training and Validation Losses')
    plt.legend()
    plt.grid(True, which='both', linestyle='-', linewidth=0.5, color="black")
    plt.minorticks_on()
    plt.grid(True, which='minor', linestyle='-', linewidth=0.3, color='gray')
    plt.draw()
    plt.pause(0.001)
    plt.savefig("./PNG/log-lossfunction.png")
    plt.close()

def plot_learning_rate(learning_rate):
    epochs = range(1, len(train_losses) + 1) 
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
    epochs = range(1, len(learning_rate) + 1) 
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


def train_step(batch_datat, model, loss_function, optimizer, device):
    # print("Reading in the images")
    # print(f"Processing filenames: {batch_data['filename']}")
    imagest, labelst = batch_datat['image'], batch_datat['label']
    imagest, labelst = imagest.to(device), labelst.to(device)
    optimizer.zero_grad()
    print("Passing Through the Network")
    outputst = model(imagest)

    print(f"Output shape: {outputst.shape}, Label shape: {labelst.shape}")
    print("Label tensor values (sample):", labelst[0, :, 0, 0, 0])
    # if torch.any(labels == 3):
        # print(f"Filename with label 3: {batch_data['filename']}")

    loss = loss_function(outputst, labelst)
    print("Loss Backward")
    loss.backward()
    print("Stepping Optimizer")
    optimizer.step()
    return loss.item(), outputst


def create_word_document(expdesc, log_loss_path, selected_train_images, selected_val_images, learning_rate_path, log_learning_rate_path, final_time):
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
    doc.add_paragraph(final_time)
    doc.add_picture(log_loss_path, width=Inches(6))
    doc.add_picture(learning_rate_path, width=Inches(6))
    doc.add_picture(log_learning_rate_path, width=Inches(6))
    for img_path in selected_train_images + selected_val_images:
        doc.add_picture(img_path, width=Inches(6))


    # Save the document
    doc.save(output_path)

    # Print the path where the file was updated or created
    print(f"Document updated/created: {output_path}")
    ############################################################################



if (epoch + 1) % 1 == 0:
    train_image_batch_paths = visualize_and_save_train(batch_datat['image'], output, batch_datat['label'], batch_indext, epoch) 
if (epoch + 1) % 2 == 0 or epoch == 0:
    train_image_batch_paths = visualize_and_save_train_3d(batch_datat['image'], output, batch_datat['label'], batch_indext, epoch)
    train_image_paths.extend(train_image_batch_paths)

if (epoch + 1) % 1 == 0:
    val_image_batch_paths = visualize_and_save_valid(batch_datav['image'], outputsv, batch_datav['label'], batch_indexv, epoch) 
if (epoch + 1) % 2 == 0 or epoch == 0:
    val_image_batch_paths = visualize_and_save_valid_3d(batch_datav['image'], outputsv, batch_datav['label'], batch_indexv, epoch)
val_image_paths.extend(val_image_batch_paths)

plot_losses(train_losses, val_losses)
plot_losses_log(train_losses, val_losses)
plot_learning_rate(learning_rates)
plot_log_learning_rate(learning_rates)
selected_train_images = select_images(train_image_paths)
selected_val_images = select_images(val_image_paths)
create_word_document(expdesc, log_loss_path, selected_train_images, selected_val_images, learning_rate_path, log_learning_rate_path, final_time)