
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



md_file_path = "./visualization.md"
hostname = socket.gethostname()
expdesc = (f"Job Test: \n Copy of Job 8 \n Train Images: 2 \n Validation Images: 5 \n Train Crop: RandSpatialCrop \n \
Validation Crop: RandSpatialCrop \n Epochs: 1500 \n Learning Rate: 1e-3 \n \
Spacing: (0.8, 0.8, 0.8) \n Batch Size: 4 \n Patch Size: (128, 128, 128) \n Compute Node: {hostname}") # REMEBER TO CHANGE THIS EACH JOB

with open('expdesc.txt', 'w') as file:
    file.write(expdesc)

# writer = SummaryWriter('runs/unet_experiment')

log_loss_path = "./PNG/log-lossfunction.png"
learning_rate_path = "./PNG/learning_rate.png"
log_learning_rate_path ="./PNG/log_learning_rate.png"
train_data_dir = "../../TrainData/"
val_data_dir = "../../ValidationData/"
model_save_path = "./model/Nate_Unet.pth"
optimizer_save_path = "./model/Nate_Unet_optimizer.pth"
train_image_paths = []
val_image_paths = []
train_losses = []
val_losses = []

word_output_path = "./JobReport.docx"
# Main training loop
num_epochs = 6
display_interval = 1 
starttime = time.time()

print("Defining the create_dataset function...")
# Function to create dataset
print("Setting data paths...")
# Set data paths

print("Creating datasets...")
train_files = create_dataset(train_data_dir)
val_files = create_dataset(val_data_dir)

downsampling_transform = Spacingd(
    keys=['image', 'label'], 
    pixdim=(0.8, 0.8, 0.8), 
    mode=('bilinear', 'nearest') ###############################################################
)

# Define the size of the cropped region
roi_size = (128, 128, 128)
# roi_size = (64, 64, 64)
print("Defining transformations...")
# Transformations
print(f"Elapsed: {(time.time()-starttime)/60:.2f} Minutes")

train_transforms = Compose([
    LoadImaged(keys=['image', 'label']),
    EnsureChannelFirstd(keys=['image', 'label']),
    downsampling_transform, #####################################################################
    # SpatialPadd(keys=['image', 'label'], spatial_size=roi_size, method='symmetric'),  # Add padding here
    RandSpatialCropd(keys=['image', 'label'], roi_size=roi_size),
    RandRotate90d(keys=['image', 'label'], prob=0.5),
    ScaleIntensityRanged(keys=['image'], a_min=-1100, a_max=3000, b_min=-1.0, b_max=1.0, clip=True),
    ToTensord(keys=['image', 'label']),
])

val_transforms = Compose([
    LoadImaged(keys=['image', 'label']),
    EnsureChannelFirstd(keys=['image', 'label']),
    downsampling_transform,
    # SpatialPadd(keys=['image', 'label'], spatial_size=roi_size, method='symmetric'),  # Add padding here
    RandSpatialCropd(keys=['image', 'label'], roi_size=roi_size),
    ScaleIntensityRanged(keys=['image'], a_min=-1100, a_max=3000, b_min=-1.0, b_max=1.0, clip=True),
    ToTensord(keys=['image', 'label']),
])

print("Initializing data loaders...")
# Data Loaders
train_ds = Dataset(train_files, train_transforms)
val_ds = Dataset(val_files, val_transforms)
train_loader = DataLoader(train_ds, batch_size=4, shuffle=True) 
val_loader = DataLoader(val_ds, batch_size=4, shuffle=True) 

print("Initializing U-Net model...")
# Model
net = UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=3,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2,
)


print(f"Elapsed: {(time.time()-starttime)/60:.2f} Minutes")
loss_function = DiceLoss(to_onehot_y=True, softmax=True) #########################################
criterion = loss_function
optimizer = Adam(net.parameters(), 1e-3)
scheduler = CosineAnnealingLR(optimizer, T_max=1500, eta_min=1e-6)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net.to(device)

if not os.path.exists("./model"):
    os.makedirs("./model", exist_ok=True)
if os.path.exists(model_save_path):
    net.load_state_dict(torch.load(model_save_path, map_location=device))
    print("Model state loaded successfully.")
if os.path.exists(optimizer_save_path):
    optimizer.load_state_dict(torch.load(optimizer_save_path))
    print("Optimizer state loaded successfully.")

print("Setting up loss function and optimizer...")





learning_rates = []
print("Starting the training loop...")
for epoch in range(num_epochs):
    epoch_train_loss = 0.0
    epoch_val_loss = 0.0
    
    # Training Phase
    net.train()
    # batch_indext = 0
    print(f"Elapsed: {(time.time()-starttime)/60:.2f} Minutes")
    print("\nReading in the first batch - T ..")
    for batch_indext, batch_datat in enumerate(train_loader):
        print(f"Training Iteration: {batch_indext}")

        imagest, labelst = batch_datat['image'], batch_datat['label']
        imagest, labelst = imagest.to(device), labelst.to(device)
        optimizer.zero_grad()
        print("Passing Through the Network")
        outputst = model(imagest)
        print(f"Output shape: {outputst.shape}, Label shape: {labelst.shape}")
        # print("Label tensor values (sample):", labelst[0, :, 0, 0, 0])

        loss = loss_function(outputst, labelst)
        print("Loss Backward")
        loss.backward()
        print("Stepping Optimizer")
        optimizer.step()

        # loss, output = train_step(batch_datat, net, criterion, optimizer, device)
        epoch_train_loss += loss.item()
       
        print("\nReading in the next batch - T [if any] ..")
        # batch_indext = batch_indext + 1
    train_losses.append(epoch_train_loss / len(train_loader))
    print("Running the validation loop ..")

    # Validation Phase
    net.eval()
    with torch.no_grad():
        print("Reading in the first batch - V ..")
        for batch_indexv, batch_datav in enumerate(val_loader):
            print(f"Validation Iteration: {batch_indexv}")
          
            imagesv, labelsv = batch_datav['image'], batch_datav['label']
            imagesv, labelsv = imagesv.to(device), labelsv.to(device)
            outputsv = net(imagesv)
            print(f"Validation - Output shape: {outputsv.shape}, Label shape: {labelsv.shape}")
            loss = criterion(outputsv, labelsv)
            epoch_val_loss += loss.item()
        
        print()    
    val_losses.append(epoch_val_loss / len(val_loader))
    scheduler.step()
    for param_group in optimizer.param_groups:
        current_lr = param_group['lr']
        print(f"Epoch: {epoch}, Current Learning Rate: {current_lr}")
    current_lr = optimizer.param_groups[0]['lr']
    learning_rates.append(current_lr)


    # Plot Losses
    print("Plotting losses ..")
   
    print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}")


    


# Call the function to create the document
torch.save(net.state_dict(), model_save_path)
torch.save(optimizer.state_dict(), optimizer_save_path)
print(f"Elapsed: {(time.time()-starttime)/60:.2f} Minutes")
elapsed_minutes = (time.time() - starttime) / 60
final_time = f"Time: {elapsed_minutes:.2f} Minutes"
