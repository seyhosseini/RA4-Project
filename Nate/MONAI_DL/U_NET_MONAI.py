import os
from monai.transforms import (Compose, LoadImaged, EnsureChannelFirstd, CenterSpatialCropd, RandRotate90d, ToTensord)
from monai.networks.nets import UNet
from monai.losses import DiceLoss
from monai.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import torch
from torch.optim import Adam


md_file_path = "./visualization.md"
# image_folder = "./PNG/"

# if os.path.exists(md_file_path):
#     md_file_path.delete!!!!!!

# if os.path.exists(md_file_path):
#     PNG

print("Defining the create_dataset function...")
# Function to create dataset
def create_dataset(data_dir):
    data_dicts = []
    for filename in os.listdir(data_dir):
        if filename.endswith("_Vx3.nrrd"):
            image_path = os.path.join(data_dir, filename)
            label_filename = filename.replace("_Vx3.nrrd", "_Label.nrrd")
            label_path = os.path.join(data_dir, label_filename)
            data_dicts.append({'image': image_path, 'label': label_path})
    return data_dicts

print("Setting data paths...")
# Set data paths
train_data_dir = "z:/W-People/Nate/Deep_Learning_Data/Train"
val_data_dir = "z:/W-People/Nate/Deep_Learning_Data/Validation"
model_save_path = "z:/W-People/Nate/Deep_Learning_Data/Nate_Unet(NEWTRY).pth"
optimizer_save_path = "z:/W-People/Nate/Deep_Learning_Data/Nate_Unet_optimizer(NEWTRY).pth"

print("Creating datasets...")
train_files = create_dataset(train_data_dir)
val_files = create_dataset(val_data_dir)

# Define the size of the cropped region
# roi_size = (128, 128, 128)
roi_size = (64, 64, 64)
print("Defining transformations...")
# Transformations
train_transforms = Compose([
    LoadImaged(keys=['image', 'label']),
    EnsureChannelFirstd(keys=['image', 'label']),
    CenterSpatialCropd(keys=['image', 'label'], roi_size=roi_size),
    RandRotate90d(keys=['image', 'label'], prob=0.5),
    ToTensord(keys=['image', 'label']),
])

val_transforms = Compose([
    LoadImaged(keys=['image', 'label']),
    EnsureChannelFirstd(keys=['image', 'label']),
    CenterSpatialCropd(keys=['image', 'label'], roi_size=roi_size),
    ToTensord(keys=['image', 'label']),
])


print("Initializing data loaders...")
# Data Loaders
train_ds = Dataset(train_files, train_transforms)
val_ds = Dataset(val_files, val_transforms)
train_loader = DataLoader(train_ds, batch_size=1, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=1)


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

loss_function = DiceLoss(to_onehot_y=True, softmax=True) ###############1
criterion = loss_function
optimizer = Adam(net.parameters(), 1e-3)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net.to(device)

if os.path.exists(model_save_path):
    net.load_state_dict(torch.load(model_save_path, map_location=device))
    print("Model state loaded successfully.")
if os.path.exists(optimizer_save_path):
    optimizer.load_state_dict(torch.load(optimizer_save_path))
    print("Optimizer state loaded successfully.")


print("Setting up loss function and optimizer...")
# Loss function and optimizer

# Training step
def train_step(batch_data, model, loss_function, optimizer, device):
    # print("Reading in the images")
    images, labels = batch_data['image'], batch_data['label']
    images, labels = images.to(device), labels.to(device)
    optimizer.zero_grad()
    print("Passing Through the Network")
    outputs = model(images)
    loss = loss_function(outputs, labels)
    print("Loss Backward")
    loss.backward()
    print("Stepping Optimizer")
    optimizer.step()
    return loss.item(), outputs

# Visualization and saving function
def visualize_and_save(inputs, outputs, labels, iteration, epoch):

    with open(md_file_path, "a") as md_file:
        slice_idx = inputs.shape[2] // 2
        for i in range(inputs.shape[0]):
            fig, ax = plt.subplots(1, 2, figsize=(10, 5))
            ax[0].imshow(labels[i, 0, :, :, slice_idx], cmap="gray")
            ax[0].set_title("Ground Truth")
            ax[1].imshow(outputs[i, 0, :, :, slice_idx].detach().cpu(), cmap="gray")
            ax[1].set_title("Prediction")
            img_path = f"./PNG/Epoch-{epoch}_iteration-{iteration}_batch-{i}.png"
            plt.savefig(img_path) #".\ABC.png"
            plt.close()
            md_file.write(f"![Epoch-{epoch} Iteration-{iteration} Batch-{i}]({img_path})\n\n")
        md_file.close()

    print(f"Markdown file updated at {md_file_path}")

# Loss Plotting Function
def plot_losses(train_losses, val_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
    plt.legend()
    plt.draw()
    plt.pause(0.001)

# Main training loop
num_epochs = 1 # Example value
display_interval = 1 # Example value

train_losses = []
val_losses = []
print("Starting the training loop...")
for epoch in range(num_epochs):
    epoch_train_loss = 0.0
    epoch_val_loss = 0.0

    # Training Phase
    net.train()
    print("\nReading in the first batch - T ..")
    for iteration, batch_data in enumerate(train_loader):
        print(f"Training Iteration: {iteration}")
        loss, output = train_step(batch_data, net, criterion, optimizer, device)
        epoch_train_loss += loss
        if iteration % display_interval == 0:
            visualize_and_save(batch_data['image'], output, batch_data['label'], iteration, epoch) 
        print("\nReading in the next batch - T [if any] ..")

    train_losses.append(epoch_train_loss / len(train_loader))

    print("Running the validation loop ..")
    # Validation Phase
    net.eval()
    with torch.no_grad():
        print("Reading in the first batch - V ..")
        for batch_data in val_loader:
            images, labels = batch_data['image'], batch_data['label']
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            loss = criterion(outputs, labels)
            epoch_val_loss += loss.item()
            print("Reading in the next batch - V [if any] ..")
        print()    
    
    val_losses.append(epoch_val_loss / len(val_loader))

    # Plot Losses
    print("Plotting losses ..")
    plot_losses(train_losses, val_losses)
    print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}")

# plt.ioff()  # Turn off interactive mode
# plt.show()  # Display final plots

torch.save(net.state_dict(), model_save_path)
torch.save(optimizer.state_dict(), optimizer_save_path)