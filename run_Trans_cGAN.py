import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
import os

from transformers.models.pvt.convert_pvt_to_pytorch import read_in_k_v

from models.Trans_cGAN import AdvancedDopplerGenerator
from models.Trans_cDis import CNNTransformerDiscriminator

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

data_dir = "/home/ubuntu/Downloads/DroneDetect_V2/READY"

# Load the saved tensors
X_train_tensor = torch.load(f"{data_dir}/X_train.pt")
y_train_tensor = torch.load(f"{data_dir}/y_train.pt")
X_test_tensor = torch.load(f"{data_dir}/X_test.pt")
y_test_tensor = torch.load(f"{data_dir}/y_test.pt")

# Print loaded data shapes to confirm
print(f"Loaded train set shapes: X_train={X_train_tensor.shape}, y_train={y_train_tensor.shape}")
print(f"Loaded test set shapes: X_test={X_test_tensor.shape}, y_test={y_test_tensor.shape}")

# Create TensorDataset and DataLoader for training and testing
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

torch.manual_seed(1)
np.random.seed(1)

# Instantiate Generator and Discriminator
seq_len = 150
latent_dim = 100
embed_dim = 16
num_heads = 8
depth = 4
output_channels = 64
num_classes = 21  # Update number of classes as per your dataset

generator = AdvancedDopplerGenerator(seq_len=seq_len, latent_dim=latent_dim, embed_dim=embed_dim,
                                     depth=depth, output_channels=output_channels,
                                     num_classes=num_classes).to(device)
discriminator = CNNTransformerDiscriminator(seq_len=seq_len, in_channels=seq_len, cnn_out_channels=batch_size,
                                            emb_dim=embed_dim, num_classes=num_classes).to(device)


# Apply Xavier initialization
def weights_init(m):
    if isinstance(m, (nn.Conv1d, nn.Linear)):
        nn.init.xavier_normal_(m.weight)


#generator.apply(weights_init)
#discriminator.apply(weights_init)

# Optimizers for Generator and Discriminator
optimizer_G = optim.SGD(generator.parameters(), lr=0.08)
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.01)

# Loss function for GAN - Using CrossEntropyLoss
D_criterion = nn.CrossEntropyLoss()
G_criterion = nn.CrossEntropyLoss()

# Training loop for Conditional GAN
num_epochs = 125
g_loss_list = []
d_loss_list = []
train_accuracy = []

stationary_labels = [0, 3, 6, 9, 12, 15, 18]
hovering_labels = [1, 4, 7, 10, 13, 16, 19]
flying_labels = [2, 5, 8, 11, 14, 17, 20]
def smooth_one_hot(true_labels, classes=21, smoothing=0.1):
    """
    Apply label smoothing to one-hot labels
    :param true_labels: tensor of integer class labels
    :param classes: number of classes
    :param smoothing: label smoothing factor
    :return: smoothed one-hot tensor
    """
    confidence = 1.0 - smoothing
    smoothing_value = smoothing / (classes - 1)
    one_hot = torch.full((true_labels.size(0), classes), smoothing_value).to(true_labels.device)
    one_hot.scatter_(1, true_labels.unsqueeze(1), confidence)
    return one_hot

lambda_gp = 10


def compute_gradient_penalty(discriminator, real_samples, fake_samples, labels, batch_split=100):
    """
    Computes gradient penalty by splitting the original batch into smaller mini-batches.
    """
    batch_size = real_samples.size(0)
    mini_batch_size = batch_size // batch_split

    gradient_penalty = 0
    for i in range(batch_split):
        real_mini_batch = real_samples[i * mini_batch_size: (i + 1) * mini_batch_size]
        fake_mini_batch = fake_samples[i * mini_batch_size: (i + 1) * mini_batch_size]
        labels_mini_batch = labels[i * mini_batch_size: (i + 1) * mini_batch_size]

        alpha = torch.rand(real_mini_batch.size(0), 1, 1).to(device)
        interpolates = alpha * real_mini_batch + ((1 - alpha) * fake_mini_batch)
        interpolates.requires_grad_(True)

        d_interpolates = discriminator(interpolates, labels_mini_batch)

        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(d_interpolates).to(device),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]

        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty += ((gradients.norm(2, dim=1) - 1) ** 2).mean() / batch_split

    return gradient_penalty

for epoch in range(num_epochs):
    generator.train()
    discriminator.train()
    g_loss_epoch = 0.0
    d_loss_epoch = 0.0
    total_batches = 0

    for i, (real_data, real_labels) in enumerate(train_loader):
        speed = torch.zeros(real_labels.size(0), device=device)
        angle = torch.zeros(real_labels.size(0), device=device)
        distance = torch.zeros(real_labels.size(0), device=device)

        for idx, label in enumerate(real_labels):
            label_value = label.item()  # Rescale back to integer labels
            if label_value in stationary_labels:
                speed[idx] = 0
                angle[idx] = 0
                distance[idx] = torch.rand(1).item() * 100  # Distance 0-100 m
            elif label_value in hovering_labels:
                speed[idx] = torch.rand(1).item() * 5  # Speed 0-5 m/s
                angle[idx] = torch.rand(1).item() * np.deg2rad(15)  # Angle 5-15 degrees in radians
                distance[idx] = torch.rand(1).item() * 450 + 50  # Distance 50-500 m
            elif label_value in flying_labels:
                speed[idx] = torch.rand(1).item() * 26  # Speed 0-26 m/s
                angle[idx] = torch.rand(1).item() * np.pi / 2  # Angle 0-90 degrees in radians
                distance[idx] = torch.rand(1).item() * 990 + 10  # Distance 10-1000 m
            else:
                raise ValueError("Label does not match any predefined category")
        real_data = real_data.to(device)
        real_labels = real_labels.long().to(device)


        # Training the Discriminator
        optimizer_D.zero_grad()
        # Generate fake data conditioned on labels
        z = torch.randn(real_data.size(0), latent_dim).to(device)
        fake_data = generator(z, real_labels).detach()

        # Discriminator loss on real data
        real_output = discriminator(real_data, real_labels)
        real_loss = D_criterion(real_output, real_labels)

        # Discriminator loss on fake data
        fake_output = discriminator(fake_data, real_labels)
        fake_loss = D_criterion(fake_output, real_labels)
        #gradient_penalty = compute_gradient_penalty(discriminator, real_data, fake_data, real_labels)

        # Total Discriminator loss
        #d_loss = real_loss + fake_loss + lambda_gp * gradient_penalty
        d_loss = (real_loss + fake_loss)/2
        d_loss.backward()
        optimizer_D.step()

        # Training the Generator
        optimizer_G.zero_grad()
        fake_data = generator(z, real_labels)
        fake_output = discriminator(fake_data, real_labels)
        g_loss = G_criterion(fake_output, real_labels)
        g_loss.backward()
        optimizer_G.step()
        # Print Discriminator gradient values
        #for name, param in discriminator.named_parameters():
        #    if param.grad is not None:
        #        print(f"Discriminator - Layer: {name}, Gradients: {param.grad.abs().mean().item()}")

        # Print Generator gradient values
        #for name, param in generator.named_parameters():
        #    if param.grad is not None:
        #        print(f"Generator - Layer: {name}, Gradients: {param.grad.abs().mean().item()}")

        # Accumulate losses for plotting
        g_loss_epoch += g_loss.item()
        d_loss_epoch += d_loss.item()
        total_batches += 1

    # Average losses for the epoch
    g_loss_list.append(g_loss_epoch / total_batches)
    d_loss_list.append(d_loss_epoch / total_batches)

    print(
        f"Epoch [{epoch + 1}/{num_epochs}] - Generator Loss: {g_loss_list[-1]:.4f}, Discriminator Loss: {d_loss_list[-1]:.4f}")

    # Evaluate discriminator after each epoch
    discriminator.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for real_data, labels in test_loader:
            real_data = real_data.to(device)
            labels = labels.to(device)

            predictions = discriminator(real_data, labels).argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

        accuracy = 100 * correct / total
        train_accuracy.append(accuracy)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Test Accuracy: {accuracy:.2f}%")

# Plot Generator and Discriminator Loss
plt.figure()
plt.plot(range(1, num_epochs + 1), g_loss_list, label='Generator Loss')
plt.plot(range(1, num_epochs + 1), d_loss_list, label='Discriminator Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
