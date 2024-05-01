import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
from sklearn.manifold import TSNE
import os
from torchvision.utils import save_image
from torchvision.utils import make_grid

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data loading and transformations
data_path = './data'
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = torchvision.datasets.FashionMNIST(root=data_path, train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.FashionMNIST(root=data_path, train=False, download=True, transform=transform)

# Setup DataLoader using the best batch size
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Ensure directory for saving images exists
os.makedirs('saved_images', exist_ok=True)

# Define Encoder
class Encoder(nn.Module):
    def __init__(self, hidden_channels=32, latent_dims=20):
        super(Encoder, self).__init__()
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(1, hidden_channels, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(hidden_channels, hidden_channels * 2, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(hidden_channels * 2, hidden_channels * 4, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc_mu = nn.Linear(hidden_channels * 4 * 3 * 3, latent_dims)
        self.fc_logvar = nn.Linear(hidden_channels * 4 * 3 * 3, latent_dims)

    def forward(self, x):
        x = self.cnn_layers(x)
        x = torch.flatten(x, start_dim=1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar


# Decoder
class Decoder(nn.Module):
    def __init__(self, hidden_channels=32, latent_dims=20):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(latent_dims, hidden_channels * 4 * 3 * 3)
        self.conv_trans_layers = nn.Sequential(
            nn.ConvTranspose2d(hidden_channels * 4, hidden_channels * 2, 3,
                               stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_channels * 2, hidden_channels, 5,
                               stride=2, padding=2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_channels, 1, 10,
                               stride=2, padding=2, output_padding=0),
            nn.Sigmoid()
        )

    def forward(self, z):
        z = self.fc(z)
        z = z.view(-1, 32 * 4, 3, 3)
        z = self.conv_trans_layers(z)
        return z


# VAE
class VAE(nn.Module):
    def __init__(self, encoder, decoder):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

# Initialize model and optimizer
model = VAE(Encoder(), Decoder()).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Best learning rate

# Loss function
def vae_loss(recon_x, x, mu, logvar, beta=0.2):
    BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + beta * KLD, BCE, KLD

# To keep track of the average losses
losses = {"total_loss": [], "BCE": [], "KLD": []}

# Function to save images
def save_original_images(data):
    # data is a batch of original images
    img = make_grid(data, nrow=8, padding=2, normalize=True)  # Change nrow to adjust the grid layout
    save_image(img, 'saved_images/original.png')

def save_reconstructed_images(recon_data, epoch, nrow=8):
    # recon_data is a batch of reconstructed images
    img = make_grid(recon_data, nrow=nrow, padding=2, normalize=True)  # Change nrow to adjust the grid layout
    save_image(img, f'saved_images/epoch_{epoch}.png')
# Pick a fixed batch of data to visualize progress
fixed_batch, _ = next(iter(test_loader))
save_original_images(fixed_batch)

# Training loop
for epoch in range(45):
    model.train()
    epoch_loss = 0.0
    epoch_bce = 0.0
    epoch_kld = 0.0
    for data, _ in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss, BCE, KLD = vae_loss(recon_batch, data, mu, logvar)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_bce += BCE.item()
        epoch_kld += KLD.item()

    # Calculate average losses and record for plotting
    avg_loss = epoch_loss / len(train_loader.dataset)
    avg_bce = epoch_bce / len(train_loader.dataset)
    avg_kld = epoch_kld / len(train_loader.dataset)
    losses["total_loss"].append(avg_loss)
    losses["BCE"].append(avg_bce)
    losses["KLD"].append(avg_kld)
    print(f"Epoch {epoch + 1}, Total Loss: {avg_loss}, BCE: {avg_bce}, KLD: {avg_kld}")

    # Visualize and save reconstructed images
    model.eval()
    with torch.no_grad():
        recon_batch, _, _ = model(fixed_batch.to(device))
    save_reconstructed_images(recon_batch.cpu(), epoch + 1, nrow=8)

    model.train()

# After training, plot and save loss graph
plt.figure(figsize=(10, 5))
for key, value in losses.items():
    plt.plot(value, label=key)
plt.title('Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig('saved_images/loss_graph.png')
plt.show()

# Evaluate t-SNE on the trained model and plot
model.eval()
latents = []
labels = []

with torch.no_grad():
    for data, target in test_loader:
        data = data.to(device)
        mu, logvar = model.encoder(data)
        latents.append(mu.cpu().numpy())
        labels.append(target.cpu().numpy())

# Concatenate all batches
latents = np.concatenate(latents, axis=0)
labels = np.concatenate(labels, axis=0)

# Use t-SNE to reduce dimensionality
tsne = TSNE(n_components=2, perplexity=30, n_iter=300)
latents_2d = tsne.fit_transform(latents)

# Plot the reduced latent space
plt.figure(figsize=(10, 5))
for i in range(10):
    indices = labels == i
    plt.scatter(latents_2d[indices, 0], latents_2d[indices, 1], label=str(i), alpha=0.5)
plt.legend()
plt.title('Latent Space (Reduced to 2D with t-SNE)')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.show()

# Show images function
def show_images(data, recon):
    fig, axs = plt.subplots(2, 10, figsize=(10, 5))
    for i in range(10):
        axs[0, i].imshow(data[i][0].cpu().numpy(), cmap='gray')
        axs[0, i].axis('off')
        axs[1, i].imshow(recon[i][0].cpu().numpy(), cmap='gray')
        axs[1, i].axis('off')
    plt.show()

# Display and save images after last epoch
data, _ = next(iter(test_loader))
data = data.to(device)
with torch.no_grad():
    recon, _, _ = model(data)
    show_images(data, recon)
    save_image(make_grid(recon.cpu(), nrow=8), 'saved_images/reconstructions.png')
    save_image(make_grid(data.cpu(), nrow=8), 'saved_images/originals.png')
