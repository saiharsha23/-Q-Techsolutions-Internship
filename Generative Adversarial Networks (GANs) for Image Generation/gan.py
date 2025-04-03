import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import numpy as np

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load CIFAR-10 Dataset 
transform = transforms.Compose([
    transforms.Resize(32), 
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
dataloader = torch.utils.data.DataLoader(
    torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform),
    batch_size=128, shuffle=True  # Increased batch size for speed
)

# Define Generator 
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(100, 256, 4, 1, 0, bias=False), nn.BatchNorm2d(256), nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False), nn.BatchNorm2d(128), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False), nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)

# Define Discriminator 
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1, bias=False), nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False), nn.BatchNorm2d(128), nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False), nn.BatchNorm2d(256), nn.LeakyReLU(0.2),
            nn.Conv2d(256, 1, 4, 1, 0, bias=False), nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x).view(-1, 1)

# Initialize Models
generator = Generator().to(device)
discriminator = Discriminator().to(device)
criterion = nn.BCELoss()

# Use RMSprop (Faster Convergence)
optimizer_G = optim.RMSprop(generator.parameters(), lr=0.0002)
optimizer_D = optim.RMSprop(discriminator.parameters(), lr=0.0002)

# Enable Mixed Precision Training for Speedup (only on GPU)
scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

# Training Loop
num_epochs = 5  # Reduced epochs for speed
noise_dim = 100

for epoch in range(num_epochs):
    for real_images, _ in dataloader:
        real_images = real_images.to(device)
        real_labels = torch.ones(real_images.size(0), 1).to(device)
        fake_labels = torch.zeros(real_images.size(0), 1).to(device)

        # Train Discriminator
        optimizer_D.zero_grad()
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()): 
            real_loss = criterion(discriminator(real_images), real_labels)
            fake_images = generator(torch.randn(real_images.size(0), noise_dim, 1, 1).to(device)).detach()
            fake_loss = criterion(discriminator(fake_images), fake_labels)
            d_loss = real_loss + fake_loss
        scaler.scale(d_loss).backward()
        scaler.step(optimizer_D)

        # Train Generator
        optimizer_G.zero_grad()
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            fake_images = generator(torch.randn(real_images.size(0), noise_dim, 1, 1).to(device))
            g_loss = criterion(discriminator(fake_images), real_labels)
        scaler.scale(g_loss).backward()
        scaler.step(optimizer_G)
        scaler.update()

    print(f"Epoch [{epoch+1}/{num_epochs}] | D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f}")

# Generate and Display Images
def show_images(fake_images):
    grid = make_grid(fake_images.cpu().detach(), normalize=True)
    plt.imshow(np.transpose(grid.numpy(), (1,2,0)))
    plt.axis("off")
    plt.show()

# Generate 16 new images
with torch.no_grad():
    fake_images = generator(torch.randn(16, 100, 1, 1).to(device))
show_images(fake_images)
