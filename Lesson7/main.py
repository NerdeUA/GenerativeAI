import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torchvision.utils
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# Hyperparameters
batch_size = 64
latent_dim = 100
lr = 0.0001
epochs = 2000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Data preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

mnist_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
data_loader = DataLoader(mnist_data, batch_size=batch_size, shuffle=True)

# Function to evaluate discriminator
def evaluate_discriminator(discriminator, real_data, fake_data):
    discriminator.eval()

    # Real images
    real_predictions = discriminator(real_data).detach().cpu().numpy()
    real_labels = [1] * len(real_predictions)

    # Fake images
    fake_predictions = discriminator(fake_data).detach().cpu().numpy()
    fake_labels = [0] * len(fake_predictions)

    # Combine results
    predictions = (np.array(real_predictions.tolist() + fake_predictions.tolist()) > 0.5).astype(int)
    true_labels = np.array(real_labels + fake_labels)

    # Confusion matrix
    cm = confusion_matrix(true_labels, predictions, labels=[1, 0])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Real", "Fake"])
    disp.plot(cmap="Blues", values_format='d')
    plt.title("Confusion Matrix of Discriminator")
    plt.show()

# Initialize weights
def weights_init(layer):
    if isinstance(layer, (nn.ConvTranspose2d, nn.Conv2d, nn.BatchNorm2d)):
        nn.init.normal_(layer.weight.data, 0.0, 0.02)

# Generator class
class Generator(nn.Module):
    def __init__(self, latent_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 128, kernel_size=7, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)

# Discriminator class
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Flatten(),
            nn.Linear(128*7*7, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# Initialize models
generator = Generator(latent_dim).apply(weights_init).to(device)
discriminator = Discriminator().apply(weights_init).to(device)

# Loss and optimizers
criterion = nn.BCELoss()
g_optimizer = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
d_optimizer = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

# Training loop
losses_g, losses_d = [], []
fixed_noise = torch.randn(64, latent_dim, 1, 1, device=device)

for epoch in range(epochs):
    epoch_loss_d, epoch_loss_g = 0, 0
    for i, (real_images, _) in enumerate(data_loader):
        batch_size = real_images.size(0)

        # Labels
        real_labels = torch.ones(batch_size, 1, device=device)
        fake_labels = torch.zeros(batch_size, 1, device=device)

        # Train Discriminator
        discriminator.zero_grad()
        real_images = real_images.to(device)
        outputs_real = discriminator(real_images)
        loss_real = criterion(outputs_real, real_labels)

        noise = torch.randn(batch_size, latent_dim, 1, 1, device=device)
        fake_images = generator(noise)
        outputs_fake = discriminator(fake_images.detach())
        loss_fake = criterion(outputs_fake, fake_labels)

        loss_d = loss_real + loss_fake
        loss_d.backward()
        d_optimizer.step()
        epoch_loss_d += loss_d.item()

        # Train Generator
        generator.zero_grad()
        outputs = discriminator(fake_images)
        loss_g = criterion(outputs, real_labels)
        loss_g.backward()
        g_optimizer.step()
        epoch_loss_g += loss_g.item()

        if (i + 1) % 10 == 0:
            print('Epoch [{}/{}], Step [{}/{}], D_loss: {:.4f}, G_loss: {:.4f}, D(x): {:.2f}, D(G(z)): {:.2f}'.format(
                epoch + 1, epochs, i + 1, len(data_loader),
                loss_d.item(), loss_g.item(),
                outputs_real.mean().item(), outputs_fake.mean().item()
            ))


    losses_d.append(epoch_loss_d / len(data_loader))
    losses_g.append(epoch_loss_g / len(data_loader))

    # Visualize generated images
    # with torch.no_grad():
    #     fake_images = generator(fixed_noise).reshape(-1, 1, 28, 28)
    #     grid = torchvision.utils.make_grid(fake_images, normalize=True)
    #     plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
    #     plt.title(f"Epoch {epoch+1}")
    #     plt.show()

    # Evaluate discriminator
    # real_images = next(iter(data_loader))[0].to(device)
    # fake_images = generator(torch.randn(batch_size, latent_dim, 1, 1, device=device)).detach()
    # print(f"Evaluating discriminator at epoch {epoch + 1}")
    # evaluate_discriminator(discriminator, real_images, fake_images)

# Final evaluation
real_images = next(iter(data_loader))[0].to(device)
fake_images = generator(torch.randn(batch_size, latent_dim, 1, 1, device=device)).detach()
print("Final evaluation of discriminator")
evaluate_discriminator(discriminator, real_images, fake_images)

# Visualize losses
plt.plot(losses_d, label="Discriminator Loss")
plt.plot(losses_g, label="Generator Loss")
plt.legend()
plt.title("Training Losses")
plt.show()
