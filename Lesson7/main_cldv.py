import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torchvision.utils
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import os
from google.colab import drive

drive.mount('/content/drive')

# Гіперпараметри
batch_size = 64
latent_dim = 100
lr = 0.0002
lambda_gp = 10
max_grad_norm = 1.0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Функція для gradient penalty
def compute_gradient_penalty(discriminator, real_samples, fake_samples):
    alpha = torch.rand((real_samples.size(0), 1, 1, 1), device=device)
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
    d_interpolates = discriminator(interpolates)
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

# Оригінальна архітектура Generator
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

# Оригінальна архітектура Discriminator
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

# Шлях до збережених моделей
save_path = '/content/drive/MyDrive/models'
start_epoch = 487
epochs = 800

# Завантаження збережених станів моделей
generator_path = os.path.join(save_path, f'generator_epoch_{start_epoch}.pth')
discriminator_path = os.path.join(save_path, f'discriminator_epoch_{start_epoch}.pth')

if not os.path.exists(generator_path) or not os.path.exists(discriminator_path):
    raise FileNotFoundError(f"Збережені моделі для епохи {start_epoch} не знайдені")

# Ініціалізація моделей та завантаження станів
generator = Generator(latent_dim).to(device)
discriminator = Discriminator().to(device)

generator.load_state_dict(torch.load(generator_path, weights_only=True))
discriminator.load_state_dict(torch.load(discriminator_path, weights_only=True))

print(f"Моделі успішно завантажені з епохи {start_epoch}")

# Оптимізатори та критерій
g_optimizer = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
d_optimizer = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
criterion = nn.BCELoss()

# Завантаження даних
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

mnist_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
data_loader = DataLoader(mnist_data, batch_size=batch_size, shuffle=True)

# Training loop
losses_g, losses_d = [], []
fixed_noise = torch.randn(64, latent_dim, 1, 1, device=device)

for epoch in range(start_epoch, epochs):
    epoch_loss_d, epoch_loss_g = 0, 0
    for i, (real_images, _) in enumerate(data_loader):
        batch_size = real_images.size(0)
        
        real_labels = torch.full((batch_size, 1), 0.95, device=device)
        fake_labels = torch.full((batch_size, 1), 0.05, device=device)

        # Train Discriminator
        discriminator.zero_grad()
        real_images = real_images.to(device)
        real_images = real_images + 0.005 * torch.randn_like(real_images)
        
        outputs_real = discriminator(real_images)
        loss_real = criterion(outputs_real, real_labels)

        noise = torch.randn(batch_size, latent_dim, 1, 1, device=device)
        fake_images = generator(noise)
        outputs_fake = discriminator(fake_images.detach())
        loss_fake = criterion(outputs_fake, fake_labels)

        gradient_penalty = compute_gradient_penalty(discriminator, real_images, fake_images.detach())
        
        loss_d = loss_real + loss_fake + lambda_gp * gradient_penalty
        loss_d.backward()
        
        torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_grad_norm)
        
        d_optimizer.step()
        epoch_loss_d += loss_d.item()

        # Train Generator
        generator.zero_grad()
        outputs = discriminator(fake_images)
        loss_g = criterion(outputs, real_labels)
        loss_g.backward()
        
        torch.nn.utils.clip_grad_norm_(generator.parameters(), max_grad_norm)
        
        g_optimizer.step()
        epoch_loss_g += loss_g.item()

        if (i + 1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(data_loader)}], "
                  f"Loss D: {loss_d.item():.4f}, Loss G: {loss_g.item():.4f}, "
                  f"D(x): {outputs_real.mean().item():.2f}, D(G(z)): {outputs_fake.mean().item():.2f}")

    epoch_loss_d /= len(data_loader)
    epoch_loss_g /= len(data_loader)
    losses_d.append(epoch_loss_d)
    losses_g.append(epoch_loss_g)

    if (epoch + 1) % 5 == 0:
        torch.save(generator.state_dict(), os.path.join(save_path, f'generator_epoch_{epoch+1}.pth'))
        torch.save(discriminator.state_dict(), os.path.join(save_path, f'discriminator_epoch_{epoch+1}.pth'))
        print(f"Models saved for epoch {epoch+1}.")

    print(f"Epoch [{epoch+1}/{epochs}], D_loss: {epoch_loss_d:.4f}, G_loss: {epoch_loss_g:.4f}")

print("Training completed!")

# Візуалізація втрат
plt.figure(figsize=(10, 5))
plt.plot(losses_d, label="Discriminator Loss")
plt.plot(losses_g, label="Generator Loss")
plt.legend()
plt.title("Training Losses")
plt.show()