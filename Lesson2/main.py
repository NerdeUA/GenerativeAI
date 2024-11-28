import torch
import matplotlib.pyplot as plt
from torch import optim
import torch.nn as nn
from model import NeuralNet
from data_utils import prepare_data
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load data
dataset_path = "./datasets"
train_loader, test_loader = prepare_data(dataset_path)

# Initialize model
model = NeuralNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training
epochs = 10
loss_values = []
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    epoch_loss = running_loss / len(train_loader)
    loss_values.append(epoch_loss)
    print(f'Epoch {epoch+1}, Loss: {epoch_loss:.4f}')

# Plot training loss
plt.plot(range(1, epochs + 1), loss_values, marker='o')
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

# Evaluation
model.eval()
correct = 0
total = 0
y_true, y_pred = [], []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())

print(f'Accuracy on test data: {100 * correct / total:.2f}%')

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=range(10), yticklabels=range(10))
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Visualization of predictions
examples = iter(test_loader)
example_data, example_targets = next(examples)
example_data, example_targets = example_data.to(device), example_targets.to(device)

with torch.no_grad():
    outputs = model(example_data)
    _, predictions = torch.max(outputs, 1)

for i in range(6):
    plt.subplot(2, 3, i+1)
    plt.imshow(example_data[i][0].cpu(), cmap='gray')
    plt.title(f'True: {example_targets[i].item()}, Pred: {predictions[i].item()}')
plt.show()
