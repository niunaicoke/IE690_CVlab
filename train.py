import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
from model import MovieCNN
import time
from datetime import datetime

try:
    matplotlib.use('Qt5Agg')
except ImportError:
    try:
        matplotlib.use('TkAgg')
    except ImportError:
        matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns


torch.manual_seed(42)
np.random.seed(42)


data_dir = "/home/yulab/Desktop/extracted_frames"
categories = ['titanic', 'lost in translation', 'fastfurious', 'conjuring']
batch_size = 32
image_size = (224, 224)
num_epochs = 100
learning_rate = 0.001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MovieFramesDataset(Dataset):
    def __init__(self, root_dir, categories, transform=None):
        self.root_dir = root_dir
        self.categories = categories
        self.transform = transform
        self.images = []
        self.labels = []

        for label, category in enumerate(categories):
            category_dir = os.path.join(root_dir, category)
            if not os.path.exists(category_dir):
                continue

            for img_name in os.listdir(category_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(category_dir, img_name)
                    self.images.append(img_path)
                    self.labels.append(label)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label




data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


full_dataset = MovieFramesDataset(data_dir, categories, transform=data_transforms['train'])
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])


val_dataset.dataset.transform = data_transforms['val']


train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")





model = MovieCNN(num_classes=len(categories)).to(device)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)


def train_model():
    train_loss_history = []
    val_loss_history = []
    train_acc_history = []
    val_acc_history = []

    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)


        model.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = running_corrects.double() / len(train_dataset)


        train_loss_history.append(epoch_loss)
        train_acc_history.append(epoch_acc.cpu().item())

        print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')


        model.eval()
        running_loss = 0.0
        running_corrects = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                _, preds = torch.max(outputs, 1)

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(val_dataset)
        epoch_acc = running_corrects.double() / len(val_dataset)

        val_loss_history.append(epoch_loss)
        val_acc_history.append(epoch_acc.cpu().item())

        print(f'Val Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')


        if epoch_acc > best_acc:
            best_acc = epoch_acc
            torch.save(model.state_dict(), 'best_model.pth')
    

        scheduler.step()
        print()

    print(f'Best val Acc: {best_acc:4f}')


    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_loss_history, label='Train Loss')
    plt.plot(val_loss_history, label='Val Loss')
    plt.legend()
    plt.title('Loss History')

    plt.subplot(1, 2, 2)
    plt.plot(train_acc_history, label='Train Accuracy')
    plt.plot(val_acc_history, label='Val Accuracy')
    plt.legend()
    plt.title('Accuracy History')

    plt.show()

    return model


trained_model = train_model()


def evaluate_model(model, dataloader):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return all_labels, all_preds



model.load_state_dict(torch.load('best_model.pth'))


true_labels, pred_labels = evaluate_model(model, val_loader)


print("Classification Report:")
print(classification_report(true_labels, pred_labels, target_names=categories))


