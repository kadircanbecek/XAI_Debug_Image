import os.path
import sys
import time
from datetime import datetime
from typing import Optional, Callable

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from lime import lime_image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from utils.losses import correlationloss
from utils.models import FeatureExtract, WholeModel
import pandas as pd

from utils.utils import plot_confusion_matrix

# check device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# parameters
RANDOM_SEED = 42
LEARNING_RATE = 0.001
BATCH_SIZE = 256
N_EPOCHS = 20

IMG_SIZE = 32
N_CLASSES = 10

lambda_corr = float(sys.argv[1])
print(lambda_corr)


def train(train_loader, model, criterion, optimizer, device):
    '''
    Function for the training step of the training loop
    '''

    model.train()
    running_loss = 0

    correct_pred = 0
    n = 0

    for i, (X, y_true) in tqdm(enumerate(train_loader), total=len(train_loader)):
        optimizer.zero_grad()

        X = X.to(device)
        y_true = y_true.to(device)

        # Forward pass
        fe = model.fe(X)
        y_hat = model.cl(fe)
        loss = criterion(y_hat, y_true)
        loss += lambda_corr * correlationloss(fe)

        running_loss += loss.item() * X.size(0)
        _, predicted_labels = torch.max(y_hat, 1)

        n += y_true.size(0)
        correct_pred += (predicted_labels == y_true).sum()

        # Backward pass
        loss.backward()
        optimizer.step()

    epoch_loss = running_loss / len(train_loader.dataset)
    accuracy = (correct_pred.float() / n).item()
    return model, optimizer, epoch_loss, accuracy


def validate(valid_loader, model, criterion, device):
    '''
    Function for the validation step of the training loop
    '''

    model.eval()
    running_loss = 0

    correct_pred = 0
    n = 0
    for i, (X, y_true) in tqdm(enumerate(valid_loader), total=len(valid_loader)):
        X = X.to(device)
        y_true = y_true.to(device)

        # Forward pass and record loss
        fe = model.fe(X)
        y_hat = model.cl(fe)
        loss = criterion(y_hat, y_true)
        loss += lambda_corr * correlationloss(fe)
        running_loss += loss.item() * X.size(0)
        _, predicted_labels = torch.max(y_hat, 1)

        n += y_true.size(0)
        correct_pred += (predicted_labels == y_true).sum()

    epoch_loss = running_loss / len(valid_loader.dataset)
    accuracy = (correct_pred.float() / n).item()

    return model, epoch_loss, accuracy


def get_accuracy(model, data_loader, device):
    '''
    Function for computing the accuracy of the predictions over the entire data_loader
    '''

    correct_pred = 0
    n = 0

    with torch.no_grad():
        model.eval()
        for X, y_true in data_loader:
            X = X.to(device)
            y_true = y_true.to(device)

            y_prob = model(X)
            _, predicted_labels = torch.max(y_prob, 1)

            n += y_true.size(0)
            correct_pred += (predicted_labels == y_true).sum()

    return correct_pred.float() / n


def plot_losses(train_losses, valid_losses, train_accuracies, val_accuracies, saveloc):
    '''
    Function for plotting training and validation losses
    '''

    # temporarily change the style of the plots to seaborn
    plt.style.use('seaborn')

    train_losses = np.array(train_losses)
    valid_losses = np.array(valid_losses)

    fig, ax = plt.subplots(figsize=(8, 4.5))

    ax.plot(train_losses, color='blue', label='Training loss')
    ax.plot(valid_losses, color='red', label='Validation loss')
    ax.set(title="Loss over epochs",
           xlabel='Epoch',
           ylabel='Loss')
    ax.legend()
    fig.savefig(os.path.join(saveloc, "losses.png"))
    # change the plot style to default
    plt.style.use('default')
    plt.close(fig)

    plt.style.use('seaborn')

    train_accuracies = np.array(train_accuracies)
    val_accuracies = np.array(val_accuracies)

    fig, ax = plt.subplots(figsize=(8, 4.5))

    ax.plot(train_accuracies, color='blue', label='Training loss')
    ax.plot(val_accuracies, color='red', label='Validation loss')
    ax.set(title="Loss over epochs",
           xlabel='Epoch',
           ylabel='Loss')
    ax.legend()
    fig.savefig(os.path.join(saveloc, "accs.png"))
    # change the plot style to default
    plt.style.use('default')
    plt.close(fig)


def training_loop(model, criterion, optimizer, train_loader, valid_loader, epochs, device, dir_out, print_every=1):
    '''
    Function defining the entire training loop
    '''

    # set objects for storing metrics
    best_loss = 1e10
    train_losses = []
    valid_losses = []
    train_accs = []
    valid_accs = []

    # Train model
    for epoch in range(0, epochs):

        # training
        model, optimizer, train_loss, train_acc = train(train_loader, model, criterion, optimizer, device)
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # validation
        with torch.no_grad():
            model, valid_loss, valid_acc = validate(valid_loader, model, criterion, device)
            valid_losses.append(valid_loss)
            valid_accs.append(valid_acc)
            if min(valid_losses) == valid_loss:
                torch.save(model.state_dict(), os.path.join(dir_out, "best.pt"))
                print("best validation so far")

        if epoch % print_every == (print_every - 1):
            print(f'{datetime.now().time().replace(microsecond=0)} --- '
                  f'Epoch: {epoch}\t'
                  f'Train loss: {train_loss:.4f}\t'
                  f'Valid loss: {valid_loss:.4f}\t'
                  f'Train accuracy: {100 * train_acc:.2f}\t'
                  f'Valid accuracy: {100 * valid_acc:.2f}')

    plot_losses(train_losses, valid_losses, train_accs, valid_accs, dir_out)

    return model, optimizer, (train_losses, valid_losses)


class CIFAR3(datasets.CIFAR10):
    def __init__(self, root: str,
                 train: bool = True,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 download: bool = False, ):
        super().__init__(root, train, transform, target_transform, download)
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        self.class_to_idx = {'automobile': 0, 'cat': 1, 'ship': 2}
        self.classes_old = [cl for cl in self.classes]
        self.classes = ['automobile', 'cat', 'ship']
        data_to_keep = [i for i, target in enumerate(self.targets) if self.idx_to_class[target] in self.classes]
        self.targets = [self.class_to_idx[self.idx_to_class[self.targets[i]]] for i in data_to_keep]
        self.data = self.data[data_to_keep]


# define transforms
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Resize(size=(32, 32)),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
transform_for_tensor = transforms.Compose(
    [transforms.Resize(size=(32, 32)),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
transform_only_tensor = transforms.ToTensor()

# download and create datasets
train_dataset = CIFAR3(root='./data', train=True,
                       download=True, transform=transform)

valid_dataset = CIFAR3(root='./data', train=False,
                       download=True,
                       transform=transform)
valid_dataset_wo_transform = CIFAR3(root='./data', train=False,
                                    download=True,
                                    transform=transform_only_tensor)
print(train_dataset.classes)
print(train_dataset.class_to_idx)
# define the data loaders
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=BATCH_SIZE,
                          shuffle=True)

valid_loader = DataLoader(dataset=valid_dataset,
                          batch_size=BATCH_SIZE,
                          shuffle=False)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fe = FeatureExtract()
        self.fc4 = nn.Linear(32, 10)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()
        self.relu4 = nn.ReLU()
        self.relu5 = nn.ReLU()

    def feature_extract(self, x):
        x = self.fe(x)
        return x

    def forward(self, x):
        x = self.feature_extract(x)
        x = self.fc4(x)
        return x


out_dir = f"cifar3/resnet18-{lambda_corr}/"
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
torch.manual_seed(RANDOM_SEED)
print(len(train_dataset.classes))
model = WholeModel(len(train_dataset.classes)).to(DEVICE)
if not os.path.exists(os.path.join(out_dir, "last.pt")):
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    criterion = nn.CrossEntropyLoss()
    model, optimizer, _ = training_loop(model, criterion, optimizer, train_loader, valid_loader, N_EPOCHS, DEVICE,
                                        out_dir)
    torch.save(model.state_dict(), os.path.join(out_dir, "last.pt"))
model.load_state_dict(torch.load(os.path.join(out_dir, "best.pt")))
model.eval()

for i, (X, y) in enumerate(valid_dataset):
    X = X.to(DEVICE)
    X = X.unsqueeze(0)
    feats = model.fe(X)
    feats = feats.cpu().detach().numpy()
    if i == 0:
        feats_all = feats
    else:
        feats_all = np.concatenate([feats_all, feats], axis=0)

ind = np.argsort(-feats_all, axis=0)[:230]
feats_max = feats_all[ind, np.arange(feats_all.shape[1])]
index_per_feat = ind.T


def batch_predict(images):
    model.eval()
    batch = torch.stack(tuple(transform(i) for i in images), dim=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch = batch.to(device)

    logits = model.fe(batch)
    return logits.detach().cpu().numpy()


def visualize_weights(W, feature_idx, class_names, show=False):
    max_abs_W = np.max(np.abs(W)) + 0.1  # For plotting
    fig, ax = plt.subplots()
    fig.set_size_inches(9, 2)
    W = W.T
    ax.barh(class_names, W[feature_idx], color=['green' if w >= 0 else 'red' for w in W[feature_idx]])
    ax.set_xlim((-max_abs_W, max_abs_W))
    ax.set_yticks(range(len(class_names)))
    ax.set_yticklabels(class_names)
    ax.yaxis.tick_right()
    ax.invert_yaxis()
    ax.set_xlabel('Weights')
    ax.set_title(f'Feature {feature_idx}')
    for i, v in enumerate(W[feature_idx]):
        if v >= 0:
            ax.text(v + 0.01, i + .05, '%.4f' % (v), color='black')
        else:
            ax.text(v - 0.13, i + .05, '%.4f' % (v), color='black')
    if show:
        plt.show()
    return fig, ax


explainer = lime_image.LimeImageExplainer()

for i, row in enumerate(index_per_feat):
    if not os.path.exists(f"cifar3-results/results-{lambda_corr}/{i}/"):
        os.makedirs(f"cifar3-results/results-{lambda_corr}/{i}/")
    start = time.time()
    weight = model.cl.linear.weight.cpu().detach().numpy()
    fig, ax = visualize_weights(weight, i, valid_dataset.classes)
    fig.savefig(f"cifar3-results/results-{lambda_corr}/{i}/weight_plot.png")
    plt.close(fig)
    class_inst = np.zeros((len(valid_dataset.classes), len(valid_dataset.classes)))
    # class_inst = []
    # class_inst_pred = []
    for col in row.tolist():
        X, label = valid_dataset[col]
        out = model(X.unsqueeze(0).to(DEVICE))
        idx = int(torch.argmax(out, dim=1).squeeze().cpu().item())
        class_inst[label, idx] = class_inst[label, idx] + 1
    fig, ax = plot_confusion_matrix(class_inst, valid_dataset.classes)
    fig.savefig(f"cifar3-results/results-{lambda_corr}/{i}/conf-mat.png")
    plt.close(fig)

    # print(class_inst)
    # D = class_inst
    # plt.bar(range(len(D)), list(D.values()), align='center')
    # plt.xticks(range(len(D)), list(D.keys()))
    # plt.savefig(f"cifar3-results/results-{lambda_corr}/{i}/most_activated.png")
    # plt.close()
    # print(class_inst_pred)
    #
    # D = class_inst_pred
    # plt.bar(range(len(D)), list(D.values()), align='center')
    # plt.xticks(range(len(D)), list(D.keys()))
    # plt.savefig(f"cifar3-results/results-{lambda_corr}/{i}/most_activated_pred.png")
    # plt.close()
    row = row[:12]
    for j, col in enumerate(row.tolist()):
        image, label = valid_dataset_wo_transform[col]

        image2 = np.uint8(image.numpy().transpose(1, 2, 0) * 255)
        image_org = image2.copy()
        image3 = cv2.resize(image2, (128, 128))
        explanation = explainer.explain_instance(image3, batch_predict, (i,),
                                                 top_labels=None,
                                                 hide_color=0,
                                                 num_samples=1000,
                                                 batch_size=256,
                                                 random_seed=RANDOM_SEED)
        _, mask = explanation.get_image_and_mask(i)
        h, w = image_org.shape[:2]
        mask = np.float32(cv2.resize(np.float32(mask), (w, h)) > 0)
        image2 = np.uint8(image2 * np.dstack([mask for _ in range(3)]))

        image2 += np.uint8(np.ones_like(image2) * 127 * np.dstack([1 - mask for _ in range(3)]))

        cv2.imwrite(f"cifar3-results/results-{lambda_corr}/{i}/{j}-{col}.png",
                    np.uint8(cv2.cvtColor(np.vstack([image2, image_org]), cv2.COLOR_RGB2BGR)))
        print(f"{lambda_corr}/{i}/{j}-{col}")
    end = time.time()

    print("Feature-time:", end - start)
