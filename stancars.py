import glob
import os.path
import shutil

import rasterio.features

from shapely import geometry
from tqdm import tqdm

from ResNet import ResNet18, resnet_init
from concavehull_scipy import alpha_shape

import cv2
import numpy
from captum.attr import LRP
import numpy as np
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from captum.attr._utils.visualization import visualize_image_attr
from torch.utils.data import DataLoader

from torchvision import datasets, transforms

import matplotlib.pyplot as plt


class TrainingInterrupt(Exception):
    def __init__(self, epochs, model, optimizer):
        super(TrainingInterrupt, self).__init__()
        self.epochs = epochs
        self.model = model
        self.optimizer = optimizer


# check device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# parameters
RANDOM_SEED = 42
LEARNING_RATE = 0.001
BATCH_SIZE = 32
layers = 18
N_EPOCHS = 50

IMG_SIZE = 96
N_CLASSES = 10


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
        y_hat = model(X)
        loss = criterion(y_hat, y_true)
        running_loss += loss.item() * X.size(0)

        # Backward pass
        loss.backward()
        optimizer.step()

        _, predicted_labels = torch.max(y_hat, 1)

        n += y_true.size(0)
        correct_pred += (predicted_labels == y_true).sum()

    epoch_loss = running_loss / len(train_loader.dataset)
    accuracy = correct_pred.float() / n

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
        y_hat = model(X)
        loss = criterion(y_hat, y_true)
        running_loss += loss.item() * X.size(0)
        _, predicted_labels = torch.max(y_hat, 1)

        n += y_true.size(0)
        correct_pred += (predicted_labels == y_true).sum()

    epoch_loss = running_loss / len(valid_loader.dataset)
    accuracy = correct_pred.float() / n

    return model, epoch_loss, accuracy


def plot_losses(train_losses, valid_losses):
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
    fig.show()

    # change the plot style to default
    plt.style.use('default')


def training_loop(model, criterion, optimizer, train_loader, valid_loader, epochs, device, print_every=1,
                  start_epoch=0):
    '''
    Function defining the entire training loop
    '''

    # set objects for storing metrics
    best_loss = 1e10
    train_losses = []
    valid_losses = []

    # Train model
    for epoch in range(start_epoch, epochs):

        try:
            # training
            model, optimizer, train_loss, train_acc = train(train_loader, model, criterion, optimizer, device)
            train_losses.append(train_loss)

            # validation
            with torch.no_grad():
                model, valid_loss, valid_acc = validate(valid_loader, model, criterion, device)
                valid_losses.append(valid_loss)

            print(f'{datetime.now().time().replace(microsecond=0)} --- '
                  f'Epoch: {epoch}\t'
                  f'Train loss: {train_loss:.4f}\t'
                  f'Valid loss: {valid_loss:.4f}\t'
                  f'Train accuracy: {100 * train_acc:.2f}\t'
                  f'Valid accuracy: {100 * valid_acc:.2f}')
        except KeyboardInterrupt:
            raise TrainingInterrupt(epoch, model, optimizer)

    plot_losses(train_losses, valid_losses)

    return model, optimizer, (train_losses, valid_losses)


# define transforms
train_transform = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(35),
        transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
        transforms.RandomGrayscale(p=0.5),
        transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
        transforms.RandomPosterize(bits=2, p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )])

# define transforms
valid_transform = transforms.Compose(
    [transforms.Resize((IMG_SIZE, IMG_SIZE)),
     transforms.ToTensor(),
     transforms.Normalize(
         mean=[0.485, 0.456, 0.406],
         std=[0.229, 0.224, 0.225]
     )])

# download and create datasets
train_dataset = datasets.STL10(root='./data', split="train",
                               download=True, transform=train_transform)

valid_dataset = datasets.STL10(root='./data', split="test",
                               download=True,
                               transform=valid_transform)

# define the data loaders
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=BATCH_SIZE,
                          shuffle=True)

valid_loader = DataLoader(dataset=valid_dataset,
                          batch_size=BATCH_SIZE,
                          shuffle=False)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
if not os.path.exists(f"stl10-{layers}/*.pt"):
    if os.path.exists(f"stl10-{layers}/"):
        shutil.rmtree(f"stl10-{layers}")
    os.mkdir(f"stl10-{layers}/")
    model = resnet_init(layers, N_CLASSES).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    try:
        model, optimizer, _ = training_loop(model, criterion, optimizer, train_loader, valid_loader, N_EPOCHS, DEVICE)
        torch.save(model.state_dict(), f"stl10-{layers}/epoch-{str(N_EPOCHS).zfill(2)}.pt")
        torch.save(optimizer.state_dict(), f"stl10-{layers}/optimizer.pt")
    except TrainingInterrupt as ti:
        torch.save(ti.model.state_dict(), f"stl10-{layers}/epoch-{str(ti.epochs + 1).zfill(2)}.pt")
        torch.save(ti.optimizer.state_dict(), f"stl10-{layers}/optimizer.pt")
        raise KeyboardInterrupt
elif int(os.path.basename(sorted(glob.glob(f"stl10-{layers}/epoch-*.pt"))[-1]).split(".")[0].split("-")[
             -1]) < N_EPOCHS:
    model = resnet_init(layers, N_CLASSES)
    model.load_state_dict(torch.load(sorted(glob.glob(f"stl10-{layers}/*.pt"))[-1]))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    optimizer.load_state_dict(torch.load(f"stl10-{layers}/optimizer.pt"))
    epoch = int(os.path.basename(sorted(glob.glob(f"stl10-{layers}/epoch-*.pt"))[-1]).split(".")[0].split("-")[-1])
    model.to(DEVICE)
    try:
        model, optimizer, _ = training_loop(model, criterion, optimizer, train_loader, valid_loader, N_EPOCHS, DEVICE,
                                            start_epoch=epoch)
        torch.save(model.state_dict(), f"stl10-{layers}/epoch-{str(N_EPOCHS).zfill(2)}.pt")
    except TrainingInterrupt as ti:
        torch.save(ti.model.state_dict(), f"stl10-{layers}/epoch-{str(ti.epochs + 1).zfill(2)}.pt")
        torch.save(ti.optimizer.state_dict(), f"stl10-{layers}/optimizer.pt")
        raise KeyboardInterrupt
modeltoeval = resnet_init(layers, N_CLASSES)
modeltoeval.load_state_dict(torch.load(sorted(glob.glob(f"stl10-{layers}/epoch-*.pt"))[-1]))
modeltoeval.eval()
modeltoeval.to(DEVICE)
lrp = LRP(modeltoeval)
i = iter(train_loader)
asd = next(i)
asd = next(i)
asd = next(i)
count = 0
method = "kmeans_concave"
out_dir = "results_" + method + "/"
if os.path.exists(out_dir):
    shutil.rmtree(out_dir)
os.mkdir(out_dir)
for asd in i:
    inp, out = asd
    inp = inp[0].unsqueeze(0).to(DEVICE)
    inp.requires_grad = True
    out = out[0].to(DEVICE)

    attribution = lrp.attribute(inp, target=out)
    ma = torch.max(attribution)
    mi = torch.min(attribution)
    print(ma)
    print(mi)
    original_image = np.transpose((inp.squeeze(0).cpu().detach().numpy() / 2) + 0.5, (1, 2, 0))
    grads = np.transpose(attribution.squeeze(0).cpu().detach().numpy(), (1, 2, 0))
    ngrads = (grads - np.min(grads)) / (np.max(grads) - np.min(grads))
    if method == "brute":
        nz = numpy.nonzero(ngrads > 0.51)
        transpose = np.array(nz, dtype=np.float32).transpose()[:, 0:2]
        hull = cv2.convexHull(transpose)
        mask = np.zeros_like(grads)[..., 0:1]
        mask = np.uint8(mask)
        int_ = np.int32(hull)
        cv2.fillPoly(mask, pts=[int_], color=(1, 1, 1))
    elif method.startswith("kmeans"):
        premask = ngrads * 255
        vectorized = np.float32(premask.reshape(-1, 1))
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

        K = 2
        attempts = 10
        ret, label, center = cv2.kmeans(vectorized, K, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)
        res = center[label.flatten()]
        res2 = res.reshape((original_image.shape))
        mask = (res2 - np.min(res2)) / (np.max(res2) - np.min(res2))
        if "concave" in method:
            nz = numpy.nonzero(mask > 0.001)
            transpose = np.array(nz, dtype=np.float32).transpose()[:, 0:2]
            point_collection = geometry.MultiPoint(list(transpose))

            concave_hull, edge_points = alpha_shape(point_collection, alpha=0.6)
            mask = rasterio.features.rasterize([concave_hull], out_shape=(32, 32))
        if len(mask.shape) < 3:
            h, w, c = original_image.shape
            mask = mask.reshape((h, w, -1))
            if mask.shape[2] != c:
                mask = np.dstack((mask,) * c)

    # boundary_points = np.int32(np.vstack(ch.boundary.exterior.coords.xy).T)
    # # boundary_points is a subset of pts corresponding to the concave hull
    # mask = np.zeros_like(mask)
    # cv2.fillPoly(mask, pts=[boundary_points], color=(1))
    mask = mask * 3 / 4 + 0.25
    mask_gray = mask * 255
    # cv2.imwrite(str(count+1).zfill(3) +"-mask.png", mask_gray)
    highlight = mask * (original_image - 0.5) * 255 * 2
    # cv2.imwrite(str(count+1).zfill(3) +"-org.png", highlight)
    orig_img = (original_image - 0.5) * 255 * 2
    # cv2.imwrite(str(count+1).zfill(3) +"-org2.png", orig_img)
    im_h = np.hstack([orig_img, mask_gray, highlight])
    tuo = out.item()
    cv2.imwrite(out_dir + str(count + 1).zfill(3) + "-instance-class-" + str(tuo) + ".png", im_h)
    # _ = visualize_image_attr(grads, original_image, "blended_heat_map", sign="all", show_colorbar=True,
    #                          title="Overlayed Gradient Magnitudes: " + str(out))
    # _ = visualize_image_attr(None, np.uint8((original_image - 0.5) * mask + 0.5), method="original_image",
    #                          title="Original Image")
    count += 1
    if count == 500:
        break
pass
