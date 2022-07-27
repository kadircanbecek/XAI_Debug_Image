import glob
import os.path
import shutil

import rasterio.features
from shapely import geometry
from tqdm import tqdm

from ResNet import resnet_init
from concavehull_scipy import alpha_shape

import cv2
import numpy
from captum.attr import LRP, NoiseTunnel, IntegratedGradients
import numpy as np
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from captum.attr._utils.visualization import visualize_image_attr, _normalize_image_attr
from torch.utils.data import DataLoader

from torchvision import datasets, transforms, models

import matplotlib.pyplot as plt


def random_split(dataset, ratio=0.9, random_state=None):
    if random_state is not None:
        state = torch.random.get_rng_state()
        torch.random.manual_seed(random_state)
    n = int(len(dataset) * ratio)
    split = torch.utils.data.random_split(dataset, [n, len(dataset) - n])
    if random_state is not None:
        torch.random.set_rng_state(state)
    return split


# check device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# parameters
RANDOM_SEED = 42
LEARNING_RATE = 0.001
BATCH_SIZE = 32
N_EPOCHS = 40

IMG_SIZE = 224
N_CLASSES = 2


class TrainingInterrupt(Exception):
    def __init__(self, epochs, model, optimizer):
        super(TrainingInterrupt, self).__init__()
        self.epochs = epochs
        self.model = model
        self.optimizer = optimizer


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
valid_transform2 = transforms.Compose(
    [transforms.Resize((IMG_SIZE, IMG_SIZE)),
     transforms.ToTensor()])


# download and create datasets


class WorkaroundDataset(datasets.OxfordIIITPet):
    def __init__(self, root, split, download, transform):
        super(WorkaroundDataset, self).__init__(root=root, split=split, download=download, transform=transform)
        self.classes = ["Cat", "Dog"]
        self.class_to_idx = {k: v for k, v in enumerate(self.classes)}

    def __getitem__(self, idx):
        item = list(super(WorkaroundDataset, self).__getitem__(idx))
        item[1] = 0 if item[1] in [0, 5, 6, 7, 9, 11, 20, 23, 26, 27, 32, 33] else 1
        return tuple(item)


train_dataset = WorkaroundDataset(root='./data', split="trainval",
                                  download=True, transform=train_transform)
val_dataset = WorkaroundDataset(root='./data', split="test",
                                download=True, transform=valid_transform)
val_dataset2 = WorkaroundDataset(root='./data', split="test",
                                 download=True, transform=valid_transform2)

# train_ds, test_ds = random_split(dataset, 0.9, random_state=42)
# define the data loaders
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=BATCH_SIZE,
                          shuffle=True)

valid_loader = DataLoader(dataset=val_dataset,
                          batch_size=BATCH_SIZE,
                          shuffle=False)
valid_loader2 = DataLoader(dataset=val_dataset2,
                           batch_size=1,
                           shuffle=True)

classes = val_dataset.classes

# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(3, 6, 5)
#         self.pool1 = nn.MaxPool2d(2, 2)
#         self.pool2 = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(6, 16, 5)
#         self.fc1 = nn.Linear(16 * 5 * 5, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 37)
#         self.relu1 = nn.ReLU()
#         self.relu2 = nn.ReLU()
#         self.relu3 = nn.ReLU()
#         self.relu4 = nn.ReLU()
#
#     def forward(self, x):
#         x = self.pool1(self.relu1(self.conv1(x)))
#         x = self.pool2(self.relu2(self.conv2(x)))
#         x = x.view(-1, 16 * 5 * 5)
#         x = self.relu3(self.fc1(x))
#         x = self.relu4(self.fc2(x))
#         x = self.fc3(x)
#         return x

layers = 18
model = resnet_init(layers, len(train_dataset.classes))

torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
if not glob.glob(f"oxfordiiit-{layers}/*.pt"):
    if os.path.exists(f"oxfordiiit-{layers}/"):
        shutil.rmtree(f"oxfordiiit-{layers}")
    os.mkdir(f"oxfordiiit-{layers}/")
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    try:
        model, optimizer, _ = training_loop(model, criterion, optimizer, train_loader, valid_loader, N_EPOCHS, DEVICE)
        torch.save(model.state_dict(), f"oxfordiiit-{layers}/epoch-{str(N_EPOCHS).zfill(2)}.pt")
        torch.save(optimizer.state_dict(), f"oxfordiiit-{layers}/optimizer.pt")
    except TrainingInterrupt as ti:
        if ti.epochs > 0:
            torch.save(ti.model.state_dict(), f"oxfordiiit-{layers}/epoch-{str(ti.epochs).zfill(2)}.pt")
            torch.save(ti.optimizer.state_dict(), f"oxfordiiit-{layers}/optimizer.pt")
        raise KeyboardInterrupt
elif int(os.path.basename(sorted(glob.glob(f"oxfordiiit-{layers}/epoch-*.pt"))[-1]).split(".")[0].split("-")[
             -1]) < N_EPOCHS:
    model.load_state_dict(torch.load(sorted(glob.glob(f"oxfordiiit-{layers}/epoch-*.pt"))[-1]))
    model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    optimizer.load_state_dict(torch.load(f"oxfordiiit-{layers}/optimizer.pt"))
    epoch = int(os.path.basename(sorted(glob.glob(f"oxfordiiit-{layers}/epoch-*.pt"))[-1]).split(".")[0].split("-")[-1])
    try:
        model, optimizer, _ = training_loop(model, criterion, optimizer, train_loader, valid_loader, N_EPOCHS, DEVICE,
                                            start_epoch=epoch)
        torch.save(model.state_dict(), f"oxfordiiit-{layers}/epoch-{str(N_EPOCHS).zfill(2)}.pt")
    except TrainingInterrupt as ti:
        torch.save(ti.model.state_dict(), f"oxfordiiit-{layers}/epoch-{str(ti.epochs + 1).zfill(2)}.pt")
        torch.save(ti.optimizer.state_dict(), f"oxfordiiit-{layers}/optimizer.pt")
        raise KeyboardInterrupt

model.load_state_dict(torch.load(sorted(glob.glob(f"oxfordiiit-{layers}/epoch-*.pt"))[-1]))
model.eval()
model.to(DEVICE)
lrp = IntegratedGradients(model)
i = iter(valid_loader2)

method = "kmeans_concave"
out_dir = "results_" + method + "/"
if os.path.exists(out_dir):
    shutil.rmtree(out_dir)
os.mkdir(out_dir)
noise_tunnel = NoiseTunnel(lrp)
limit = 500
for count, asd in tqdm(enumerate(i), total=limit):
    inp, out_orig = asd
    norm = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    input_norm = norm(inp)
    input_norm = input_norm[0].unsqueeze(0).to(DEVICE)
    input_norm.requires_grad = True
    output = model(input_norm)
    output = F.softmax(output, dim=1)
    score, out = torch.topk(output, 1)


    # attribution = lrp.attribute(inp, target=out)

    def get_circular_kernel(diameter):
        mid = (diameter - 1) / 2
        distances = np.indices((diameter, diameter)) - np.array([mid, mid])[:, None, None]
        kernel = ((np.linalg.norm(distances, axis=0) - mid) <= 0).astype(int)

        return kernel


    attributions_ig_nt = lrp.attribute(input_norm, target=out)
    # ma = torch.max(attributions_ig_nt)
    # mi = torch.min(attributions_ig_nt)
    # print(ma)
    # print(mi)
    original_image = np.transpose(inp.squeeze(0).cpu().detach().numpy(), (1, 2, 0))
    grads = np.transpose(attributions_ig_nt.squeeze(0).cpu().detach().numpy(), (1, 2, 0))
    norm_img = _normalize_image_attr(grads, "positive", 2)
    ngrads = norm_img
    # if method == "brute":
    #     nz = numpy.nonzero(ngrads > 0.2)
    #     transpose = np.array(nz, dtype=np.float32).transpose()[:, 0:2]
    #     hull = cv2.convexHull(transpose)
    #     mask = np.zeros_like(grads)[..., 0:1]
    #     mask = np.uint8(mask)
    #     int_ = np.int32(hull)
    #     cv2.fillPoly(mask, pts=[int_], color=(1, 1, 1))
    # elif method.startswith("kmeans"):
    #     premask = ngrads * 255
    #     vectorized = np.float32(premask.reshape(-1, 1))
    #     criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    #
    #     K = 2
    #     attempts = 10
    #     ret, label, center = cv2.kmeans(vectorized, K, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)
    #     res = center[label.flatten()]
    #     res2 = res.reshape((original_image.shape))
    #     mask = (res2 - np.min(res2)) / (np.max(res2) - np.min(res2))
    #     if "concave" in method:
    #         nz = numpy.nonzero(ngrads > (np.true_divide(ngrads.sum(), (ngrads != 0).sum())*2))
    #         transpose = np.array(nz, dtype=np.float32).transpose()[:, 0:2]
    #         point_collection = geometry.MultiPoint(list(transpose))
    #
    #         concave_hull, edge_points = alpha_shape(point_collection, alpha=0.4)
    #         mask = rasterio.features.rasterize([concave_hull], out_shape=(32, 32)).transpose()
    #     if len(mask.shape) < 3:
    #         h, w, c = original_image.shape
    #         mask = mask.reshape((h, w, -1))
    #         if mask.shape[2] != c:
    #             mask = np.dstack((mask,) * c)

    # boundary_points = np.int32(np.vstack(ch.boundary.exterior.coords.xy).T)
    # # boundary_points is a subset of pts corresponding to the concave hull
    # mask = np.zeros_like(mask)
    # cv2.fillPoly(mask, pts=[boundary_points], color=(1))
    h, w, c = original_image.shape

    kernel = np.uint8(get_circular_kernel(5))
    mask = norm_img
    mask[mask > 0.25] = 1
    mask = cv2.dilate(mask, kernel)

    blur = cv2.GaussianBlur(mask * 255, (13, 13), 11)
    blur = np.uint8(blur)
    heatmap_img = cv2.applyColorMap(blur, cv2.COLORMAP_JET)

    if len(mask.shape) < 3:
        mask = mask.reshape((h, w, -1))
    if mask.shape[2] != c:
        mask = np.dstack((mask,) * c)
    mask = mask * 0.9 + 0.1
    mask_gray = mask * 255
    # cv2.imwrite(str(count+1).zfill(3) +"-mask.png", mask_gray)
    orig_img = np.uint8((original_image) * 255)
    orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
    highlight = cv2.addWeighted(heatmap_img, 0.5, orig_img, 0.5, 0)

    # cv2.imwrite(str(count+1).zfill(3) +"-org.png", highlight)
    # cv2.imwrite(str(count+1).zfill(3) +"-org2.png", orig_img)
    im_h = np.hstack(
        [cv2.resize(orig_img, [orig_img.shape[1] // 2, orig_img.shape[0] // 2], interpolation=cv2.INTER_AREA),
         cv2.resize(heatmap_img, [heatmap_img.shape[1] // 2, heatmap_img.shape[0] // 2], interpolation=cv2.INTER_AREA)])

    im_v = np.vstack([im_h, highlight])
    # cv2.imwrite(str(count+1).zfill(3) +"-org2.png", orig_img)
    tuo = out.item()
    cv2.imwrite(out_dir + str(count + 1).zfill(3) + "-instance-class-" + str(classes[tuo]) + "-" + str(
        classes[out_orig.item()]) + "-" + str(
        round(score.squeeze().item(), 3)) + ".png", im_v)
    cv2.imwrite(out_dir + str(count + 1).zfill(3) + "-instance-class-" + str(classes[tuo]) + "-" + str(
        classes[out_orig.item()]) + "-norm_mask.png",
                np.uint8(norm_img * 255))
    # _ = visualize_image_attr(grads, original_image, "blended_heat_map", sign="all", show_colorbar=True,
    #                          title="Overlayed Gradient Magnitudes: " + str(out))
    # _ = visualize_image_attr(None, np.uint8((original_image - 0.5) * mask + 0.5), method="original_image",
    #                          title="Original Image")
    if count+1 == limit:
        break
pass
