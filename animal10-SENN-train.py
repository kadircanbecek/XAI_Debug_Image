import glob
import os
import sys
import time
from collections import OrderedDict
from datetime import datetime

import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms import transforms
from PIL import Image
from torchvision.models import resnet18
from torchvision.utils import make_grid
from tqdm import tqdm
from lime import lime_image

from utils.losses import BVAE_loss, mnist_robustness_loss
from utils.models import TwoPartResNetSENN

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# parameters
RANDOM_SEED = 42
LEARNING_RATE = 0.001
BATCH_SIZE = 8
N_EPOCHS = 20

IMG_SIZE = 32
N_CLASSES = 10
concept_reg = 1.0
beta = 4.0

robust_reg = float(sys.argv[1])
print(robust_reg)


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
    plt.close(fig)

    # change the plot style to default
    plt.style.use('default')


def training_loop(model, criterionVAE, criterionCl, criterionRobust, optimizer, train_loader, valid_loader, epochs,
                  device, dir_out, print_every=1):
    '''
    Function defining the entire training loop
    '''

    # set objects for storing metrics
    best_loss = 1e10
    train_losses = []
    valid_losses = []

    # Train model
    for epoch in range(0, epochs):

        # training
        model, optimizer, train_loss, train_acc = train(train_loader, model, criterionVAE, criterionCl, criterionRobust,
                                                        optimizer, device)
        train_losses.append(train_loss)

        # validation
        with torch.no_grad():
            model, valid_loss, valid_acc = validate(valid_loader, model, criterionVAE, criterionCl, criterionRobust,
                                                    device)
            valid_losses.append(valid_loss)
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

    plot_losses(train_losses, valid_losses)

    return model, optimizer, (train_losses, valid_losses)


class FeatureExtract(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        rn18 = resnet18(pretrained=pretrained)
        self.conv1 = rn18.conv1
        self.bn1 = rn18.bn1
        self.relu = rn18.relu
        self.maxpool = rn18.maxpool
        self.layer1 = rn18.layer1
        self.layer2 = rn18.layer2
        self.layer3 = rn18.layer3
        self.layer4 = rn18.layer4

        self.layer_nlm = nn.Conv2d(512, 32, kernel_size=1, stride=1, padding=0, bias=False)

        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.layer_nlm(x)
        x = self.gap(x).reshape([-1, 32])

        return x


class Classifier(nn.Module):
    def __init__(self, classes):
        super().__init__()
        self.linear = nn.Linear(32, classes)

    def forward(self, x):
        x = self.linear(x)

        return x

    print(robust_reg)


def train(train_loader, model, criterionVAE, criterionCl, criterionRobust, optimizer, device):
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
        X.requires_grad_(True)
        # Forward pass
        predictions, explanations, x_reconst = model(X)
        concepts, (mu, logvar), relevances = explanations
        robustness_loss = criterionRobust(X, predictions, concepts, relevances)
        classification_loss = criterionCl(predictions, y_true)
        recon_loss, kl_div = criterionVAE(X, x_reconst, mu, logvar)
        concept_loss = recon_loss + beta * kl_div
        loss = classification_loss + \
               robust_reg * robustness_loss + \
               concept_reg * concept_loss
        running_loss += loss.item() * X.size(0)
        _, predicted_labels = torch.max(predictions, 1)

        n += y_true.size(0)
        correct_pred += (predicted_labels == y_true).sum()

        # Backward pass
        loss.backward()
        optimizer.step()

    epoch_loss = running_loss / len(train_loader.dataset)
    accuracy = correct_pred.float() / n
    return model, optimizer, epoch_loss, accuracy


def validate(valid_loader, model, criterionVAE, criterionCl, criterionRobust, device):
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
        X.requires_grad_(True)

        # Forward pass and record loss

        predictions, explanations, x_reconst = model(X)
        concepts, (mu, logvar), relevances = explanations
        robustness_loss = criterionRobust(X, predictions, concepts, relevances)
        classification_loss = criterionCl(predictions, y_true)
        recon_loss, kl_div = criterionVAE(X, x_reconst, mu, logvar)
        concept_loss = recon_loss + beta * kl_div
        loss = classification_loss + \
               robust_reg * robustness_loss + \
               concept_reg * concept_loss
        running_loss += loss.item() * X.size(0)
        _, predicted_labels = torch.max(predictions, 1)

        n += y_true.size(0)
        correct_pred += (predicted_labels == y_true).sum()

    epoch_loss = running_loss / len(valid_loader.dataset)
    accuracy = correct_pred.float() / n

    return model, epoch_loss, accuracy


class Animal10(Dataset):
    def __init__(self, root_dir, transform=None):

        self.root_dir = root_dir
        self.transform = transform
        self.files = glob.glob(os.path.join(root_dir, "**/*.png"), recursive=True)
        self.classes = sorted(set(list([os.path.dirname(f).split("/")[-1] for f in self.files])))
        self.class_to_idx = {k: v for v, k in enumerate(self.classes)}

        self.idx_to_class = {k: v for k, v in enumerate(self.classes)}

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.files[idx]
        image = Image.open(img_name).convert('RGB')

        if self.transform:
            image = self.transform(image)

        sample = (image, self.class_to_idx[os.path.dirname(img_name).split("/")[-1]])
        return sample


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Resize(size=(224, 224)),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
transform_for_tensor = transforms.Compose(
    [transforms.Resize(size=(224, 224)),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
a10_train = Animal10("./data/animals-10/train",
                     transform=transform)

a10_val = Animal10("./data/animals-10/test",
                   transform=transform)

a10_val_wo_transform = Animal10("./data/animals-10/test",
                                transform=transforms.Compose(
                                    [transforms.ToTensor()]))
sample_0 = a10_train[0]
# imaj = sample_0["image"].numpy()
# print(imaj.shape)
# cv2.imshow(a10.idx_to_class[sample_0["class"]], imaj.transpose(1, 2, 0))
# cv2.waitKey()
# print()
train_loader = DataLoader(a10_train, batch_size=BATCH_SIZE,
                          shuffle=True, num_workers=4)

valid_loader = DataLoader(a10_val, batch_size=BATCH_SIZE,
                          shuffle=True, num_workers=4)
# model_fe = FeatureExtract()
# model_cl = Classifier(len(a10_train.classes))
# inp = torch.rand([1, 3, 224, 224])
# outp = model_fe(inp)
# out_classes = model_cl(outp)
# print(out_classes.shape)

torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
out_dir = f"animal10-SENN/resnet18-{robust_reg}/"
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
model = TwoPartResNetSENN(classes=len(a10_train.classes))
model.to(DEVICE)

if not os.path.exists(os.path.join(out_dir, "last.pt")):
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterionVAE = BVAE_loss
    criterionCL = nn.CrossEntropyLoss()
    criterionRobust = mnist_robustness_loss
    model, optimizer, _ = training_loop(model, criterionVAE, criterionCL, criterionRobust, optimizer, train_loader,
                                        valid_loader, N_EPOCHS, DEVICE,
                                        out_dir)
    torch.save(model.state_dict(), os.path.join(out_dir, "last.pt"))
model.load_state_dict(torch.load(os.path.join(out_dir, "best.pt")))
model.eval()

for i, (X, y) in enumerate(a10_val):
    X = X.to(DEVICE)
    X = X.unsqueeze(0)
    feats = model.resnetSENN(X)
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

    logits = model.resnetSENN(batch)
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


def explain(x, num_prototypes=20, traversal_range=0.45, use_cdf=True,
            show=False, save_as=None, gridsize=(1, 6), col_span=3, figure_size=(18, 3)):
    model.eval()
    y_pred, explanations = model.forward(x.unsqueeze(0))
    concepts, (x_posterior_mean, x_posterior_logvar), relevances = explanations
    x_posterior_mean = x_posterior_mean.squeeze(-1)
    x_posterior_logvar = x_posterior_logvar.squeeze(-1)

    num_concepts = concepts.shape[1]

    # generate new concept vector for each prototype
    # by traversing independently in each dimension
    concepts_sample = concepts.repeat(num_prototypes, 1)
    mean = x_posterior_mean.cpu().detach().numpy()
    std = torch.exp(x_posterior_logvar.detach() / 2).cpu().numpy()
    concepts_traversals = [model.traverse(concepts_sample, dim, traversal_range,
                                          num_prototypes, mean[:, dim], std[:, dim], use_cdf)
                           for dim in range(num_concepts)]
    concepts_traversals = torch.cat(concepts_traversals, dim=0)
    prototypes = model.vae_conceptizer.decoder(concepts_traversals)
    prototype_imgs = prototypes.view(-1, x.shape[0], x.shape[1], x.shape[2])

    # nrow is number of images in a row which must be the number of prototypes
    prototype_grid_img = make_grid(prototype_imgs, nrow=num_prototypes).cpu().detach().numpy()

    # prepare to plot
    relevances = relevances.squeeze(0).cpu().detach().numpy()
    predict_class = y_pred.argmax(1).item()
    relevances_pred = relevances[:, predict_class]
    concepts = concepts.squeeze(0)
    product_pred = concepts * relevances_pred
    pred_colors = ['g' if r > 0 else 'r' for r in product_pred]

    # plot input image, relevances, concepts, prototypes side by side
    plt.style.use('seaborn-paper')
    fig = plt.figure(figsize=figure_size)
    ax1 = plt.subplot2grid(gridsize, (0, 0))
    ax2 = plt.subplot2grid(gridsize, (0, 1))
    ax4 = plt.subplot2grid(gridsize, (0, 2), colspan=col_span)

    ax1.imshow(x.cpu().numpy().squeeze(), cmap='gray')
    ax1.set_axis_off()
    ax1.set_title(f'Input Prediction: {y_pred.argmax(1).item()}', fontsize=18)

    ax2.barh(range(num_concepts), product_pred, color=pred_colors)
    ax2.set_xlabel(f"Class:{predict_class} Contribution", fontsize=18)
    ax2.xaxis.set_label_position('top')
    ax2.tick_params(axis='x', which='major', labelsize=12)
    ax2.set_yticks([])

    ax4.imshow(prototype_grid_img.transpose(1, 2, 0))
    ax4.set_title('Prototypes', fontsize=18)
    ax4.set_axis_off()

    fig.tight_layout()

    if save_as is not None: fig.savefig(save_as)
    if show: plt.show()
    plt.close()


#
# explainer = lime_image.LimeImageExplainer()
# 
# for i, row in enumerate(index_per_feat):
#     if not os.path.exists(f"animal10-SENN-results/results-{robust_reg}/{i}/"):
#         os.makedirs(f"animal10-SENN-results/results-{robust_reg}/{i}/")
#     start = time.time()
#     weight = model.cl.linear.weight.cpu().detach().numpy()
#     fig, ax = visualize_weights(weight, i, a10_val.classes)
#     fig.savefig(f"animal10-SENN-results/results-{robust_reg}/{i}/weight_plot.png")
#     plt.close(fig)
#     class_inst = {}
#     for col in row.tolist():
#         f = a10_val_wo_transform.files[col]
#         cname = os.path.dirname(f).split("/")[-1]
#         class_inst[cname] = class_inst.get(cname, 0) + 1
#     print(class_inst)
#     D = class_inst
#     plt.bar(range(len(D)), list(D.values()), align='center')
#     plt.xticks(range(len(D)), list(D.keys()))
#     plt.savefig(f"animal10-SENN-results/results-{robust_reg}/{i}/most_activated.png")
#     plt.close()
#     row = row[:12]
#     for j, col in enumerate(row.tolist()):
#         image, label = a10_val_wo_transform[col]
# 
#         image2 = np.uint8(image.numpy().transpose(1, 2, 0) * 255)
#         image_org = image2.copy()
#         image3 = cv2.resize(image2, (224, 224))
#         explanation = explainer.explain_instance(image3, batch_predict, (i,),
#                                                  top_labels=None,
#                                                  hide_color=0,
#                                                  num_samples=1000,
#                                                  batch_size=256,
#                                                  random_seed=RANDOM_SEED)
#         _, mask = explanation.get_image_and_mask(i)
#         h, w = image_org.shape[:2]
#         mask = np.float32(cv2.resize(np.float32(mask), (w, h)) > 0)
#         image2 = np.uint8(image2 * np.dstack([mask for _ in range(3)]))
# 
#         image2 += np.uint8(np.ones_like(image2) * 127 * np.dstack([1 - mask for _ in range(3)]))
# 
#         cv2.imwrite(f"animal10-SENN-results-{robust_reg}/{i}/{j}-{col}.png",
#                     np.uint8(cv2.cvtColor(np.vstack([image2, image_org]), cv2.COLOR_RGB2BGR)))
#         print(f"{robust_reg}/{i}/{j}-{col}")
#     end = time.time()
# 
#     print("Feature-time:", end - start)

# for i, row in enumerate(index_per_feat):
#     images = []
#     for j, col in enumerate(row):
#         image, label = a10_val_wo_transform[col]
#         image2 = np.uint8(image.numpy().transpose(1, 2, 0) * 255)
#         images.append(image2)
#     if not os.path.exists(f"results/{i}/"):
#         os.makedirs(f"results/{i}/")
#
#     explanation = explainer.explain_instance(images, batch_predict, (i,),
#                                              top_labels=None,
#                                              hide_color=0,
#                                              num_samples=1000)
#     _, mask = explanation.get_image_and_mask(i)
#     h, w = image2.shape[:2]
#     mask = np.float32(cv2.resize(mask, (w, h)) > 0)
#     image2 = np.uint8(image2 * np.dstack([mask for _ in range(3)]))
#
#     image2 += np.uint8(np.ones_like(image2) * 127 * np.dstack([1 - mask for _ in range(3)]))
#
#     cv2.imwrite(f"results/{i}/{j}-{col}.png",
#                 np.uint8(cv2.cvtColor(np.vstack([image2, image_org]), cv2.COLOR_RGB2BGR)))
del model
torch.cuda.empty_cache()
pass
