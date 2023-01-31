import glob
import os
import sys
import time
from collections import OrderedDict
from datetime import datetime
from itertools import combinations

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
from tqdm import tqdm
from lime import lime_image

from utils.utils import plot_confusion_matrix

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# parameters
RANDOM_SEED = 42
LEARNING_RATE = 0.001
BATCH_SIZE = 32
N_EPOCHS = 90

IMG_SIZE = 32
N_CLASSES = 10

lambda_corr = float(sys.argv[1])
print(lambda_corr)


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

    # Train model
    for epoch in range(0, epochs):

        # training
        model, optimizer, train_loss, train_acc = train(train_loader, model, criterion, optimizer, device)
        train_losses.append(train_loss)

        # validation
        with torch.no_grad():
            model, valid_loss, valid_acc = validate(valid_loader, model, criterion, device)
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


print(lambda_corr)


def correlationloss(output):
    output = output.cpu()
    batch, dim = output.shape

    mean_of_batch = torch.mean(output)
    ones_vector = torch.ones((batch, dim))
    corr_mat_1 = output - mean_of_batch * ones_vector
    corr_mat_2 = torch.transpose(corr_mat_1, 0, 1)
    corr_mat = torch.matmul(corr_mat_2, corr_mat_1)
    loss = (1 / (dim ** 2)) * torch.sum(torch.abs(corr_mat))
    return loss


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
        fe = model.fe(X)
        y_hat = model.cl(fe)
        loss = criterion(y_hat, y_true)
        loss += lambda_corr * correlationloss(fe)
        running_loss += loss.item() * X.size(0)
        _, predicted_labels = torch.max(y_hat, 1)

        n += y_true.size(0)
        correct_pred += (predicted_labels == y_true).sum()

    epoch_loss = running_loss / len(valid_loader.dataset)
    accuracy = correct_pred.float() / n

    return model, epoch_loss, accuracy


class WholeModel(nn.Module):
    def __init__(self, classes, pretrained=True):
        super().__init__()
        self.fe = FeatureExtract(pretrained)
        self.cl = Classifier(classes)

    def forward(self, x):
        x = self.fe(x)
        x = self.cl(x)
        return x


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
a10_train = Animal10("./data/animals-3/backup/train",
                     transform=transform)

a10_val = Animal10("./data/animals-3/backup/test",
                   transform=transform)

a10_val_wo_transform = Animal10("./data/animals-3/backup/test",
                                transform=transforms.Compose(
                                    [transforms.ToTensor()]))
sample_0 = a10_train[0]
# imaj = sample_0["image"].numpy()
# print(imaj.shape)
# cv2.imshow(a10.idx_to_class[sample_0["class"]], imaj.transpose(1, 2, 0))
# cv2.waitKey()
# print()
train_loader = DataLoader(a10_train, batch_size=128,
                          shuffle=True, num_workers=4)

valid_loader = DataLoader(a10_val, batch_size=128,
                          shuffle=True, num_workers=4)
# model_fe = FeatureExtract()
# model_cl = Classifier(len(a10_train.classes))
# inp = torch.rand([1, 3, 224, 224])
# outp = model_fe(inp)
# out_classes = model_cl(outp)
# print(out_classes.shape)

torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
out_dir = f"animal3-backup/resnet18-{lambda_corr}/"
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
model = WholeModel(len(a10_train.classes), pretrained=False)
model.to(DEVICE)

if not os.path.exists(os.path.join(out_dir, "last.pt")):
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    model, optimizer, _ = training_loop(model, criterion, optimizer, train_loader, valid_loader, N_EPOCHS, DEVICE,
                                        out_dir)
    torch.save(model.state_dict(), os.path.join(out_dir, "last.pt"))
    exit()
model.load_state_dict(torch.load(os.path.join(out_dir, "best.pt")))
model.eval()
non_point_ = [10, 18, 19, 23, 30]
cats_lp = [4, 8, 15, 17, 13, 22, 2, 14, 9]
cats_mp = [20, 29, 1, 16, 24, 5, 12, 21, 11]
cats_hp = [6, 7, 26, 27, 28, 31, 3, 25, 0]
cows_lp = [15, 20, 22, 24, 29, 0, 8, 17, 21]
cows_mp = [13, 14, 25, 3, 6, 7, 2, 5, 1]
cows_hp = [12, 28, 31, 11, 9, 27, 26, 4, 16]
spiders_lp = [16, 26, 3, 27, 0, 25, 28, 31, 11]
spiders_mp = [4, 7, 6, 12, 21, 5, 9, 1, 2]
spiders_hp = [14, 13, 24, 20, 29, 8, 17, 22, 15]
bias = model.cl.linear.bias.cpu().detach().numpy()
print("bias", bias)

ll = [l for l in cats_lp if l in cows_lp and l in spiders_lp]

print(ll)


def disable_weight(weight, disabled_feats, classes=None):
    disable_mat = torch.ones_like(weight)
    if classes is None:
        disable_mat[:, disabled_feats] = 0
    else:
        disable_mat[classes, disabled_feats] = 0
    weight = weight * disable_mat
    return weight


# co = sorted(cats_hp)
# model.load_state_dict(torch.load(os.path.join(out_dir, "best.pt")))
# 
# model.cl.linear.weight = torch.nn.Parameter(disable_weight(model.cl.linear.weight, co))
# print(model.cl.linear.weight.detach().cpu().numpy())
# preds = []
# class_inst = np.zeros((3, 3))
# 
# for i, (X, y) in enumerate(a10_val):
#     X = X.to(DEVICE)
#     X = X.unsqueeze(0)
#     outs = model(X)
#     _, predicted_labels = torch.max(outs, 1)
#     cla = predicted_labels.item()
#     if cla + y == 0:
#         pass
#     label = y
#     preds.append(cla)
#     class_inst[label, cla] = class_inst[label, cla] + 1
# fig, ax = plot_confusion_matrix(class_inst, a10_val.classes)
# fig.savefig(f"animal3-results-backup/results-{lambda_corr}/conf-mat-cat-wo-hp-cat-feature.png")
# plt.close(fig)
# exit()
cattoes = [("cat", [("lp", cats_lp), ("mp", cats_mp), ("hp", cats_hp)]),
           ("cow", [("lp", cows_lp), ("mp", cows_mp), ("hp", cows_hp)]),
           ("spider", [("lp", spiders_lp), ("mp", spiders_mp), ("hp", spiders_hp)]), ]


class ScoreKeeperAll:
    def __init__(self):
        self.total_acc = 0
        self.total_tp = 0
        self.total_fp = 0
        self.total_fn = 0
        self.total_tn = 0
        self.score_insts = []
        self.macro_f1 = 0
        self.micro_f1 = 0
        self.weighted_f1 = 0
        pass

    def __str__(self):
        text = ""
        text += "Total Acc: "
        text += str(round(self.total_acc, 3))
        text += "\n"
        text += "Macro F1: "
        text += str(round(self.macro_f1, 3))
        text += "\n"
        text += "Micro F1: "
        text += str(round(self.micro_f1, 3))
        text += "\n"

        for scoreskeep in self.score_insts:
            text += str(scoreskeep)
            text += "\n"

        return text


class ScoreKeeperInst:
    def __init__(self, cname):
        self.cname = cname
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.tn = 0
        self.acc = 0
        self.f1 = 0
        pass

    def __str__(self):
        text = f"{self.cname}:\n"
        text += f"Total Acc: "
        text += str(round(self.acc, 3))
        text += "\n"
        text += "Macro F1: "
        text += str(round(self.f1, 3))
        text += "\n"
        return text


# cat1 = combinations([cats_lp, cats_mp, cats_hp], 1)
def scores(class_inst):
    scoreall = ScoreKeeperAll()
    scoreall.total_acc = np.sum(np.diagonal(class_inst)) / np.sum(class_inst)

    for i in range(len(class_inst)):
        score_inst = ScoreKeeperInst(a10_val.classes[i])
        score_inst.tp = tp = class_inst[i, i]
        scoreall.total_tp += tp
        score_inst.fp = fp = np.sum(class_inst[:, i]) - tp
        scoreall.total_fp += fp
        score_inst.fn = fn = np.sum(class_inst[i, :]) - tp
        scoreall.total_fn += fn
        score_inst.tn = tn = np.sum(class_inst) - fp - tp - fn
        scoreall.total_tn += tn
        score_inst.acc = accuracy = (tp + tn) / (tp + fp + fn + tn)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        score_inst.f1 = 2 * precision * recall / (precision + recall)
        scoreall.score_insts.append(score_inst)
    f1s = [inst.f1 for inst in scoreall.score_insts]
    scoreall.macro_f1 = sum(f1s) / len(f1s)
    microprecision = scoreall.total_tp / (scoreall.total_tp + scoreall.total_fp)
    microrecall = scoreall.total_tp / (scoreall.total_tp + scoreall.total_fn)
    scoreall.micro_f1 = 2 * microprecision * microrecall / (microprecision + microrecall)

    return scoreall


cat_cos = [[cats_mp[0],cats_mp[1],cats_mp[2]],[cats_mp[3],cats_mp[4],cats_mp[5]],[cats_mp[6],cats_mp[7],cats_mp[8]]]
for cat_co in cat_cos:
    cat_co = sorted(list(cat_co))
    model.load_state_dict(torch.load(os.path.join(out_dir, "best.pt")))

    model.cl.linear.weight = torch.nn.Parameter(disable_weight(model.cl.linear.weight, cat_co, classes=[1]))
    # print(model.cl.linear.weight.detach().cpu().numpy())
    preds = []
    class_inst = np.zeros((3, 3))

    for i, (X, y) in enumerate(a10_val):
        X = X.to(DEVICE)
        X = X.unsqueeze(0)
        outs = model(X)
        _, predicted_labels = torch.max(outs, 1)
        cla = predicted_labels.item()
        label = y
        preds.append(cla)
        class_inst[label, cla] = class_inst[label, cla] + 1
    fig, ax = plot_confusion_matrix(class_inst, a10_val.classes)
    fig.savefig(
        f"animal3-results-backup/results-{lambda_corr}/conf-mat-cat-wo-{cat_co[0]}-{cat_co[1]}-{cat_co[2]}-cat-feature.png")
    plt.close(fig)

    scorekeep = scores(class_inst)
    print(scorekeep)

    with open(
            f"animal3-results-backup/results-{lambda_corr}/scores-cat-wo-{cat_co[0]}-{cat_co[1]}-{cat_co[2]}-cat-feature.txt",
            "w+") as score_file:
        score_file.write(str(scorekeep))

    model.load_state_dict(torch.load(os.path.join(out_dir, "best.pt")))

    model.cl.linear.weight = torch.nn.Parameter(disable_weight(model.cl.linear.weight, cat_co))
    # print(model.cl.linear.weight.detach().cpu().numpy())
    preds = []
    class_inst = np.zeros((3, 3))

    for i, (X, y) in enumerate(a10_val):
        X = X.to(DEVICE)
        X = X.unsqueeze(0)
        outs = model(X)
        _, predicted_labels = torch.max(outs, 1)
        cla = predicted_labels.item()
        label = y
        preds.append(cla)
        class_inst[label, cla] = class_inst[label, cla] + 1
    fig, ax = plot_confusion_matrix(class_inst, a10_val.classes)
    fig.savefig(
        f"animal3-results-backup/results-{lambda_corr}/conf-mat-cat-wo-{cat_co[0]}-{cat_co[1]}-{cat_co[2]}-whole-feature.png")
    plt.close(fig)

    scorekeep = scores(class_inst)
    print(scorekeep)

    with open(
            f"animal3-results-backup/results-{lambda_corr}/scores-cat-wo-{cat_co[0]}-{cat_co[1]}-{cat_co[2]}-whole-feature.txt",
            "w+") as score_file:
        score_file.write(str(scorekeep))
exit()
model.cl.linear.weight = torch.nn.Parameter(disable_weight(model.cl.linear.weight, non_point_))
class_inst = np.zeros((3, 3))
preds = []
for i, (X, y) in enumerate(a10_val):
    X = X.to(DEVICE)
    X = X.unsqueeze(0)
    outs = model(X)
    _, predicted_labels = torch.max(outs, 1)
    cla = predicted_labels.item()
    label = y
    preds.append(cla)
    class_inst[label, cla] = class_inst[label, cla] + 1
fig, ax = plot_confusion_matrix(class_inst, a10_val.classes)
fig.savefig(f"animal3-results-backup/results-{lambda_corr}/conf-mat-non-point-disable.png")
plt.close(fig)
scorekeep = scores(class_inst)
print(scorekeep)

with open(f"animal3-results-backup/results-{lambda_corr}/scores-non-point-disable.txt",
          "w+") as score_file:
    score_file.write(str(scorekeep))

model.load_state_dict(torch.load(os.path.join(out_dir, "best.pt")))
class_inst = np.zeros((3, 3))
preds = []
for i, (X, y) in enumerate(a10_val):
    X = X.to(DEVICE)
    X = X.unsqueeze(0)
    outs = model(X)
    _, predicted_labels = torch.max(outs, 1)
    cla = predicted_labels.item()
    label = y
    preds.append(cla)
    class_inst[label, cla] = class_inst[label, cla] + 1
# fig, ax = plot_confusion_matrix(class_inst, a10_val.classes)
# fig.savefig(f"animal3-results-backup/results-{lambda_corr}/conf-mat-{classname}-wo-{dis}-whole-feature.png")
# plt.close(fig)
scorekeep = scores(class_inst)
print(scorekeep)

with open(f"animal3-results-backup/results-{lambda_corr}/scores-og.txt",
          "w+") as score_file:
    score_file.write(str(scorekeep))
for classname, cattos in cattoes:
    cat2 = combinations(cattos, 2)
    feat_to_remove = [a10_val.classes.index(classname)]
    # print(feat_to_remove)
    for dis, co in cattos:
        print(classname, dis)
        if dis == "hp":
            pass
        else:
            continue
        co = sorted(co)
        model.load_state_dict(torch.load(os.path.join(out_dir, "best.pt")))

        model.cl.linear.weight = torch.nn.Parameter(disable_weight(model.cl.linear.weight, co))
        # print(model.cl.linear.weight.detach().cpu().numpy())
        preds = []
        class_inst = np.zeros((3, 3))

        for i, (X, y) in enumerate(a10_val):
            X = X.to(DEVICE)
            X = X.unsqueeze(0)
            outs = model(X)
            _, predicted_labels = torch.max(outs, 1)
            cla = predicted_labels.item()
            label = y
            preds.append(cla)
            class_inst[label, cla] = class_inst[label, cla] + 1
        fig, ax = plot_confusion_matrix(class_inst, a10_val.classes)
        fig.savefig(f"animal3-results-backup/results-{lambda_corr}/conf-mat-{classname}-wo-{dis}-whole-feature.png")
        plt.close(fig)
        scorekeep = scores(class_inst)
        print(scorekeep)

        with open(f"animal3-results-backup/results-{lambda_corr}/scores-{classname}-wo-{dis}-whole-feature.txt",
                  "w+") as score_file:
            score_file.write(str(scorekeep))
        model.load_state_dict(torch.load(os.path.join(out_dir, "best.pt")))
        model.cl.linear.weight = torch.nn.Parameter(disable_weight(model.cl.linear.weight, co, feat_to_remove))
        # print(model.cl.linear.weight.detach().cpu().numpy())

        preds = []
        class_inst = np.zeros((3, 3))

        for i, (X, y) in enumerate(a10_val):
            X = X.to(DEVICE)
            X = X.unsqueeze(0)
            outs = model(X)
            _, predicted_labels = torch.max(outs, 1)
            cla = predicted_labels.item()
            label = y
            preds.append(cla)
            class_inst[label, cla] = class_inst[label, cla] + 1
        fig, ax = plot_confusion_matrix(class_inst, a10_val.classes)
        fig.savefig(
            f"animal3-results-backup/results-{lambda_corr}/conf-mat-{classname}-wo-{dis}-{classname}-feature.png")
        plt.close(fig)
        scorekeep = scores(class_inst)
        print(scorekeep)

        with open(f"animal3-results-backup/results-{lambda_corr}/scores-{classname}-wo-{dis}-{classname}-feature.txt",
                  "w+") as score_file:
            score_file.write(str(scorekeep))

    for co in cat2:
        (dis1, cat_co_1), (dis2, cat_co_2) = co
        dis = dis1 + "_" + dis2

        if "hp" in dis:
            pass
        else:
            continue
        print(classname, dis)
        cat_co = sorted(cat_co_1 + cat_co_2)
        model.load_state_dict(torch.load(os.path.join(out_dir, "best.pt")))

        model.cl.linear.weight = torch.nn.Parameter(disable_weight(model.cl.linear.weight, cat_co))
        # print(model.cl.linear.weight.detach().cpu().numpy())
        preds = []
        class_inst = np.zeros((3, 3))

        for i, (X, y) in enumerate(a10_val):
            X = X.to(DEVICE)
            X = X.unsqueeze(0)
            outs = model(X)
            _, predicted_labels = torch.max(outs, 1)
            cla = predicted_labels.item()
            label = y
            preds.append(cla)
            class_inst[label, cla] = class_inst[label, cla] + 1
        fig, ax = plot_confusion_matrix(class_inst, a10_val.classes)
        fig.savefig(f"animal3-results-backup/results-{lambda_corr}/conf-mat-{classname}-wo-{dis}-whole-feature.png")
        plt.close(fig)

        scorekeep = scores(class_inst)
        print(scorekeep)

        with open(f"animal3-results-backup/results-{lambda_corr}/scores-{classname}-wo-{dis}-whole-feature.txt",
                  "w+") as score_file:
            score_file.write(str(scorekeep))

        model.load_state_dict(torch.load(os.path.join(out_dir, "best.pt")))
        model.cl.linear.weight = torch.nn.Parameter(disable_weight(model.cl.linear.weight, cat_co, feat_to_remove))
        # print(model.cl.linear.weight.detach().cpu().numpy())
        preds = []
        class_inst = np.zeros((3, 3))

        for i, (X, y) in enumerate(a10_val):
            X = X.to(DEVICE)
            X = X.unsqueeze(0)
            outs = model(X)
            _, predicted_labels = torch.max(outs, 1)
            cla = predicted_labels.item()
            label = y
            preds.append(cla)
            class_inst[label, cla] = class_inst[label, cla] + 1
        fig, ax = plot_confusion_matrix(class_inst, a10_val.classes)
        fig.savefig(
            f"animal3-results-backup/results-{lambda_corr}/conf-mat-{classname}-wo-{dis}-{classname}-feature.png")
        plt.close(fig)

        scorekeep = scores(class_inst)
        print(scorekeep)

        with open(f"animal3-results-backup/results-{lambda_corr}/scores-{classname}-wo-{dis}-{classname}-feature.txt",
                  "w+") as score_file:
            score_file.write(str(scorekeep))
exit()
for i, (X, y) in enumerate(a10_val):
    X = X.to(DEVICE)
    X = X.unsqueeze(0)
    feats = model.fe(X)
    feats = feats.cpu().detach().numpy()
    if i == 0:
        feats_all = feats
    else:
        feats_all = np.concatenate([feats_all, feats], axis=0)


# ind = np.argsort(-feats_all, axis=0)
# feats_max = feats_all[ind, np.arange(feats_all.shape[1])]
# index_per_feat = ind.T
# index_per_feat_only_pos = []
# for i, row in enumerate(index_per_feat):
#     # print(row)
#     index_per_feat_only_pos_row = []
#     for c in row:
#         if feats_all[c, i] > 0:
#             index_per_feat_only_pos_row.append(c)
#     index_per_feat_only_pos.append(index_per_feat_only_pos_row)
#

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


def visualize_contribution(feat_w_, class_names, show=False):
    max_abs_W = np.max(np.abs(feat_w_)) + 0.1  # For plotting
    fig, ax = plt.subplots()
    fig.set_size_inches(9, 2)
    W = feat_w_
    ax.boxplot(W, vert=False, labels=class_names)
    ax.set_xlim((-max_abs_W, max_abs_W))
    ax.invert_yaxis()
    ax.set_xlabel('Weights')
    # ax.set_title(f'Feature {feature_idx}')
    # for i, v in enumerate(W[feature_idx]):
    #     if v >= 0:
    #         ax.text(v + 0.01, i + .05, '%.4f' % (v), color='black')
    #     else:
    #         ax.text(v - 0.13, i + .05, '%.4f' % (v), color='black')
    # if show:
    #     plt.show()
    return fig, ax


explainer = lime_image.LimeImageExplainer()
if not os.path.exists(f"animal3-results-backup/results-{lambda_corr}"):
    os.makedirs(f"animal3-results-backup/results-{lambda_corr}")

weight = model.cl.linear.weight.cpu().detach().numpy()
bias = model.cl.linear.bias.cpu().detach().numpy()
preds = np.argmax(feats_all @ weight.T + bias, axis=1)
class_inst = np.zeros((3, 3))
for i in range(len(a10_val)):
    X, label = a10_val[i]
    cla = preds[i]
    class_inst[label, cla] = class_inst[label, cla] + 1
fig, ax = plot_confusion_matrix(class_inst, a10_val.classes)
fig.savefig(f"animal3-results-backup/results-{lambda_corr}/conf-mat.png")
plt.close(fig)

# if lambda_corr == 0.001:
#     exit(0)
for i, row in enumerate(feats_all.T):
    # if i < 11:
    #     continue

    if not os.path.exists(f"animal3-results-backup/results-{lambda_corr}/{i}/"):
        os.makedirs(f"animal3-results-backup/results-{lambda_corr}/{i}/")
    start = time.time()
    w_feat = weight[:, i]
    feat = feats_all[:, i]
    feats_w_ = np.array([w_feat * f for f in feat])
    fig, ax = visualize_weights(weight, i, a10_val.classes)
    fig.savefig(f"animal3-results-backup/results-{lambda_corr}/{i}/weight_plot.png")
    plt.close(fig)
    fig, ax = visualize_contribution(feats_w_, a10_val.classes)
    fig.savefig(f"animal3-results-backup/results-{lambda_corr}/{i}/contr_plot.png")
    plt.close(fig)
    fwshape = feats_w_.shape
    feats_w_f = feats_w_.reshape(-1)
    feats_ind = np.argsort(-feats_w_f)
    feats_ind = list(set([f // fwshape[1] for f in feats_ind]))
    class_inst = np.zeros((len(a10_val.classes), len(a10_val.classes)))
    row_to_select = []

    # class_inst = []
    # class_inst_pred = []
    for col in feats_ind:
        cla = preds[col]
        if feats_w_[col, cla] <= 0:
            continue
        X, label = a10_val[col]
        class_inst[label, cla] = class_inst[label, cla] + 1
        row_to_select.append([col, 1 if feat[col] > 0 else -1])
    if len(row_to_select) < 1:
        continue
    fig, ax = plot_confusion_matrix(class_inst, a10_val.classes)
    fig.savefig(f"animal3-results-backup/results-{lambda_corr}/{i}/conf-mat.png")
    plt.close(fig)
    # rselect = [int(cl + col * 3) for col, cl in row_to_select]
    fmax = np.array([np.max(feats_w_[col, :]) for col, _ in row_to_select])
    inds = np.random.choice(len(row_to_select), min(45, len(row_to_select)), p=(fmax / np.sum(fmax)), replace=False)

    row = [row_to_select[i] for i in inds]

    for j, (col, sign) in enumerate(row):

        cla = preds[col]
        image, label = a10_val_wo_transform[col]

        image2 = np.uint8(image.numpy().transpose(1, 2, 0) * 255)
        image_org = image2.copy()
        wh = (224, 224)
        image3 = cv2.resize(image2, wh)
        explanation = explainer.explain_instance(image3, batch_predict, (i,),
                                                 top_labels=None,
                                                 hide_color=0,
                                                 num_samples=1000,
                                                 batch_size=256,
                                                 random_seed=RANDOM_SEED)
        # _, mask = explanation.get_image_and_mask(i, positive_only=False)
        exp = explanation.local_exp[i]
        mask = np.zeros(explanation.segments.shape, explanation.segments.dtype)

        if sign > 0:
            fs = [x[0] for x in exp
                  if x[1] > 0]
        else:
            fs = [x[0] for x in exp
                  if x[1] < 0]

        for f in fs:
            mask[explanation.segments == f] = 1
        h, w = image3.shape[:2]
        image2 = cv2.resize(image2, wh)
        image2 = np.uint8(image2 * np.dstack([mask for _ in range(3)]))

        image2 += np.uint8(np.ones_like(image2) * 127 * np.dstack([1 - mask for _ in range(3)]))
        nnnnname = "neg" if sign < 0 else "pos"
        cv2.imwrite(
            f"animal3-results-backup/results-{lambda_corr}/{i}/{j}-{col}-{nnnnname}-{a10_val.classes[cla]}-{a10_val.classes[label]}-({feat[col]}).png",
            np.uint8(cv2.cvtColor(np.vstack([image2, image3]), cv2.COLOR_RGB2BGR)))
        print(f"{lambda_corr}/{i}/{j}-{col}")
    end = time.time()

    print("Feature-time:", end - start)

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
