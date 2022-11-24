import glob
import os

import cv2
import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms import transforms
from PIL import Image
from torchvision.models import resnet18


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
        image = Image.open(img_name)

        if self.transform:
            image = self.transform(image)

        sample = {'image': image, 'class': self.class_to_idx[os.path.dirname(img_name).split("/")[-1]]}
        return sample


a10 = Animal10("./data/animals-10",
               transform=transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()]))
sample_0 = a10[0]
imaj = sample_0["image"].numpy()
# print(imaj.shape)
# cv2.imshow(a10.idx_to_class[sample_0["class"]], imaj.transpose(1, 2, 0))
# cv2.waitKey()
# print()
dataloader = DataLoader(a10, batch_size=4,
                        shuffle=True, num_workers=4)
model_fe = FeatureExtract()
model_cl = Classifier(len(a10.classes))
inp = torch.rand([1, 3, 224, 224])
outp = model_fe(inp)
out_classes = model_cl(outp)
print(out_classes.shape)
