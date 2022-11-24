import glob

import cv2
import torch
import torch.nn.functional as F
import tqdm
from captum._utils.models import SkLearnLinearRegression
from captum._utils.models.linear_model import SkLearnLinearModel
from PIL import Image

import os
import json
import numpy as np
from captum.attr._core.lime import get_exp_kernel_similarity_function, Lime, LimeBase
from captum.attr._utils.visualization import _normalize_image_attr
from matplotlib.colors import LinearSegmentedColormap

import torchvision
from torchvision import models
from torchvision import transforms

from captum.attr import IntegratedGradients, LRP
from captum.attr import GradientShap
from captum.attr import Occlusion
from captum.attr import NoiseTunnel
from captum.attr import visualization as viz
from shapely import geometry

from concavehull_scipy import alpha_shape

model = models.resnet18(pretrained=True)
model = model.eval()

labels_path = os.getenv("HOME") + '/.torch/models/imagenet_class_index.json'
with open(labels_path) as json_data:
    idx_to_labels = json.load(json_data)

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor()
])
def similarity_kernel(
    original_input,
    perturbed_input,
    perturbed_interpretable_input,
    **kwargs):
        # kernel_width will be provided to attribute as a kwarg
        kernel_width = kwargs["kernel_width"]
        l2_dist = torch.norm(original_input - perturbed_input)
        return torch.exp(- (l2_dist**2) / (kernel_width**2))


# Define sampling function
# This function samples in original input space
def perturb_func(original_input,
    **kwargs):
        return original_input + torch.randn_like(original_input)

# For this example, we are setting the interpretable input to
# match the model input, so the to_interp_rep_transform
# function simply returns the input. In most cases, the interpretable
# input will be different and may have a smaller feature set, so
# an appropriate transformation function should be provided.

def to_interp_transform(curr_sample, original_inp,
                                     **kwargs):
    return curr_sample

transform_normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

lime_attr = Lime(model)
images = glob.glob("imagenet-sample-images-master/*")
for imfile in tqdm.tqdm(images):
    img = Image.open(imfile)
    img = img.convert('RGB')
    transformed_img = transform(img)

    input = transform_normalize(transformed_img)
    input = input.unsqueeze(0)

    output = model(input)
    output = F.softmax(output, dim=1)
    prediction_score, pred_label_idx = torch.topk(output, 1)

    predicted_label = idx_to_labels[str(pred_label_idx[0].item())][1]
    # print('Predicted:', predicted_label, '(', prediction_score.squeeze().item(), ')')
    #
    # print('Predicted:', predicted_label, '(', prediction_score.squeeze().item(), ')')



    attrs = lime_attr.attribute(input,
        target=pred_label_idx).squeeze(0)
    print(attrs.min().item(), attrs.max().item())
    _ = viz.visualize_image_attr_multiple(np.transpose(attrs.cpu().detach().numpy(), (1, 2, 0)),
                                          np.transpose(transformed_img.squeeze().cpu().detach().numpy(), (1, 2, 0)),
                                          ["original_image", "heat_map"],
                                          ["all", "all"],
                                          show_colorbar=True)
    continue
    integrated_gradients = IntegratedGradients(model)
    # attributions_ig = integrated_gradients.attribute(input, target=pred_label_idx, n_steps=200)
    default_cmap = LinearSegmentedColormap.from_list('custom blue',
                                                     [(0, '#cccccc'),
                                                      (0.25, '#000000'),
                                                      (1, '#000000')], N=256)
    # print(model)
    noise_tunnel = NoiseTunnel(integrated_gradients)


    def get_circular_kernel(diameter):
        mid = (diameter - 1) / 2
        distances = np.indices((diameter, diameter)) - np.array([mid, mid])[:, None, None]
        kernel = ((np.linalg.norm(distances, axis=0) - mid) <= 0).astype(int)

        return kernel


    attributions_ig_nt = integrated_gradients.attribute(input, target=pred_label_idx)
    # _ = viz.visualize_image_attr_multiple(np.transpose(attributions_ig_nt.squeeze().cpu().detach().numpy(), (1, 2, 0)),
    #                                       np.transpose(transformed_img.squeeze().cpu().detach().numpy(), (1, 2, 0)),
    #                                       ["original_image", "heat_map"],
    #                                       ["all", "positive"],
    #                                       show_colorbar=True)
    original_image = np.transpose(transformed_img.squeeze().cpu().detach().numpy(), (1, 2, 0))
    grads = np.transpose(attributions_ig_nt.squeeze().cpu().detach().numpy(), (1, 2, 0))
    norm_img = _normalize_image_attr(grads, "positive", 2)
    ngrads = norm_img
    # ngrads[ngrads < 0] = 0
    ngrads = (ngrads) / (np.max(ngrads))
    method = "kmeans concave"
    h, w, c = original_image.shape

    # if method == "brute":
    #     nz = np.nonzero(grads > (np.max(grads) / 20))
    #     transpose = np.array(nz, dtype=np.float32).transpose()[:, 0:2]
    #     hull = cv2.convexHull(transpose)
    #     mask = np.zeros_like(grads)[..., 0:1]
    #     mask = np.uint8(mask)
    #     int_ = np.int32(hull)
    #     cv2.fillPoly(mask, pts=[int_], color=(1, 1, 1))
    # elif method.startswith("kmeans"):
    #     ngrads = ngrads > 0.2
    #     # premask = ngrads * 255
    #     # vectorized = np.float32(premask.reshape(-1, 1))
    #     # criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    #     #
    #     # K = 1
    #     # attempts = 10
    #     # ret, label, center = cv2.kmeans(vectorized, K, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)
    #     # res = center[label.flatten()]
    #     # if len(res.shape) < 3:
    #     #     res = res.reshape((h, w, -1))
    #     # if res.shape[2] != c:
    #     #     res = np.dstack((res,) * c)
    #     # res2 = res.reshape((original_image.shape))
    #     # mask = (res2 - np.min(res2)) / (np.max(res2) - np.min(res2))
    #     if "concave" in method:
    #         nz = np.nonzero(ngrads)
    #         transpose = np.array(nz, dtype=np.float32).transpose()[:, 0:2]
    #         point_collection = geometry.MultiPoint(list(transpose))
    #
    #         concave_hull, edge_points = alpha_shape(point_collection, alpha=.1)
    #         # edges = np.array([e[0] for e in edge_points])
    #         # mask = np.zeros_like(grads)[..., 0:1]
    #         # mask = np.uint8(mask)
    #         # int_ = np.int32(edges)
    #         # cv2.fillPoly(mask, pts=[int_], color=(1, 1, 1))
    #         mask = rasterio.features.rasterize([concave_hull], out_shape=(h, w)).transpose()
    # kernel = np.uint8(get_circular_kernel(15))
    # kernel2 = np.uint8(get_circular_kernel(6))
    # opening = cv2.morphologyEx(np.uint8(ngrads * 255), cv2.MORPH_CLOSE, kernel)
    # opening = cv2.dilate(opening, kernel)
    # opening = cv2.morphologyEx(opening, cv2.MORPH_OPEN, kernel)
    # opening = cv2.erode(opening, kernel)
    # opening = cv2.morphologyEx(opening, cv2.MORPH_OPEN, kernel)
    # opening = cv2.dilate(opening, kernel)
    # opening = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    # opening = cv2.erode(opening, kernel2)
    # nz = np.nonzero(opening)
    # transpose = np.array(nz, dtype=np.float32).transpose()[:, 0:2]
    # point_collection = geometry.MultiPoint(list(transpose))
    #
    # concave_hull, edge_points = alpha_shape(point_collection, alpha=.7)
    # # edges = np.array([e[0] for e in edge_points])
    # # mask = np.zeros_like(grads)[..., 0:1]
    # # mask = np.uint8(mask)
    # # int_ = np.int32(edges)
    # # cv2.fillPoly(mask, pts=[int_], color=(1, 1, 1))
    # mask = rasterio.features.rasterize([concave_hull], out_shape=(h, w)).transpose()
    # mask = opening / 255
    kernel = np.uint8(get_circular_kernel(5))
    mask = norm_img
    # mask[mask > 0.25] = 1
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
        [cv2.resize(orig_img, [orig_img.shape[1] //2, orig_img.shape[0] // 2], interpolation=cv2.INTER_AREA),
         cv2.resize(heatmap_img, [heatmap_img.shape[1] // 2, heatmap_img.shape[0] // 2], interpolation=cv2.INTER_AREA)])

    im_v = np.vstack([im_h, highlight])

    # opening = cv2.dilate(opening, kernel)
    imname = os.path.basename(imfile).rsplit(".", 2)[0]
    cv2.imwrite("imagenet-results/" + imname + "-" + predicted_label + "-" + str(
        round(prediction_score.squeeze().item(), 3)) + ".png", im_v)
    cv2.imwrite("imagenet-results/" + imname + "-" + predicted_label + "-" + str(
        round(prediction_score.squeeze().item(), 3)) + "-imgrad.png", np.uint8(norm_img * 255))
    cv2.imwrite("imagenet-results/" + imname + "-" + predicted_label + "-" + str(
        round(prediction_score.squeeze().item(), 3)) + "-ngrad.png", np.uint8(ngrads * 255))
# _ = visualize_image_attr(grads, original_image, "blended_heat_map", sign="all", show_colorbar=True,
#                          title="Overlayed Gradient Magnitudes: " + str(out))
# _ = visualize_image_attr(None, np.uint8((original_image - 0.5) * mask + 0.5), method="original_image",
#                          title="Original Image")
