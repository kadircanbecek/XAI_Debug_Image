import math
import os.path

import cv2
import numpy as np
import glob

from PIL import Image

# imgfiles = glob.glob("/home/sight/Documents/thesis/XAI_Debug_Image/Cthulhu Fluxx-81-1-0/FluxxCthulhu/*.png")
names = list([str(i) + ".png" for i in range(1, 93)]) + ["back.jpg"]
height = 83
width = 54
A4h = 297
A4w = 210
r = 0
c = 0
rotate = False
r1 = int(A4h / height)
c1 = int(A4w / width)
c2 = int(A4w / height)
r2 = int(A4h / width)

r = r1
c = c1
h = height
w = width
# if r1 * c1 > r2 * c2:
#     r = r1
#     c = c1
#     h = height
#     w = width
# else:
#     r = r2
#     c = c2
#     w = height
#     h = width
#     rotate = True

A4_h_space = 2  # A4h_remnant / (r + 1)
A4_w_space = 2  # A4w_remnant / (c + 1)

A4h_remnant = A4h - r * h - A4_h_space * (r - 1)
A4w_remnant = A4w - c * w - A4_w_space * (c - 1)
scale = 16

imgfiles = ["/home/sight/Documents/thesis/XAI_Debug_Image/Cthulhu Fluxx-81-1-0/FluxxCthulhu/" + name for name in names]
back = imgfiles[-1]
imgfiles = imgfiles[:-1]
for i in imgfiles:
    print(os.path.exists(i))


def collage_image(imlist, idx):
    new_image = np.ones((int(A4h * scale), int(A4w * scale), 3), dtype=np.uint8) * 255
    for i, im in enumerate(imlist):
        c_card = i // r
        r_card = i % r
        print([r_card, c_card])
        start = [int(A4w_remnant / 2 * scale + (c_card) * (w + A4_w_space) * scale),
                 int(A4h_remnant / 2 * scale + (r_card) * (h + A4_h_space) * scale)]
        end = [int(A4w_remnant / 2 * scale + (c_card + 1) * (w + A4_w_space) * scale - A4_w_space * scale),
               int(A4h_remnant / 2 * scale + (r_card + 1) * (h + A4_h_space) * scale) - A4_h_space * scale]

        card = cv2.imread(im)
        if rotate:
            card = cv2.rotate(card, cv2.ROTATE_90_CLOCKWISE)
        card = cv2.resize(card, (int(w * scale * .97), int(h * scale * .97)))
        print(start)
        print(end)
        y_mid = (start[1] + end[1]) // 2
        x_mid = (start[0] + end[0]) // 2
        print(x_mid)
        print(y_mid)
        print(card.shape)
        new_image[y_mid - math.floor(card.shape[0] / 2): y_mid + math.ceil(card.shape[0] / 2),
        x_mid - math.floor(card.shape[1] / 2):x_mid + math.ceil(card.shape[1] / 2)] = card
        # cv2.rectangle(new_image, start, end, (0, 0, 0), thickness=scale//3)
        for s in [[start[0] - int(A4_w_space / 2 * scale), start[1] - int(A4_w_space / 2 * scale)],
                  [end[0] + int(A4_w_space / 2 * scale), end[1] + int(A4_w_space / 2 * scale)],
                  [start[0] - int(A4_w_space / 2 * scale), end[1] + int(A4_w_space / 2 * scale)],
                  [end[0] + int(A4_w_space / 2 * scale), start[1] - int(A4_w_space / 2 * scale)]]:
            cv2.line(new_image, [s[0] - scale, s[1]], [s[0] + scale, s[1]], (0, 0, 0), thickness=scale // 6)
            cv2.line(new_image, [s[0], s[1] - scale], [s[0], s[1] + scale], (0, 0, 0), thickness=scale // 6)



    print(new_image.shape)
    new_image = cv2.resize(new_image, (int(new_image.shape[1]), int(new_image.shape[0])), cv2.INTER_AREA)
    fname = "/home/sight/Documents/thesis/XAI_Debug_Image/Cthulhu Fluxx-81-1-0/" + str(idx) + ".png"
    cv2.imwrite(fname, new_image)
    return cv2.cvtColor(new_image,cv2.COLOR_RGB2BGR)


first_im = None
imlist = []
for iss in range(1, len(imgfiles[:-1]) // 9 + 2):

    front = Image.fromarray(collage_image(imgfiles[(iss - 1) * 9:(iss) * 9], str(iss) + "-front"))
    if not first_im:
        first_im = front
    else:
        imlist.append(front)
    backu = Image.fromarray(collage_image([back for i in range(9)], str(iss) + "-back"))

    imlist.append(backu)
first_im.save(r'/home/sight/Documents/thesis/XAI_Debug_Image/Cthulhu Fluxx-81-1-0/fluxx.pdf', save_all=True, append_images=imlist)
