import os.path

import cv2
import glob

import numpy as np

imglist = glob.glob("animal3-results-backup/results-0.001/collages/*.png", recursive=True)
for img in imglist:
    imgn = cv2.imread(img)
    imgn = cv2.resize(imgn, (imgn.shape[1]//4,imgn.shape[0]//4 ))
    cv2.imwrite(img,imgn)
exit()
img_fold_list = list(set(os.path.dirname(imgfile) for imgfile in imglist))

collage_dir = os.path.join("animal3-results/results-0.001", "collages")
if not os.path.exists(collage_dir):
    os.makedirs(collage_dir)
img_fold_list = sorted([img_fold for img_fold in img_fold_list if not "collages" in img_fold ],
                       key=lambda x: int(os.path.basename(x)))
cnt = 0
for i, img_fold in enumerate(img_fold_list):
    img_fold_2 = os.path.join(img_fold, "**/*.png")
    imglist2 = [img for img in imglist if img.startswith(img_fold)]
    imglist2 = [img for img in imglist2 if os.path.basename(img)[0].isdigit()]
    if len(imglist2) == 0:
        continue
    imglist2 = sorted(imglist2)

    print(cnt, os.path.basename(img_fold))
    cnt+=1
    continue

    img_coll_big = np.zeros([224 * 3 * 2, 224 * 3 * 4, 3], dtype=np.uint8)
    for i in range(5):
        imgs = imglist2[i * 9:(i + 1) * 9]

        imgs = [cv2.imread(img)[:224, :224, :] for img in imgs]

        img_coll = np.zeros([224 * 3, 224 * 3, 3], dtype=np.uint8)

        for j in range(9):
            r = j // 3
            c = j % 3
            img_coll[r * 224:(r + 1) * 224, c * 224:(c + 1) * 224, :] = imgs[j]
        if i == 0:
            img_coll_big[:224 * 3 * 2, 224 * 3:224 * 3 * 3] = cv2.resize(img_coll, (224 * 3 * 2, 224 * 3 * 2))
        else:
            k = i - 1
            r = k // 2
            c = k % 2
            if c == 1:
                c = 3

            img_coll_big[(r) * 224 * 3: (r + 1) * 224 * 3, c * 224 * 3:(c + 1) * 224 * 3] = img_coll
    cv2.imwrite(os.path.join(collage_dir, f"coll-{os.path.basename(img_fold)}.png"), img_coll_big)
