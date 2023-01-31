import glob
import os.path
import random
import shutil

datas = glob.glob("data/animals-3/raw-img/*")
for data in datas:
    all_data = list(glob.glob(os.path.join(data, "*.png")))
    length = len(all_data)
    train_length = int(length * 0.9)
    random.shuffle(all_data)
    train_out = data.replace("raw-img", "train")
    test_out = data.replace("raw-img", "test")
    if os.path.exists(train_out):
        os.removedirs(train_out)
    os.makedirs(train_out)
    if os.path.exists(test_out):
        os.removedirs(test_out)
    os.makedirs(test_out)
    train_data = all_data[:train_length]
    test_data = all_data[train_length:]
    print(len(all_data) == len(train_data) + len(test_data))
    for traind in train_data:
        train_dir = traind.replace("raw-img", "train")
        shutil.copy(traind, train_dir)
    for testd in test_data:
        test_dir = testd.replace("raw-img", "test")
        shutil.copy(testd, test_dir)
