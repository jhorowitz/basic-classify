from random import shuffle
import os
import re
import shutil

TRAIN = "train"
VALIDATE = "validate"
TEST = "test"

sets = [
    TRAIN,
    VALIDATE,
    TEST
]

CLASS_LOCATION = os.getenv("CLASS_LOCATION")
if CLASS_LOCATION is None:
    print("Must specify a CLASS_LOCATION")

CLASSES = os.getenv("CLASSES")
if CLASS_LOCATION is None:
    print("Must specify a CLASSES")

classes = CLASSES.split(",")
TRAINING_DATA_DIR_NAME = "training_data"
os.mkdir(TRAINING_DATA_DIR_NAME)


img_paths = {}


def list_pictures(directory, ext='jpg|jpeg|bmp|png|ppm'):
    return [os.path.join(root, f)
            for root, _, files in os.walk(directory) for f in files
            if re.match(r'([\w]+\.(?:' + ext + '))', f)]


def split_to_3(lst, train_ratio=.7, val_ratio=.15, test_ratio=.15):
    train_quantity = int(len(lst) * train_ratio)
    val_quantity = int(len(lst) * val_ratio)
    test_quantity = int(len(lst) * test_ratio)
    train = lst[0: train_quantity]
    val = lst[train_quantity: train_quantity + val_quantity]
    test = lst[train_quantity + val_quantity: train_quantity + val_quantity + test_quantity]

    assert len(train) + len(val) + len(test) == len(lst)
    return {TRAIN: train, VALIDATE: val, TEST: test}


for c in classes:
    path = os.path.join(CLASS_LOCATION, c)
    pic_paths = list_pictures(path)
    shuffle(pic_paths)
    img_paths[c] = split_to_3(pic_paths)

prev = 0
for s in sets:
    set_path = os.path.join(TRAINING_DATA_DIR_NAME, s)
    os.mkdir(set_path)
    for c in classes:
        set_class_path = os.path.join(set_path, c)
        os.mkdir(set_class_path)
        for img_path in img_paths[c][s]:
            shutil.copy2(img_path, set_class_path)
