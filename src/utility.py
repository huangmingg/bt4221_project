import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from typing import Tuple


def find_largest_resolution(image_dir: os.path) -> Tuple[int, int]:
    cat_folders = os.listdir(image_dir)
    max_height = 0
    max_width = 0
    for cat in tqdm(cat_folders):
        for f in os.listdir(os.path.join(image_dir, cat)):
            filepath = os.path.join(image_dir, cat, f)
            if not os.path.isfile(filepath):
                continue
            img = plt.imread(filepath)
            if len(img.shape) == 2:
                print("Image is greyscale (no color), values will be in R")
            elif len(img.shape) == 3 and img.shape[2] == 4:
                print("RGBa detected, ignoring a")            
            else:
                w, h, c = img.shape                
                max_height = max_height if h <= max_height else h
                max_width = max_width if w <= max_width else w
    return (max_width, max_height)


def unzip(zipfile: os.path, destdir: os.path) -> bool:
    try:
        with zipfile.ZipFile(zipfile, 'r') as zr:
            zr.extractall(destdir)
        return True
    except Exception as e:
        print(e)
        return False
