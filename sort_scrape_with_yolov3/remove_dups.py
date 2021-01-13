import os
import cv2
import shutil
import numpy as np

def drop_duplicates():
    dup = []
    files = 'data/sorted'
    dupfolder = 'data/duplicates'
    if not os.path.exists(dupfolder):
        os.makedirs(dupfolder)
    totalfiles = os.listdir(files)
    for file in os.listdir(files):
        print(file)
        if file in dup:
            continue
        im1 = cv2.imread(f'{files}/{file}')
        totalfiles.remove(file)
        for file2 in totalfiles:
            im2 = cv2.imread(f'{files}/{file2}')
            if im1.shape == im2.shape:
                if np.allclose(im1, im2):
                    dup.append(file2)
                    shutil.move(f'{files}/{file2}', f"{dupfolder}/{file2}")
