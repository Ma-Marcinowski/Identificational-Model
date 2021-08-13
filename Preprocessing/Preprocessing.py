import os
import cv2
import numpy as np
from tqdm import tqdm

def Preprocessing(in_path, out_path, filters_dir):

    k_names = os.listdir(filters_dir)
    kernels = [filters_dir + n for n in k_names]

    img_names = os.listdir(in_path)
    img_paths = [in_path + n for n in img_names]
    listed = zip(img_paths, img_names)

    for j, i in tqdm(listed, total=len(img_paths), desc='cvl-loop'):

        img = cv2.imread(j, 0)

        inv = np.bitwise_not(img)

        y=930
        x=270
        h=2048
        w=2048

        cropped = inv[y:y+h, x:x+w]

        resized = cv2.resize(cropped,(1024,1024))

        horizontal = np.split(resized, 4, axis=1)

        for idx, h in enumerate(horizontal, start=1):

            vertical = np.split(h, 4, axis=0)

            for ind, v in enumerate(vertical, start=1):

                thv, denv = cv2.threshold(v, 55, 255, cv2.THRESH_TOZERO)

                filtered = []

                for kern in kernels:

                    kernel = cv2.imread(kern, 0)

                    ksum = np.sum(np.multiply(denv, kernel))

                    filtered.append(ksum)

                if 0 not in filtered:

                    cv2.imwrite(out_path + 'cvl-' + i[:-4] + '-' + str(idx) + str(ind) + '.png', v)

                else:

                    continue

    print('Preprocessing done: 100%')

Preprocessing = Preprocessing(in_path='/path/to/CVL/raw/images/',
                              out_path='path/to/CVL/preprocessed/images/',
                              filters_dir='/path/to/filters/folder/' )
