import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

def Preprocessing(in_path, out_path, filters_dir):

    k_names = os.listdir(filters_dir)
    kernels = [filters_dir + n for n in k_names]

    InitDf = pd.read_csv(in_path)

    path = InitDf['RawPath'].tolist()
    ids = InitDf['RawID'].tolist()
    form = InitDf['RawForm'].tolist()
    listed = zip(path, ids, form)

    for j, i, f in tqdm(listed, total=len(ids), desc='iam-loop'):

        img = cv2.imread(j, 0)

        inv = np.bitwise_not(img)

        y=730
        x=230
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

                    cv2.imwrite(out_path + str(i) + '-' + str(f) + '-' + str(idx) + str(ind) + '.png', v)

                else:

                    continue

    print('IAM preprocessing done: 100%')

Preprocessing = Preprocessing(in_path='/path/to/dataframe/of/raw/IAM/images/IamRawDf.csv',
                              out_path='path/to/CVL/preprocessed/images/',
                              filters_dir='/path/to/filters/folder/')
