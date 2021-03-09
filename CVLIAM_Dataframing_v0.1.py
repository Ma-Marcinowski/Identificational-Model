import os
import csv
import random
import sklearn
import numpy as np
import pandas as pd

from tqdm import tqdm

def Dataframe(dr_img_path, df_img_path, lab_df_path, tran_df_path, test_df_path, test_fraction):

    lab_df = pd.read_csv(lab_df_path)
    aut_id = lab_df['AuthorIDs'].tolist()
    
    imgs = os.listdir(dr_img_path)

    with open(tran_df_path, 'a+') as f:

        writer = csv.writer(f)

        for j in tqdm(imgs, leave=False):
            
            for i in aut_id:

                if j[:8] == i:
                    
                    dat_row = lab_df[lab_df['AuthorIDs'] == i].values
                    
                    img_name = df_img_path + j
                                        
                    dat_row[0, 0] = img_name
                    
                    writer.writerow(dat_row[0])

                else:

                    continue

        print('Done labelling images.')

    tr_df = pd.read_csv(tran_df_path, header=None)

    tr_df = sklearn.utils.shuffle(tr_df)

    ts_df = tr_df.sample(frac=test_fraction)
    tr_df = pd.concat([tr_df, ts_df]).drop_duplicates(keep=False)    

    img_header = 'image_paths'
    dat_headers = lab_df.columns.tolist()
    dat_headers[0] = img_header

    tr_df.to_csv(tran_df_path, header=dat_headers, index=False)
    ts_df.to_csv(test_df_path, header=dat_headers, index=False)

    print('Done dataframing.')

ModelDataframe = Dataframe(dr_img_path='/directory/of/preprocessed/images',
                           df_img_path='/directory/of/dataframed/images/to/load/into/a/model',
                           lab_df_path='/dataframe/of/labels/Labels_Df_v0.1.csv',
                           tran_df_path='/path/to/the/training/dataframe/TrainingSheet.csv',
                           test_df_path='/path/to/the/testing/dataframe/TestSheet.csv',
                           test_fraction=0.2) #any desired fraction of training instances to sample into the training subset

print('Dataframes done.')
