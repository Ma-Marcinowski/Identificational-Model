import tensorflow as tf
import numpy as np
import pandas as pd

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.layers import Input, Conv2D, Dense, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization, GaussianDropout
from tensorflow.keras.preprocessing.image import load_img, img_to_array

def load_image(img):
    return img_to_array(load_img(img, color_mode='grayscale')) / 255.
    
class DataSequence(tf.keras.utils.Sequence):

    def __init__(self, dataframe, batch_size):
        self.df = pd.read_csv(dataframe)
        self.batch_size = batch_size

        self.heads = self.df.columns.tolist()
        self.labels = self.df[self.heads[1:146]].values
        self.path_names = self.df['image_paths'].tolist()

    def __len__(self):
        return int(np.ceil(len(self.df) / float(self.batch_size)))

    def get_batch_labels(self, idx):
        batch_labels = self.labels[idx * self.batch_size: (idx + 1) * self.batch_size]
        return np.array(batch_labels)

    def get_batch_path_names(self, idx):
        batch_path_names = self.path_names[idx * self.batch_size: (idx + 1) * self.batch_size]
        return np.array([load_image(i) for i in batch_path_names])

    def __getitem__(self, idx):
        batch_x = self.get_batch_path_names(idx)
        batch_y = self.get_batch_labels(idx)
        return ({'input': batch_x}, {'output': batch_y})

BatchSize = 128

TrainSeq = DataSequence(dataframe='/path/TrainSheet.csv', batch_size = BatchSize)
ValidSeq = DataSequence(dataframe='/path/TestSheet.csv', batch_size = BatchSize)

input = Input(shape=(256, 256, 1), name='input')

def CoreNet(ix):

    x = Conv2D(64, (3, 3), strides=1, dilation_rate=2, padding='same', activation='relu', name='1stConv')(ix)
    x = BatchNormalization(axis=-1, scale=True, trainable=True)(x)
    x = Conv2D(64, (3, 3), strides=1, dilation_rate=2, padding='same', activation='relu', name='2ndConv')(x)
    x = BatchNormalization(axis=-1, scale=True, trainable=True)(x)    
    x = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same', name='1stPool')(x)

    x = Conv2D(128, (3, 3), strides=1, dilation_rate=2, padding='same', activation='relu', name='3rdConv')(x)
    x = BatchNormalization(axis=-1, scale=True, trainable=True)(x)
    x = Conv2D(128, (3, 3), strides=1, dilation_rate=2, padding='same', activation='relu', name='4thConv')(x)
    x = BatchNormalization(axis=-1, scale=True, trainable=True)(x)    
    x = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same', name='2ndPool')(x)

    x = Conv2D(256, (3, 3), strides=1, dilation_rate=2, padding='same', activation='relu', name='5thConv')(x)
    x = BatchNormalization(axis=-1, scale=True, trainable=True)(x)
    x = Conv2D(256, (3, 3), strides=1, dilation_rate=2, padding='same', activation='relu', name='6thConv')(x)
    x = BatchNormalization(axis=-1, scale=True, trainable=True)(x)
    x = Conv2D(256, (3, 3), strides=1, dilation_rate=2, padding='same', activation='relu', name='7thConv')(x)
    x = BatchNormalization(axis=-1, scale=True, trainable=True)(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same', name='3rdPool')(x)


    x = Conv2D(512, (3, 3), strides=1, dilation_rate=2, padding='same', activation='relu', name='9thConv')(x)
    x = BatchNormalization(axis=-1, scale=True, trainable=True)(x)
    x = Conv2D(512, (3, 3), strides=1, dilation_rate=2, padding='same', activation='relu', name='10thConv')(x)
    x = BatchNormalization(axis=-1, scale=True, trainable=True)(x)
    x = Conv2D(512, (3, 3), strides=1, dilation_rate=2, padding='same', activation='relu', name='11thConv')(x)
    x = BatchNormalization(axis=-1, scale=True, trainable=True)(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same', name='4thPool')(x)


    x = Conv2D(512, (3, 3), strides=1, dilation_rate=2, padding='same', activation='relu', name='12thConv')(x)
    x = BatchNormalization(axis=-1, scale=True, trainable=True)(x)
    x = Conv2D(512, (3, 3), strides=1, dilation_rate=2, padding='same', activation='relu', name='13thConv')(x)
    x = BatchNormalization(axis=-1, scale=True, trainable=True)(x)
    x = Conv2D(512, (3, 3), strides=1, dilation_rate=2, padding='same', activation='relu', name='14thConv')(x)
    x = BatchNormalization(axis=-1, scale=True, trainable=True)(x)
    ox = tf.keras.layers.GlobalAveragePooling2D(name='1stGAP')(x)

    return 

interm = CoreNet(ix=input)

x = GaussianDropout(rate=0.25)(interm) 
x = Dense(4096, activation='relu', name='1stFCL')(x) 
x = BatchNormalization(axis=-1, scale=True, trainable=True)(x)

x = GaussianDropout(rate=0.25)(x) 
x = Dense(4096, activation='relu', name='2ndFCL')(x) 
x = BatchNormalization(axis=-1, scale=True, trainable=True)(x)

x = GaussianDropout(rate=0.25)(x)            
output = Dense(145, activation='softmax', name='output')(x)

model = Model(inputs=[input], outputs=[output])

#model = load_model(filepath='/path/IM_v1.1.0.h5', compile=True)

Adam = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, amsgrad=False)

model.compile(optimizer=Adam, loss='categorical_crossentropy', metrics=['accuracy'])

csv_logger = tf.keras.callbacks.CSVLogger('/path/IM_v1.1.0_Training_Log.csv', separator=',', append=True)

checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath='/path/IM_v1.1.0-{epoch:02d}-{val_accuracy:.2f}.h5',
                                                monitor='val_accuracy',
                                                verbose=1,
                                                save_best_only=True,
                                                save_weights_only=False,
                                                mode='max',
                                                save_freq='epoch')

reduceLR = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                               factor=0.1,
                                               patience=5,
                                               verbose=1,
                                               mode='min',
                                               min_delta=0.0001,
                                               cooldown=0,
                                               min_lr=0)

history = model.fit(x=TrainSeq,
                    validation_data=ValidSeq,
                    callbacks=[csv_logger, checkpoint, reduceLR],
                    use_multiprocessing=False,
                    shuffle=True,
                    max_queue_size=10,
                    workers=1,
                    verbose=1,
                    validation_freq=1,
                    initial_epoch=0,
                    epochs=60)

#model.save(filepath='/path/IM_v1.1.0.h5', overwrite=True, include_optimizer=True, save_format='h5')
