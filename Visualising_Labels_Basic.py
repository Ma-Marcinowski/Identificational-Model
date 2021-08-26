import tensorflow as tf
import numpy as np
import cv2

from tqdm.notebook import tqdm

from tensorflow.keras.models import Model, load_model

def InitializeImage():

    random_tensor2 = tf.keras.backend.random_uniform([256, 256], minval=0.0, maxval=1.0, dtype='float32')
    random_tensor3 = tf.keras.backend.expand_dims(random_tensor2, axis=-1)
    random_tensor4 = tf.keras.backend.expand_dims(random_tensor3, axis=0)

    return random_tensor4

def GradientAscent(image, label_index, sub_model, ascent_steps, ascent_rate):

    for step in tqdm(range(ascent_steps), desc='Label ' + str(label_index + 1) + ' gradient ascent:', leave=False):

        with tf.GradientTape() as tape:
          
            tape.watch(image)

            sub_model_output = sub_model(image)

            filter_activation = sub_model_output[:, filter_index]
            
        label_gradient = tape.gradient(label_activation, image)

        grad_mean = tf.math.reduce_mean(filter_gradient)
        grad_diff = tf.math.subtract(filter_gradient, grad_mean)
        grad_stdv = tf.math.add(tf.math.reduce_std(filter_gradient), 1e-9)
        normalized_gradient = tf.math.divide(grad_diff, grad_stdv)
        
        image = tf.math.add(image, (tf.math.multiply(normalized_gradient, ascent_rate)))

    return image

def Visualising_Labels(model_load_path, img_out_path, 
                       input_layer_name, output_layer_name, num_of_labels, 
                       ascent_steps, ascent_rate):

    model = load_model(model_load_path)

    sub_model = Model(inputs=model.get_layer(name=input_layer_name).input, outputs=model.get_layer(name=output_layer_name).output)

    nulls = len(str(num_of_labels))

    for label_index in tqdm(range(num_of_labels), desc=output_layer_name + ' labels visualisation:', leave=True):

        initial_image = InitializeImage()

        label_tensor = GradientAscent(image=initial_image, 
                                      label_index=label_index,
                                      sub_model=sub_model,
                                      ascent_steps=ascent_steps,
                                      ascent_rate=ascent_rate)

        img_tensor = label_tensor.numpy()
        img_matrix = img_tensor[0, :, :, 0]
  
        normalized = cv2.normalize(img_matrix, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F).astype(np.uint8)

        ind = str(label_index + 1).zfill(nulls)
        
        blurred = cv2.GaussianBlur(normalized,(5, 5),0)
        retv, ots = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        cv2.imwrite(img_out_path + output_layer_name + '_labels_' + ind + '.png', normalized)
        cv2.imwrite(img_out_path + output_layer_name + '_labels_bin_' + ind + '.png', ots)

visualising_author_labels = Visualising_Labels(model_load_path='/saved/model/directory/model.h5',
                                               img_out_path='/visualised/labels/directory/',
                                               input_layer_name='model_input_layer',
                                               output_layer_name='labels_layer_to_visualise',
                                               num_of_labels=27, #Number of the named layer labels.
                                               ascent_steps=1000,
                                               ascent_rate=0.1)

visualising_feature_labels = Visualising_Labels(model_load_path='/saved/model/directory/model.h5',
                                                img_out_path='/visualised/labels/directory/',
                                                input_layer_name='model_input_layer',
                                                output_layer_name='labels_layer_to_visualise',
                                                num_of_labels=84, #Number of the named layer labels.
                                                ascent_steps=1000,
                                                ascent_rate=0.1)
