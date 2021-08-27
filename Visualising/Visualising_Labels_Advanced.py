import tensorflow_addons as tfa
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

def GradientAscent(image, label_index, sub_model, 
                   ascent_steps, ascent_rate, 
                   sigma, sigma_step_rate):
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=ascent_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, amsgrad=False)

    for step in tqdm(range(ascent_steps), desc='Label ' + str(label_index + 1) + ' features ascent:', leave=False):
        
        with tf.GradientTape() as tape:
            
            image = tf.Variable(image)
          
            tape.watch(image)

            sub_model_output = sub_model(image)

            label_activation = sub_model_output[:, label_index]

        label_gradient = tape.gradient(label_activation, image)

        grad_mean = tf.math.reduce_mean(filter_gradient)
        grad_diff = tf.math.subtract(filter_gradient, grad_mean)
        grad_stdv = tf.math.add(tf.math.reduce_std(filter_gradient), 1e-9)
        normalized_gradient = tf.math.divide(grad_diff, grad_stdv)
        
        if sigma > 0.5:

            kernel_size = (np.round((sigma * 3), 0)).astype(np.int32)

        else:

            kernel_size = 2      
            
        blurred_gradient = tfa.image.gaussian_filter2d(image=normalized_gradient, 
                                                       filter_shape=[kernel_size, kernel_size], 
                                                       sigma=sigma, 
                                                       padding='CONSTANT', 
                                                       constant_values=0.0)
        
        sigma = sigma - sigma_step_rate

        optimizer.apply_gradients(zip([blurred_gradient], [image]))

        image = tf.math.l2_normalize(updated_image, epsilon=1e-12)
       
    return image, sigma

def OctaveRescaling(image, label_index, sub_model, 
                    ascent_steps, ascent_rate, 
                    octave_rescales, octave_scales, octave_probs,
                    sigma, sigma_step_rate):
  
    for rescale in tqdm(range(octave_rescales), desc='Label ' + str(label_index + 1) + ' octave rescaling:', leave=False):

        shape = tf.cast(tf.shape(image[0, :, :, 0]), tf.float32)

        octave_scaling_factor = np.random.choice(octave_scales, p=octave_probs)

        reshape = tf.cast(tf.math.multiply(shape, octave_scaling_factor), tf.int32)

        resized = tf.image.resize(image, reshape)

        if resized.shape[1] % 2 != 0:

          padded = tf.pad(resized, paddings=[[0,0], [0,1], [0,1], [0, 0]], mode='CONSTANT', constant_values=0)

        else:

          padded = resized

        off_limit = (tf.shape(padded)[1].numpy() - 256).astype(np.int32)
        off_randw = np.random.randint(0, high=off_limit)
        off_randh = np.random.randint(0, high=off_limit)
 
        image = tf.image.crop_to_bounding_box(padded, off_randw, off_randh, 256, 256)

        image, sigma = GradientAscent(image=image, 
                                      label_index=label_index,
                                      sub_model=sub_model,
                                      ascent_steps=ascent_steps,
                                      ascent_rate=ascent_rate,
                                      sigma=sigma,
                                      sigma_step_rate=sigma_step_rate)


    return image

def Visualising_Labels(model_load_path, img_out_path,
                       input_layer_name, output_layer_name, num_of_labels, 
                       ascent_steps, ascent_rate,
                       octave_rescales, octave_scales, octave_probs,
                       sigma_max, sigma_min):

    model = load_model(model_load_path)

    sub_model = Model(inputs=model.get_layer(name=input_layer_name).input, outputs=model.get_layer(name=output_layer_name).output)

    nulls = len(str(num_of_labels))

    if octave_rescales > 0:

        sigma_step_rate = (sigma_max - sigma_min) / (ascent_steps + (ascent_steps * octave_rescales))

    else:

        sigma_step_rate = (sigma_max - sigma_min) / ascent_steps

    for label_index in tqdm(range(num_of_labels), desc=output_layer_name + ' labels visualisation:', leave=True):

        initial_image = InitializeImage()

        label_tensor, reduced_sigma = GradientAscent(image=initial_image, 
                                                     label_index=label_index,
                                                     sub_model=sub_model,
                                                     ascent_steps=ascent_steps,
                                                     ascent_rate=ascent_rate,
                                                     sigma=sigma_max,
                                                     sigma_step_rate=sigma_step_rate)

        rescaled_tensor = OctaveRescaling(image=label_tensor, 
                                          label_index=label_index,
                                          sub_model=sub_model,
                                          ascent_steps=ascent_steps,
                                          ascent_rate=ascent_rate,
                                          octave_rescales=octave_rescales, 
                                          octave_scales=octave_scales,
                                          octave_probs=octave_probs,
                                          sigma=reduced_sigma,
                                          sigma_step_rate=sigma_step_rate)

        img_tensor = rescaled_tensor.numpy()
        img_matrix = img_tensor[0, :, :, 0]
  
        normalized = cv2.normalize(img_matrix, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F).astype(np.uint8)

        blurred = cv2.GaussianBlur(normalized,(5, 5),0)
        retv, ots = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        cv2.imwrite(img_out_path + output_layer_name + '_labels_' + ind + '.png', normalized)
        cv2.imwrite(img_out_path + output_layer_name + '_labels_bin_' + ind + '.png', ots)

visualising_author_labels = Visualising_Labels(model_load_path='/saved/model/directory/model.h5',
                                               img_out_path='/visualised/labels/directory/',
                                               input_layer_name='model_input_layer',
                                               output_layer_name='labels_layer_to_visualise',
                                               num_of_labels=27, #Number of the named layer labels.
                                               ascent_steps=5000,
                                               ascent_rate=0.00001,
                                               octave_rescales=0, 
                                               octave_scales=[1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9],
                                               octave_probs=[0.20, 0.18, 0.16, 0.14, 0.12, 0.08, 0.06, 0.04, 0.02],
                                               sigma_max=2.50,
                                               sigma_min=0.01)


visualising_feature_labels = Visualising_Labels(model_load_path='/saved/model/directory/model.h5',
                                                img_out_path='/visualised/labels/directory/',
                                                input_layer_name='model_input_layer',
                                                output_layer_name='labels_layer_to_visualise',
                                                num_of_labels=84, #Number of the named layer labels.
                                                ascent_steps=5000,
                                                ascent_rate=0.00001,
                                                octave_rescales=0, 
                                                octave_scales=[1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9],
                                                octave_probs=[0.20, 0.18, 0.16, 0.14, 0.12, 0.08, 0.06, 0.04, 0.02],
                                                sigma_max=2.50,
                                                sigma_min=0.01)
