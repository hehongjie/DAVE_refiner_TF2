import os
import ssl # these two lines solved issues loading pretrained model
ssl._create_default_https_context = ssl._create_unverified_context
import numpy as np
import cv2
# from scipy.misc import imresize
# np.random.seed(T_G_SEED)

# TensorFlow Includes
import tensorflow as tf
# tf.set_random_seed(T_G_SEED)

# Keras Imports & Defines 
from tensorflow import keras
import tensorflow.keras.backend as K
import datetime
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
# Local Imports
from models import Unet_2levels, Dunet_2levels, DVAE_refiner, DVAE
from utils import log_gaussian, get_mix_coef, get_my_metrics, topo_metric

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
def DataGenerator(file_path, batch_size):
    """
    generate image and mask at the same time
    use the same seed for image_datagen and mask_datagen
    to ensure the transformation for image and mask is the same
    """
    aug_dict = dict(rotation_range=0.2,
                        width_shift_range=0.05,
                        height_shift_range=0.05,
                        shear_range=0.05,
                        zoom_range=0.05,
                        horizontal_flip=True,
                        fill_mode='nearest')
    aug_dict = dict(horizontal_flip=True,
                        fill_mode='nearest')

    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    image_generator = image_datagen.flow_from_directory(
        file_path,
        classes=["images"],
        color_mode = "rgb",
        target_size = (512, 512),
        class_mode = None,
        batch_size = batch_size, seed=1)

    mask_generator = mask_datagen.flow_from_directory(
        file_path,
        classes=["labels"],
        color_mode = "grayscale",
        target_size = (512, 512),
        class_mode = None,
        batch_size = batch_size, seed=1)

    train_generator = zip(image_generator, mask_generator)
    for (img,mask) in train_generator:
        img = img / 255.

        mask = mask/255.
        

        #img=tf.image.rgb_to_grayscale(img, name=None)
        yield (img,mask)
def jaccard_index_b(y_true, y_pred):

    safety = 0.001

    y_true_f = K.cast(K.greater(K.flatten(y_true),0.5),'float32')
    y_pred_f = K.cast(K.greater(K.flatten(y_pred),0.5),'float32')

    top = K.sum(K.minimum(y_true_f, y_pred_f))
    bottom = K.sum(K.maximum(y_true_f, y_pred_f))

    return top / (bottom + safety)


# A binary jaccard (non-differentiable)
def jaccard_loss_b(y_true, y_pred):

    return 1 - jaccard_index_b(y_true, y_pred)

# An example loss based on multiple metrics
def joint_loss(y_true, y_pred):

    return 0.4 * jaccard_loss_b(y_true, y_pred) + 0.2 * soft_jaccard_loss(y_true, y_pred) + 0.2 * jaccard_loss(y_true, y_pred) + 0.2 * keras.losses.mean_squared_error(y_true, y_pred)

# A computation of the jaccard index
def jaccard_index(y_true, y_pred):
    
    safety = 0.001

    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)

    top = K.sum(K.minimum(y_true_f, y_pred_f))
    bottom = K.sum(K.maximum(y_true_f, y_pred_f))

    return top / (bottom + safety)

# An example loss based on jaccard index
def jaccard_loss(y_true, y_pred):

    return 1 - jaccard_index(y_true, y_pred)


# a 'soft' version of the jaccard index
def soft_jaccard_index(y_true, y_pred):

    safety = 0.001

    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)

    top = K.sum(y_true_f * y_pred_f)
    bottom = K.sum(y_true_f * y_true_f) + K.sum(y_pred_f * y_pred_f) - top

    return top / (bottom + safety)


def soft_jaccard_loss(y_true, y_pred):

    return 1 - soft_jaccard_index(y_true, y_pred)
def train_model(model,traindata_path,valdata_path):
    trainset = DataGenerator(traindata_path, batch_size=1)
    valset = DataGenerator(valdata_path, batch_size=1)
    #with one TITAN GPU, one epoch will only spend 5 minutes.
    #if it runs on CPU, one epoch will spend more than 1 hour.
    log_dir = "./logs/fit/"+datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir,histogram_freq=1)
    model.compile(optimizer=Adam(lr = 1e-4), loss=soft_jaccard_loss, metrics=[jaccard_loss_b, keras.losses.binary_crossentropy, joint_loss, keras.losses.mean_squared_error, soft_jaccard_loss, jaccard_loss,'accuracy'])

    model.fit(trainset,steps_per_epoch=500,epochs=10,validation_data=valset, validation_steps=1, validation_freq=1, callbacks=[tensorboard_callback])
    model.save_weights("model.h5")
    return model
def test_model(model,testdata_path,num):
    testSet = DataGenerator(testdata_path, batch_size=5)
    

    if not os.path.exists("./results"): os.mkdir("./results")
    
    for idx, (img, mask) in enumerate(testSet):
        
        cv2.imwrite("./results/origin_%d.png" %idx, img[0]*255.)
        cv2.imwrite("./results/mask_%d.png" %idx, mask[0]*255.)
        pred_mask = model.predict(img)[0]
        #cv2.imwrite("./results/mask1234_%d.png" %idx, pred_mask*255.)
        pred_mask[pred_mask > 0.5] = 1
        pred_mask[pred_mask <= 0.5] = 0


        image_accuracy = np.mean(mask[0,:,:,0] == pred_mask[:,:,0])
        
        image_path = "./results/pred_"+str(idx)+".png"
        print("=> accuracy: %.4f, saving %s" %(image_accuracy, image_path))
        cv2.imwrite(image_path, pred_mask[:,:,0]*255.)
        
        
        if idx == num: break
if __name__=="__main__":
    
    traimg_path=r"C:\Users\17733\Desktop\buildings\train"
    valimg_path=r"C:\Users\17733\Desktop\buildings\train"
    testimg_path=r"C:\Users\17733\Desktop\buildings\train"
    # set the pipeline to be executed and hyperparameters
    MODEL = 'DVAEr'
    LOSS_TYPE = 'FL' # can be one of BCE, BCEw, FL

    # picking a model according to the selection
    if MODEL == 'Unet':
        nets = Unet_2levels()
    elif MODEL == 'Dunet':
        nets = Dunet_2levels()
    elif MODEL == 'DVAE':
        #ZDIM number of feature maps of the 3D encoding space, matters when using the DVAE refiner model
        nets = DVAE(zdim=100,batchsize =1)
    else:
        nets = DVAE_refiner(zdim=100,batchsize=1)
    model = nets(input_size = (512,512,3))
    if not os.path.exists("./model.h5"):
        model=train_model(model,traimg_path,valimg_path)
    else:
        imglst=os.listdir(testimg_path+'/image')
        model.load_weights("model.h5")
        test_model(model,testimg_path,len(imglst))
