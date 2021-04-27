import tensorflow as tf
import tensorflow.keras.layers as layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model
from utils import *
import tensorflow_addons as tfa
# =============================================================================
# # Custom loss layer
# class CustomVariationalLayer(layers.Layer):
#     def __init__(self,mu,logvar,z, **kwargs):
#         self.is_placeholder = True
#         super(CustomVariationalLayer, self).__init__(**kwargs)
# 
#     def vae_loss(self, x, x_decoded_mean):
#         xent_loss = original_dim * metrics.binary_crossentropy(x, x_decoded_mean)#Square Loss
#         kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)# KL-Divergence Loss
#         return K.mean(xent_loss + kl_loss)
# 
#     def call(self, inputs):
#         x = inputs[0]
#         x_decoded_mean = inputs[1]
#         loss = self.vae_loss(x, x_decoded_mean)
#         self.add_loss(loss, inputs=inputs)
#         # We won't actually use the output.
#         return x
# =============================================================================
alpha = 0.6
class Unet_2levels(object):
    def __init__(self):
        super().__init__()

        self.relu = tf.keras.layers.ReLU()

        self.upsample = layers.UpSampling2D(size = (2,2), interpolation='bilinear')
        self.maxpool = layers.MaxPool2D(pool_size=(2,2),strides=2,padding='same')

        self.l11 = layers.Conv2D(64,(3,3),padding='same')
        self.l12 = layers.Conv2D(64,(3,3),padding='same')
        self.l21 = layers.Conv2D(128,(3,3),padding='same')
        self.l22 = layers.Conv2D(128,(3,3),padding='same')
        self.l31 = layers.Conv2D(256,(3,3),padding='same')
        self.l32 = layers.Conv2D(256,(3,3),padding='same')
        self.l41 = layers.Conv2D(256,(3,3),padding='same')
        self.l42 = layers.Conv2D(128,(3,3),padding='same')
        self.l51 = layers.Conv2D(64,(3,3),padding='same')
        self.l52 = layers.Conv2D(64,(3,3),padding='same')
        self.l53 = layers.Conv2D(1,(1,1),padding='same')

        self.up1 = layers.Conv2DTranspose(128,(2,2),strides = (2,2),padding='same')
        self.up2 = layers.Conv2DTranspose(64,(2,2),strides = (2,2),padding='same')

    def __call__(self, inputs = None, input_size = None,phase = 'training',**kwargs):
        assert inputs is not None or input_size is not None
        
        if inputs is None:
            assert isinstance(input_size, tuple)
            inputs = layers.Input(shape = (input_size))
        x = inputs
        h11 = self.relu(self.l11(x))
        h12 = self.relu(self.l12(h11))
        h21 = self.relu(self.l21(self.maxpool(h12)))
        h22 = self.relu(self.l22(h21))
        h31 = self.relu(self.l31(self.maxpool(h22)))
        h32 = self.relu(self.l32(h31))

        h41 = self.relu(self.l41(layers.concatenate([h22, self.up1(h32)],axis = -1)))
        h42 = self.relu(self.l42(h41))
        h51 = self.relu(self.l51(layers.concatenate([h12, self.up2(h42)],axis = -1)))
        h52 = self.relu(self.l52(h51))
        output = tf.keras.activations.sigmoid(self.l53(h52))
        if phase == 'calling':
            return output
        else:
            model = Model(x,output)
            model.summary()
            model.compile(optimizer = Adam(lr = 1e-3),loss='binary_crossentropy',metrics = ['accuracy',topo_metric])
            return model
   
def DUNET_loss(y_true,y_pred):
    bce1 = 0.5*tf.keras.losses.binary_crossentropy(y_true,y_pred[0])
    bce2 = 0.5*tf.keras.losses.binary_crossentropy(y_true,y_pred[1])
    return alpha*bce1+(1-alpha)*bce2
class Dunet_2levels(object):
    def __init__(self):
        super().__init__()

        self.segmentator = Unet_2levels()
        self.refiner = Unet_2levels()

    def segment(self, x,phase):
        return self.segmentator(x,input_size = None,phase=phase)

    def refine(self, x,phase):
        return self.refiner(x,input_size = None,phase=phase)

    def __call__(self, inputs = None, input_size = None,phase = 'training',**kwargs):
        assert inputs is not None or input_size is not None
        
        if inputs is None:
            assert isinstance(input_size, tuple)
            inputs = layers.Input(shape = (input_size))
        x = inputs
        seg = self.segment(x,'calling')
        ref = self.refine(seg,'calling')
        if phase == 'calling':
            return [seg,ref]
        else:
                
            model = Model(x,[seg,ref])
            model.summary()
            model.compile(optimizer = Adam(lr = 1e-3),loss=DUNET_loss,metrics = ['accuracy',topo_metric])
            return model

class DVAE(object):
    def __init__(self, zdim=100, batchsize = 5):
        super().__init__()
        
        self.relu = tf.keras.layers.ReLU()

        self.upsample = layers.UpSampling2D(size = (2,2), interpolation='bilinear')
        self.maxpool = layers.MaxPool2D(pool_size=(2,2),strides=2,padding='same')

        self.enc_l1 = layers.Conv2D(64,(3,3),padding='same',dilation_rate=(1,1))
        self.enc_l2 = layers.Conv2D(64,(3,3),padding='same',dilation_rate=(1,1))
        self.enc_l3 = layers.Conv2D(256,(3,3),padding='same',dilation_rate=(1,1))
        self.enc_l4 = layers.Conv2D(256,(3,3),padding='same',dilation_rate=(1,1))
        self.enc_l51 = layers.Conv2D(zdim,(1,1),padding='same',dilation_rate=(1,1))
        self.enc_l52 = layers.Conv2D(zdim,(1,1),padding='same',dilation_rate=(1,1))

        self.dec_l1 = layers.Conv2DTranspose(256,(4,4),strides = (2,2),padding='same')
        self.dec_l2 = layers.Conv2DTranspose(256,(4,4),strides = (2,2),padding='same')
        self.dec_l3 = layers.Conv2DTranspose(64,(4,4),strides = (2,2),padding='same')
        self.dec_l4 = layers.Conv2D(64,(3,3),padding='same',dilation_rate=(1,1))
        self.dec_l5 = layers.Conv2D(1,(3,3),padding='same',dilation_rate=(1,1))
        self.batchsize = batchsize 
    def encode(self, x):
        enc_h1 = self.relu(self.enc_l1(x))
        enc_h2 = self.relu(self.enc_l2(self.maxpool(enc_h1)))
        enc_h3 = self.relu(self.enc_l3(self.maxpool(enc_h2)))
        enc_h4 = self.relu(self.enc_l4(self.maxpool(enc_h3)))
        return self.enc_l51(enc_h4), self.enc_l52(enc_h4)
        
    def sample(self, mu, logvar, phase):
        if phase=='testing':
            return mu
        else:
            std = tf.math.sqrt(tf.keras.activations.exponential(logvar))
            eps = tf.random.normal([std.shape[1],std.shape[2],std.shape[3]], 0, 1, tf.float32)

            return tf.expand_dims(eps,0) * std + mu

    def decode(self, z):
        dec_h1 = self.relu(self.dec_l1(z))
        dec_h2 = self.relu(self.dec_l2(dec_h1))
        dec_h3 = self.relu(self.dec_l3(dec_h2))
        dec_h4 = self.relu(self.dec_l4(dec_h3))
        return tf.keras.activations.sigmoid(self.dec_l5(dec_h4))

    def __call__(self, inputs = None, input_size = None,phase = 'training',**kwargs):
        assert inputs is not None or input_size is not None
        
        if inputs is None:
            assert isinstance(input_size, tuple)
            inputs = layers.Input(shape = (input_size))
        x = inputs
        mu, logvar = self.encode(x)
        z = self.sample(mu, logvar, 'calling')
        ref = self.decode(z)
        if phase == 'calling':
            return ref
        else:
            
            #mu, logvar, z, ref
            model = Model(x,ref)
            model.summary()
            model.compile(optimizer = Adam(lr = 1e-3),loss='binary_crossentropy',metrics = ['accuracy',topo_metric])
            return model

def DVAEr_loss(y_true,y_pred):
    bce1 = 0.5*tf.keras.losses.binary_crossentropy(y_true,y_pred[0])
    sfc = tfa.losses.SigmoidFocalCrossEntropy(y_true,y_pred[1])
    return alpha*bce1+(1-alpha)*sfc
class DVAE_refiner(object):
    def __init__(self, zdim=100,batchsize = 5):
        super().__init__()

        self.segmentator = Unet_2levels()
        self.refiner = DVAE(zdim,batchsize)
        self.batchsize = batchsize 
    def segment(self, x, phase):
        return self.segmentator(x,input_size = None,phase=phase)

    def refine(self, x,phase):
        return self.refiner(x,input_size = None,phase=phase)

    def __call__(self, inputs = None, input_size = None,phase = 'training',**kwargs):
        assert inputs is not None or input_size is not None
        
        if inputs is None:
            assert isinstance(input_size, tuple)
            inputs = layers.Input(shape = (input_size))
        x = inputs
        seg = self.segment(x,'calling')
        ref = self.refine(seg,'calling')
        #seg, mu, logvar, z, ref
        model = Model(x,[seg,ref])
        model.summary()

        model.compile(optimizer = Adam(lr = 1e-3),loss=DVAEr_loss,metrics = ['accuracy',topo_metric])
        return model
