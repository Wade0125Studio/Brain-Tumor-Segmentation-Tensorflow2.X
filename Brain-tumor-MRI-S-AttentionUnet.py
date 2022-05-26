import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
from skimage import io
import cv2
import random
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=10240)])
  except RuntimeError as e:
    print(e)
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import add
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import multiply
from tensorflow.keras.layers import concatenate
from tensorflow.keras import backend as K
from tensorflow.keras.metrics import MeanIoU
from sklearn.model_selection import train_test_split
from tensorflow import keras
from plot_keras_history import show_history, plot_history
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint,ReduceLROnPlateau
from tensorflow.keras.models import Model, load_model, save_model


path = "C:/Users/user/Downloads/Brain_Tumor_Segmentation/lgg-mri-segmentation/"

df = pd.read_csv(path + "data.csv")
df.head()

masks_dir = glob(path + "*/*_mask*")

images_dir = []
for img in masks_dir:
    images_dir.append(img.replace("_mask", ""))



# Create a new datafarme for brain images and masks
data_brain = pd.DataFrame(data={"file_images":images_dir,"file_masks":masks_dir  })
data_brain.head()


def positive_negative_diagnosis(file_masks):
    mask = cv2.imread(file_masks)
    value = np.max(mask)
    if value > 0:
        return 1
    else:
        return 0


data_brain["mask"] = data_brain["file_masks"].apply(lambda x: positive_negative_diagnosis(x))
data_brain


data_brain["mask"].value_counts()


# **Data Visualization**

# In[11]:


def show_image(df):

  fig, ax = plt.subplots(nrows=10, ncols=3, figsize=(20, 40))
  count = 0
  i = 0
  for mask in df["mask"]:
    if mask == 1:
      # Show images
      image = io.imread(df.file_images[i])
      ax[count][0].title.set_text("Brain MRI")
      ax[count][0].imshow(image)

      # Show masks
      mask = io.imread(df.file_masks[i])
      ax[count][1].title.set_text("Mask Brain MRI")
      ax[count][1].imshow(mask, cmap="gray")

      # Show MRI Brain with mask
      image[mask == 255] = (0, 255, 0)    # Here, we want to modify the color of pixel at the position of mask
      ax[count][2].title.set_text("MRI Brain with mask")
      ax[count][2].imshow(image)
      count += 1
    i += 1
    if count == 10:
      break
  fig.tight_layout()
  plt.show()


show_image(data_brain)


class ImageLoading():
  def __init__(self, img_path, mask_path):
    self.img_path = img_path
    self.mask_path = mask_path
    self.IMG_HEIGHT = 256
    self.IMG_WIDTH = 256
    # The number of classes for segmentation
    self.NUM_CLASSES = 1
    # Load images
    self.images_training = self.resize_images()
    print(self.images_training.shape)
    # Load masks
    self.masks_training = self.resize_masks()
    print(self.masks_training.shape)
  # resise the image
  def resize_images(self):
    images_training = []
    for imagePath in self.img_path:
      image = cv2.imread(imagePath)
      image = cv2.resize(image, (self.IMG_WIDTH, self.IMG_HEIGHT))
      images_training.append(image)
    # Convert to numpy array
    images_training = np.array(images_training)
    return images_training

  # resise the mask
  def resize_masks(self):
    masks_training = []
    for maskPath in self.mask_path:
      mask = cv2.imread(maskPath, 0)
      mask = cv2.resize(mask, (self.IMG_WIDTH, self.IMG_HEIGHT), interpolation=cv2.INTER_NEAREST)
      masks_training.append(mask)
    # Convert to numpy array
    masks_training = np.array(masks_training)
    return masks_training

image_loader = ImageLoading(images_dir[:1000], masks_dir[:1000])
images_train = image_loader.resize_images()
masks_train = image_loader.resize_masks()


# Normalize images
images_train = np.array(images_train) / 255.
masks_train = np.expand_dims((np.array(masks_train)), 3) /255.


X_train, X_test, Y_train, Y_test = train_test_split(images_train, masks_train, test_size=0.2, random_state=42)

print("X_train shape = {}".format(X_train.shape))
print("X_test shape = {}".format(X_test.shape))
print("Y_train shape = {}".format(Y_train.shape))
print("Y_test shape = {}".format(Y_test.shape))

IMG_HEIGHT = X_train.shape[1]
IMG_WIDTH = X_train.shape[2]
IMG_CHANNELS = X_train.shape[3]
# Binary class
NUM_CLASS = 1
BATCH_SIZE = 8
input_shape = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

# loss function and metrics
epsilon = 1e-5
smooth = 1

def dice_loss(y_true, y_pred):
    y_true=K.flatten(y_true)
    y_pred=K.flatten(y_pred)
    intersec=K.sum(y_true* y_pred)
    return (-((2* intersec + 0.1) / (K.sum(y_true) + K.sum(y_pred) + 0.1)))

def iou(y_true,y_pred):
    intersec = K.sum(y_true * y_pred)
    union = K.sum(y_true + y_pred)
    iou = (intersec + 0.1) / (union- intersec + 0.1)
    return iou
def dice_coef(y_pred, Y):
    y_flatten = K.flatten(Y)
    y_pred_flatten = K.flatten(y_pred)
    intersection = K.sum(y_flatten * y_pred_flatten)
    dice = (0.2 * intersection + 1.0) / (K.sum(y_flatten) + K.sum(y_pred_flatten) + 1.0)
    return dice

def jacard_coef(y_pred, Y):
    y_flatten = K.flatten(Y)
    y_pred_flatten = K.flatten(y_pred)
    intersection = K.sum(y_flatten * y_pred_flatten)
    jacard = (intersection + 1.0) / (K.sum(y_flatten) + K.sum(y_pred_flatten) - intersection + 1.0)
    return jacard

def jacard_coef_loss(y_pred, Y):
    return -jacard_coef(y_pred, Y)

def dice_coef_loss(y_pred, Y):
    return -dice_coef(y_pred, Y)
def tversky(y_true, y_pred):
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)
    true_pos = K.sum(y_true_pos * y_pred_pos)
    false_neg = K.sum(y_true_pos * (1-y_pred_pos))
    false_pos = K.sum((1-y_true_pos)*y_pred_pos)
    alpha = 0.7
    return (true_pos + smooth)/(true_pos + alpha*false_neg + (1-alpha)*false_pos + smooth)
def focal_tversky(y_true,y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    pt_1 = tversky(y_true, y_pred)
    gamma = 0.75
    return K.pow((1-pt_1), gamma)
def tversky_loss(y_true, y_pred):
    return 1 - tversky(y_true,y_pred)
def dice_coef(y_true, y_pred):
    y_truef = keras.backend.flatten(y_true)
    y_predf = keras.backend.flatten(y_pred)
    And = keras.backend.sum(y_truef*y_predf)
    return((2* And + smooth) / (keras.backend.sum(y_truef) + keras.backend.sum(y_predf) + smooth))
def jac_distance(y_true, y_pred):
    y_truef = keras.backend.flatten(y_true)
    y_predf = keras.backend.flatten(y_pred)
    return - iou(y_true, y_pred)
def dice_coef(y_true, y_pred):
    y_truef = keras.backend.flatten(y_true)
    y_predf = keras.backend.flatten(y_pred)
    And = keras.backend.sum(y_truef*y_predf)
    return((2* And + smooth) / (keras.backend.sum(y_truef) + keras.backend.sum(y_predf) + smooth))
def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)
from tensorflow.keras.metrics import binary_crossentropy

# Attention Unet
class Attention_UNet():
    def __init__(self, input_shape, num_classes=1, dropout=0, BatchNorm=True):
        super(Attention_UNet, self).__init__()
        self.max_pooling = True
        self.num_classes = num_classes
        self.input_shape = input_shape
        # number of basic filters for the first layer
        self.num_filters = 64
        # size of the convolutional filter
        self.filter_size = 3
        # size of upsampling filters
        self.upsampling_filter = 2
    def make_conv_block(self, input_layer, filter_size, num_filters, dropout=0, BatchNorm=False):
        conv_layer = Conv2D(num_filters, (filter_size, filter_size), padding="same")(input_layer)
        if BatchNorm is True:
            conv_layer = BatchNormalization(axis=3)(conv_layer)
        conv_layer = Activation("relu")(conv_layer)
        conv_layer = Conv2D(num_filters, (filter_size, filter_size), padding="same")(conv_layer)
        if BatchNorm is True:
            conv_layer = BatchNormalization(axis=3)(conv_layer)
        conv_layer = Activation("relu")(conv_layer)
        if dropout > 0:
            conv_layer = Dropout(dropout)(conv_layer)
        return conv_layer
    def make_repeat_elements(self, tensor, rep):
        return Lambda(lambda x, repnum: K.repeat_elements(x, repnum, axis=3), arguments={"repnum":rep})(tensor)
    def _gate_signal(self, input_channels, output_channels, BatchNorm=False):
        g = Conv2D(output_channels, (1,1), padding="same")(input_channels)
        if BatchNorm:
            g = BatchNormalization()(g)
        g = Activation("relu")(g)
        return g
    # We add attention block after shortcut connection in UNet
    def make_attention_block(self, input_layer, gating, num_filters):
        input_layer_shape = K.int_shape(input_layer)
        gating_shape = K.int_shape(gating)
        # Here, we should get the input_layer signal to the same shape as the gating signal
        input_layer_theta = Conv2D(num_filters, (2, 2), strides = (2, 2), padding = "same")(input_layer)
        input_layer_theta_shape = K.int_shape(input_layer_theta)
        # we should get the gating signal to the same number of filters as the num_filters
        gating_phi = Conv2D(num_filters, (1, 1), padding = "same")(gating)
        gating_upsample = Conv2DTranspose(num_filters,
                                          (3, 3),
                                          strides = (input_layer_theta_shape[1] // gating_shape[1],
                                                     input_layer_theta_shape[2] // gating_shape[2]),
                                          padding = "same")(gating_phi)
        concat_layer = add([gating_upsample, input_layer_theta])
        concat_layer = Activation("relu")(concat_layer)
        concat_layer = Conv2D(1, (1, 1), padding="same")(concat_layer)
        concat_layer = Activation("sigmoid")(concat_layer)   # To get weigth between 0 and 1
        concat_layer_shape = K.int_shape(concat_layer)
        concat_layer_upsampling = UpSampling2D(size = (input_layer_shape[1] // concat_layer_shape[1],
                                                       input_layer_shape[2] // concat_layer_shape[2]))(concat_layer)
        concat_layer_upsampling = self.make_repeat_elements(concat_layer_upsampling, input_layer_shape[3])
        y = multiply([concat_layer_upsampling, input_layer])
        # Final layer
        conv_result = Conv2D(input_layer_shape[3], (1, 1), padding="same")(y)
        conv_result_batchNorm = BatchNormalization()(conv_result)
        return conv_result_batchNorm
    def build_attention_unit(self, dropout=0, BatchNorm=True):
        input_layer = Input(self.input_shape, dtype=tf.float32)

        ############ Add downsampling layer ############
        # Block 1, 128
        encoder_128 = self.make_conv_block(input_layer, self.filter_size, self.num_filters, dropout=dropout, BatchNorm=BatchNorm)
        if self.max_pooling:
            encoder_pool_64 = MaxPooling2D(pool_size=(2, 2))(encoder_128)
        # Block 2, 64 layer
        encoder_64 = self.make_conv_block(encoder_pool_64, self.filter_size, 2 * self.num_filters, dropout=dropout, BatchNorm=BatchNorm)
        if self.max_pooling:
            encoder_pool_32 = MaxPooling2D(pool_size=(2, 2))(encoder_64)
        # Block 3, 32 layer
        encoder_32 = self.make_conv_block(encoder_pool_32, self.filter_size, 4 * self.num_filters, dropout=dropout, BatchNorm=BatchNorm)
        if self.max_pooling:
            encoder_pool_16 = MaxPooling2D(pool_size=(2, 2))(encoder_32)
        # Block 4, 8 layer
        encoder_16 = self.make_conv_block(encoder_pool_16, self.filter_size, 8 * self.num_filters, dropout=dropout, BatchNorm=BatchNorm)
        if self.max_pooling:
            encoder_pool_8 = MaxPooling2D(pool_size=(2, 2))(encoder_16)
        # Block 5, just convolutional block
        encoder_8 = self.make_conv_block(encoder_pool_8, self.filter_size, 16 * self.num_filters, dropout=dropout, BatchNorm=BatchNorm)

        ############ Upsampling layers #############
        # Block 6, attention gated concatenation + upsampling + double residual convolution
        gate_16 = self._gate_signal(encoder_8, 8 * self.num_filters, BatchNorm=BatchNorm)
        attention_block_16 = self.make_attention_block(encoder_16, gate_16, 8 * self.num_filters)
        decoder_16 = UpSampling2D(size=(self.upsampling_filter, self.upsampling_filter), data_format="channels_last")(
            encoder_8)
        decoder_16 = concatenate([decoder_16, attention_block_16], axis=3)
        decoder_conv_16 = self.make_conv_block(decoder_16, self.filter_size, 8 * self.num_filters, dropout=dropout, BatchNorm=BatchNorm)
        # Block 7
        gate_32 = self._gate_signal(decoder_conv_16, 4 * self.num_filters, BatchNorm=BatchNorm)
        attention_block_32 = self.make_attention_block(encoder_32, gate_32, 4 * self.num_filters)
        decoder_32 = UpSampling2D(size=(self.upsampling_filter, self.upsampling_filter), data_format="channels_last")(
            decoder_conv_16)
        decoder_32 = concatenate([decoder_32, attention_block_32], axis=3)
        decoder_conv_32 = self.make_conv_block(decoder_32, self.filter_size, 4 * self.num_filters, dropout=dropout, BatchNorm=BatchNorm)
        # Block 8
        gate_64 = self._gate_signal(decoder_conv_32, 2 * self.num_filters, BatchNorm=BatchNorm)
        attention_block_64 = self.make_attention_block(encoder_64, gate_64, 2 * self.num_filters)
        decoder_64 = UpSampling2D(size=(self.upsampling_filter, self.upsampling_filter), data_format="channels_last")(
            decoder_conv_32)
        decoder_64 = concatenate([decoder_64, attention_block_64], axis=3)
        decoder_conv_64 = self.make_conv_block(decoder_64, self.filter_size, 2 * self.num_filters, dropout=dropout, BatchNorm=BatchNorm)
        # Block 9
        gate_128 = self._gate_signal(decoder_conv_64, self.num_filters, BatchNorm=BatchNorm)
        attention_block_128 = self.make_attention_block(encoder_128, gate_128, self.num_filters)
        decoder_128 = UpSampling2D(size=(self.upsampling_filter, self.upsampling_filter), data_format="channels_last")(decoder_conv_64)
        decoder_128 = concatenate([decoder_128, attention_block_128], axis=3)
        decoder_conv_128 = self.make_conv_block(decoder_128, self.filter_size, self.num_filters, dropout=dropout, BatchNorm=BatchNorm)

        # Final convolutional layers (1 * 1)
        final_conv_lr = Conv2D(self.num_classes, kernel_size=1)(decoder_conv_128)
        final_conv_lr = BatchNormalization(axis=3)(final_conv_lr)
        # If a binary classification, we need to set "sigmoid" while for multichannel we should change to softmax
        final_conv_lr = Activation("sigmoid")(final_conv_lr)

        # Set the model
        model = Model(input_layer, final_conv_lr, name="Attention_UNet")
        print(model.summary())
        return model

unet_model = Attention_UNet(input_shape)

att_unet_model = unet_model.build_attention_unit()

save_name='C:/Users/user/Downloads/Brain_Tumor_Segmentation/model_weight/AttentionUNet-segModel-weights.h5'
att_unet_model.compile(optimizer="adam", 
                       loss =[dice_coef_loss,focal_tversky],metrics=["binary_accuracy", iou, dice_coef,tversky,jacard_coef])
callbacks = [ModelCheckpoint(save_name,verbose=1,save_best_only=True),
             # ReduceLROnPlateau(monitor='val_loss', factor=0.1,verbose=1,patience=5),
             EarlyStopping(monitor='val_loss',mode='min', verbose=1,patience=20)]

history = att_unet_model.fit(X_train, Y_train, 
                             verbose=1,
                             batch_size = 8,
                             validation_data=(X_test, Y_test),
                             shuffle=True,
                             epochs=100)

plot_history(history, path="C:/Users/user/Downloads/Brain_Tumor_Segmentation/img/Training_history_AttentionUNet_Segmentation.png",
             title="Training history AttentionUNet Segmentation")

model.save(save_name)


num_img_test = random.randint(0, X_test.shape[0]-1)
img_test = X_test[num_img_test]
test_label = Y_test[num_img_test]

img_test_input = np.expand_dims(img_test, 0)
pred = (att_unet_model.predict(img_test_input)[0,:,:,0] > 0.5).astype(np.uint8)

# Visualize the result 
plt.figure(figsize=(15, 10))
plt.subplot(231)
plt.title('Testing Image')
plt.imshow(img_test, cmap='gray')
plt.subplot(232)
plt.title('Testing Label')
plt.imshow(test_label[:,:,0], cmap='gray')
plt.subplot(233)
plt.title('Prediction on test image')
plt.imshow(pred, cmap='gray')




plt.figure(figsize=(10,50))

titles = ["Image", "Original Mask", "Predicted Mask"]

for j in range(15):
    num_img_test = random.randint(0, X_test.shape[0]-1)
    img_test = X_test[num_img_test]
    test_label = Y_test[num_img_test]

    img_test_input = np.expand_dims(img_test, 0)
    pred = (att_unet_model.predict(img_test_input)[0,:,:,0] > 0.5).astype(np.uint8)
    images = [img_test,test_label[:,:,0],pred]
    for i in range(0, 3):
        ax = plt.subplot(15, 3, (j*3)+i+1)
        ax.set_title(titles[i], fontsize = 16)
        plt.imshow(X=images[i], cmap='gray')
        
plt.tight_layout()
plt.savefig("C:/Users/user/Downloads/Brain_Tumor_Segmentation/AttentionUNet predict result.png")
plt.show()




