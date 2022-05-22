import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import zipfile
import cv2
from skimage import io
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5120)])
  except RuntimeError as e:
    print(e)
from tensorflow.python.keras import Sequential
from tensorflow.keras import layers, optimizers
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.applications import ResNet50,InceptionV3,InceptionResNetV2
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, LearningRateScheduler
from IPython.display import display
from tensorflow.keras import backend as K
from sklearn.preprocessing import StandardScaler, normalize
import os
import glob
import random
get_ipython().run_line_magic('matplotlib', 'inline')
from plot_keras_history import show_history, plot_history
import os 
from skimage import io
from PIL import Image
from tensorflow.keras import backend as K

data = pd.read_csv('C:/Users/GIGABYTE/Downloads/Brain_Tumor_Segmentation/lgg-mri-segmentation/data.csv')
data.info()

data_map = []
for sub_dir_path in glob.glob("C:/Users/GIGABYTE/Downloads/Brain_Tumor_Segmentation/lgg-mri-segmentation/"+"*"):
    #if os.path.isdir(sub_path_dir):
    try:
        dir_name = sub_dir_path.split('/')[-1]
        for filename in os.listdir(sub_dir_path):
            image_path = sub_dir_path + '/' + filename
            data_map.extend([dir_name, image_path])
    except Exception as e:
        print(e)


df = pd.DataFrame({"patient_id" : data_map[::2],"path" : data_map[1::2]})
df_imgs = df[~df['path'].str.contains("mask")] # if have not mask
df_masks = df[df['path'].str.contains("mask")]# if have mask
imgs = sorted(df_imgs["path"].values)
masks = sorted(df_masks["path"].values)
idx = random.randint(0, len(imgs)-1)
print("Path to the Image:", imgs[idx], "\nPath to the Mask:", masks[idx])
brain_df = pd.DataFrame({"patient_id": df_imgs.patient_id.values,
                         "image_path": imgs,
                         "mask_path": masks
                        })
def pos_neg_diagnosis(mask_path):
    value = np.max(cv2.imread(mask_path))
    if value > 0:
        return 1
    else:
        return 0
brain_df['mask'] = brain_df['mask_path'].apply(lambda x: pos_neg_diagnosis(x))
brain_df


brain_df['mask'].value_counts()
sns.countplot(brain_df['mask'])
plt.show()



count = 0
i = 0
fig,axs = plt.subplots(12,3, figsize=(20,50))
for mask in brain_df['mask']:
    if (mask==1):
        img = io.imread(brain_df.image_path[i])
        axs[count][0].title.set_text("Brain MRI")
        axs[count][0].imshow(img)
        mask = io.imread(brain_df.mask_path[i])
        axs[count][1].title.set_text("Mask")
        axs[count][1].imshow(mask, cmap='gray')
        img[mask==255] = (255,0,0)  # change pixel color at the position of mask
        axs[count][2].title.set_text("MRI with Mask")
        axs[count][2].imshow(img)
        count +=1
    i += 1
    if (count==12):
        break
fig.tight_layout()
plt.savefig("C:/Users/GIGABYTE/Downloads/Brain Tumor Segmentation/show dataset.png")



# Drop the patient id column
brain_df_train = brain_df.drop(columns = ['patient_id'])
brain_df_train.shape
brain_df_train['mask'] = brain_df_train['mask'].apply(lambda x: str(x))
brain_df_train.info()

from sklearn.model_selection import train_test_split
train, test = train_test_split(brain_df_train, test_size = 0.15)
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Create a data generator which scales the data from 0 to 1 and makes validation split of 0.15
datagen = ImageDataGenerator(rescale=1./255., validation_split = 0.15)

train_generator=datagen.flow_from_dataframe(
dataframe=train,
directory= 'C:/Users/GIGABYTE/Downloads/Brain_Tumor_Segmentation/lgg-mri-segmentation/',
x_col='image_path',
y_col='mask',
subset="training",
batch_size=16,
shuffle=True,
class_mode="categorical",
target_size=(256,256))

valid_generator=datagen.flow_from_dataframe(
dataframe=train,
directory= 'C:/Users/GIGABYTE/Downloads/Brain_Tumor_Segmentation/lgg-mri-segmentation/',
x_col='image_path',
y_col='mask',
subset="validation",
batch_size=16,
shuffle=True,
class_mode="categorical",
target_size=(256,256))
# Create a data generator for test images
test_datagen=ImageDataGenerator(rescale=1./255.)
test_generator=test_datagen.flow_from_dataframe(
dataframe=test,
directory= 'C:/Users/GIGABYTE/Downloads/Brain_Tumor_Segmentation/lgg-mri-segmentation/',
x_col='image_path',
y_col='mask',
batch_size=16,
shuffle=False,
class_mode='categorical',
target_size=(256,256))

#----------------------Build-Classifer-Model----------------------------
basemodel = InceptionResNetV2(weights ='imagenet', include_top = False, input_tensor = Input(shape=(256, 256, 3)))
#basemodel.summary()
for layer in basemodel.layers:
  layers.trainable = False
headmodel = basemodel.output
headmodel = AveragePooling2D(pool_size = (4,4))(headmodel)
headmodel = Flatten(name= 'flatten')(headmodel)
headmodel = Dense(512, activation = "relu")(headmodel)
headmodel = Dropout(0.4)(headmodel)#
headmodel = Dense(512, activation = "relu")(headmodel)
headmodel = Dropout(0.4)(headmodel)
#headmodel = Dense(256, activation = "relu")(headmodel)
#headmodel = Dropout(0.3)(headmodel)
headmodel = Dense(2, activation = 'softmax')(headmodel)

model = Model(inputs = basemodel.input, outputs = headmodel)

model.summary()
model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics= ["accuracy"])

earlystopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)
checkpointer = ModelCheckpoint(filepath="C:/Users/GIGABYTE/Downloads/Brain_Tumor_Segmentation/classifier-InceptionResNetV2-weights.h5", verbose=1, save_best_only=True)

history = model.fit(train_generator,epochs=100,validation_data=valid_generator,callbacks=[checkpointer, earlystopping])

plot_history(history, path="C:/Users/GIGABYTE/Downloads/Brain_Tumor_Segmentation/img/Training_history_InceptionResNetV2_classify.png",
             title="Training history InceptionResNetV2 classify")

prediction = model.predict(test_generator)
pred = np.argmax(prediction, axis=1)
original = np.asarray(test['mask']).astype('int')
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
accuracy = accuracy_score(original, pred)
print(accuracy)

conf_matrix = confusion_matrix(original, pred)
sns.heatmap(conf_matrix,xticklabels = ["Normal","Brain Tumour"], yticklabels =["Normal","Brain Tumour"],annot=True,fmt='g')
plt.title('InceptionResNetV2 Confusion Matrix Accuracy'+str(np.round(accuracy,4)))
plt.savefig('C:/Users/GIGABYTE/Downloads/Brain_Tumor_Segmentation/img/InceptionResNetV2 predict confusion matrix.png')    
plt.show()



#----------------------Build-Segmentation-Model----------------------------


brain_df_mask = brain_df[brain_df['mask'] == 1]
brain_df_mask.shape

# creating test, train and val sets
X_train, X_val = train_test_split(brain_df_mask, test_size=0.1)
X_test, X_val = train_test_split(X_val, test_size=0.5)
print("Train size is {}, valid size is {} & test size is {}".format(len(X_train), len(X_val), len(X_test)))

train_ids = list(X_train.image_path)
train_mask = list(X_train.mask_path)
val_ids = list(X_val.image_path)
val_mask= list(X_val.mask_path)


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, ids , mask, image_dir = ' ', batch_size = 16, img_h = 256, img_w = 256, shuffle = True):
        self.ids = ids
        self.mask = mask
        self.image_dir = image_dir
        self.batch_size = batch_size
        self.img_h = img_h
        self.img_w = img_w
        self.shuffle = shuffle
        self.on_epoch_end()
    def __len__(self):
        'Get the number of batches per epoch'
        return int(np.floor(len(self.ids)) / self.batch_size)
    def __getitem__(self, index):
        'Generate a batch of data'
        #generate index of batch_size length
        indexes = self.indexes[index* self.batch_size : (index+1) * self.batch_size]
        #get the ImageId corresponding to the indexes created above based on batch size
        list_ids = [self.ids[i] for i in indexes]
        #get the MaskId corresponding to the indexes created above based on batch size
        list_mask = [self.mask[i] for i in indexes]
        #generate data for the X(features) and y(label)
        X, y = self.__data_generation(list_ids, list_mask)
        #returning the data
        return X, y
    def on_epoch_end(self):
        self.indexes = np.arange(len(self.ids))
        #if shuffle is true, shuffle the indices
        if self.shuffle:
            np.random.shuffle(self.indexes)
    def __data_generation(self, list_ids, list_mask):
        'generate the data corresponding the indexes in a given batch of images'
        # create empty arrays of shape (batch_size,height,width,depth) 
        #Depth is 3 for input and depth is taken as 1 for output becasue mask consist only of 1 channel.
        X = np.empty((self.batch_size, self.img_h, self.img_w, 3))
        y = np.empty((self.batch_size, self.img_h, self.img_w, 1))
    #iterate through the dataframe rows, whose size is equal to the batch_size
        for i in range(len(list_ids)):
            #path of the image
            img_path = str(list_ids[i])
            #mask path
            mask_path = str(list_mask[i])
            #reading the original image and the corresponding mask image
            img = io.imread(img_path)
            mask = io.imread(mask_path)
            #resizing and coverting them to array of type float64
            img = cv2.resize(img,(self.img_h,self.img_w))
            img = np.array(img, dtype = np.float64)
            mask = cv2.resize(mask,(self.img_h,self.img_w))
            mask = np.array(mask, dtype = np.float64)
            #standardising 
            img -= img.mean()
            img /= img.std()
            mask -= mask.mean()
            mask /= mask.std()
            #Adding image to the empty array
            X[i,] = img
            #expanding the dimnesion of the image from (256,256) to (256,256,1)
            y[i,] = np.expand_dims(mask, axis = 2)
        #normalizing y
        y = (y > 0).astype(int)
        return X, y
train_data = DataGenerator(train_ids, train_mask)
val_data = DataGenerator(val_ids, val_mask)

def resblock(X, f):
    X_copy = X
    X = Conv2D(f, kernel_size = (1,1) ,strides = (1,1),kernel_initializer ='he_normal')(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X) 
    X = Conv2D(f, kernel_size = (3,3), strides =(1,1), padding = 'same', kernel_initializer ='he_normal')(X)
    X = BatchNormalization()(X)
    X_copy = Conv2D(f, kernel_size = (1,1), strides =(1,1), kernel_initializer ='he_normal')(X_copy)
    X_copy = BatchNormalization()(X_copy)
    X = Add()([X,X_copy])
    X = Activation('relu')(X)
    return X

def upsample_concat(x, skip):
    x = UpSampling2D((2,2))(x)
    merge = Concatenate()([x, skip])
    return merge



input_shape = (256,256,3)
X_input = Input(input_shape)

conv1_in = Conv2D(16,3,activation= 'relu', padding = 'same', kernel_initializer ='he_normal')(X_input)
conv1_in = BatchNormalization()(conv1_in)
conv1_in = Conv2D(16,3,activation= 'relu', padding = 'same', kernel_initializer ='he_normal')(conv1_in)
conv1_in = BatchNormalization()(conv1_in)
pool_1 = MaxPool2D(pool_size = (2,2))(conv1_in)
conv2_in = resblock(pool_1, 32)
pool_2 = MaxPool2D(pool_size = (2,2))(conv2_in)
conv3_in = resblock(pool_2, 64)
pool_3 = MaxPool2D(pool_size = (2,2))(conv3_in)
conv4_in = resblock(pool_3, 128)
pool_4 = MaxPool2D(pool_size = (2,2))(conv4_in)
conv5_in = resblock(pool_4, 256)
up_1 = upsample_concat(conv5_in, conv4_in)
up_1 = resblock(up_1, 128)
up_2 = upsample_concat(up_1, conv3_in)
up_2 = resblock(up_2, 64)
up_3 = upsample_concat(up_2, conv2_in)
up_3 = resblock(up_3, 32)
up_4 = upsample_concat(up_3, conv1_in)
up_4 = resblock(up_4, 16)
output = Conv2D(1, (1,1), padding = "same", activation = "sigmoid")(up_4)
model_seg = Model(inputs = X_input, outputs = output )
model_seg.summary()

from tensorflow.keras.metrics import binary_crossentropy
epsilon = 1e-5
smooth = 1
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

adam = tf.keras.optimizers.Adam(lr = 0.001, epsilon = 0.1)
model_seg.compile(optimizer = adam,loss = focal_tversky, metrics = [tversky])
earlystopping = EarlyStopping(monitor='val_loss',mode='min',verbose=1,patience=20)

# save the best model with lower validation loss
checkpointer = ModelCheckpoint(filepath="C:/Users/GIGABYTE/Downloads/Brain_Tumor_Segmentation/ResUNet-segModel-weights.h5", verbose=1,save_best_only=True) 
reduce_lr = ReduceLROnPlateau(monitor='val_loss',mode='min',verbose=1,patience=10,min_delta=0.0001,factor=0.2)

hist = model_seg.fit(train_data, epochs = 100, validation_data = val_data,callbacks = [checkpointer, earlystopping, reduce_lr])
                  

plot_history(hist, path="C:/Users/GIGABYTE/Downloads/Brain_Tumor_Segmentation/img/Training_history_ResNet-Unet_Segmentation.png",
             title="Training history ResNet-Unet Segmentation")

test_ids = list(X_test.image_path)
test_mask = list(X_test.mask_path)
test_data = DataGenerator(test_ids, test_mask)
_, tv = model_seg.evaluate(test_data)
print("Segmentation tversky is {:.2f}%".format(tv*100))


def prediction(test, model, model_seg):
    mask, image_id, has_mask = [], [], []
    for i in test.image_path:
        img = io.imread(i)
        img = img *1./255.
        img = cv2.resize(img, (256,256))
        img = np.array(img, dtype=np.float64)
        img = np.reshape(img, (1,256,256,3))
        is_defect = model.predict(img)
        if np.argmax(is_defect)==0:
            image_id.append(i)
            has_mask.append(0)
            mask.append('No mask :)')
            continue
        X = np.empty((1,256,256,3))
        img = io.imread(i)
        img = cv2.resize(img, (256,256))
        img = np.array(img, dtype=np.float64)
        img -= img.mean()
        img /= img.std()
        X[0,] = img
        predict = model_seg.predict(X)
        if predict.round().astype(int).sum()==0:
            image_id.append(i)
            has_mask.append(0)
            mask.append('No mask :)')
        else:
            image_id.append(i)
            has_mask.append(1)
            mask.append(predict)
    return pd.DataFrame({'image_path': image_id,'predicted_mask': mask,'has_mask': has_mask})

df_pred = prediction(test, model,model_seg)
df_pred

df_pred = test.merge(df_pred, on='image_path')
df_pred.head(10)

count = 0
fig, axs = plt.subplots(15,5, figsize=(30,90))
for i in range(len(df_pred)):
    if df_pred.has_mask[i]==1 and count<15:
        #read mri images
        img = io.imread(df_pred.image_path[i])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        axs[count][0].imshow(img)
        axs[count][0].title.set_text('Brain MRI')
        #read original mask
        mask = io.imread(df_pred.mask_path[i])
        axs[count][1].imshow(mask)
        axs[count][1].title.set_text('Original Mask')
        #read predicted mask
        pred = np.array(df_pred.predicted_mask[i]).squeeze().round()
        axs[count][2].imshow(pred)
        axs[count][2].title.set_text('AI predicted mask')
        #overlay original mask with MRI
        img[mask==255] = (255,0,0)
        axs[count][3].imshow(img)
        axs[count][3].title.set_text('Brain MRI with original mask (Ground Truth)')
        #overlay predicted mask and MRI
        img_ = io.imread(df_pred.image_path[i])
        img_ = cv2.cvtColor(img_, cv2.COLOR_BGR2RGB)
        img_[pred==1] = (0,255,150)
        axs[count][4].imshow(img_)
        axs[count][4].title.set_text('MRI with AI PREDICTED MASK')
        count +=1
    if (count==15):
        break
fig.tight_layout()
plt.savefig('C:/Users/GIGABYTE/Downloads/Brain_Tumor_Segmentation/img/ResNet-Unet Segmentation predict result.png')



