from lxml import etree
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
from skimage import io
from skimage.transform import resize
import keras
from keras.layers import UpSampling2D,Conv2D, MaxPooling2D, Flatten,Dropout,Input,BatchNormalization,Conv2DTranspose
from keras.layers.core import Dense
from keras.models import Sequential,Model,load_model
from keras.layers.core import Activation
from keras.callbacks import ModelCheckpoint,EarlyStopping,TensorBoard
from keras.layers.merge import concatenate,add
import tensorflow as tf
import keras.metrics
from sklearn.metrics import confusion_matrix


# parameters that you should set before running this script
filter = ['aeroplane', 'car', 'chair', 'dog', 'bird']       # select class, this default should yield 1489 training and 1470 validation images
voc_root_folder = "D:/Sem 2/CV/Project/VOCdevkit/"  # please replace with the location on your laptop where you unpacked the tarball
image_size = 64    # image size that you will use for your network (input images will be resampled to this size), lower if you have troubles on your laptop (hint: use io.imshow to inspect the quality of the resampled images before feeding it into your network!)


# step1 - build list of filtered filenames
annotation_folder = os.path.join(voc_root_folder, "VOC2009/Annotations")
annotation_files = os.listdir(annotation_folder)
filtered_filenames = []
for a_f in annotation_files:
    tree = etree.parse(os.path.join(annotation_folder, a_f))
    if np.any([tag.text == filt for tag in tree.iterfind(".\\name") for filt in filter]):
        filtered_filenames.append(a_f[:-4])

# step2 - build (x,y) for TRAIN/VAL (classification)
classes_folder = os.path.join(voc_root_folder, "VOC2009\\ImageSets\\Main")
classes_files = os.listdir(classes_folder)
train_files = [os.path.join(classes_folder, c_f) for filt in filter for c_f in classes_files if filt in c_f and '_train.txt' in c_f]
val_files = [os.path.join(classes_folder, c_f) for filt in filter for c_f in classes_files if filt in c_f and '_val.txt' in c_f]

#step3- build the segmentation dataset
classes_seg_folder = os.path.join(voc_root_folder, "VOC2009\\ImageSets\\Segmentation")
train_seg_files = [os.path.join(classes_seg_folder, 'train.txt')]
val_seg_files = [os.path.join(classes_seg_folder, 'val.txt')]

def build_classification_dataset(list_of_files):
    """ build training or validation set

    :param list_of_files: list of filenames to build trainset with
    :return: tuple with x np.ndarray of shape (n_images, image_size, image_size, 3) and  y np.ndarray of shape (n_images, n_classes)
    """
    temp = []
    train_labels = []
    for f_cf in list_of_files:
        with open(f_cf) as file:
            lines = file.read().splitlines()
            #print(lines)
            temp.append([line.split()[0] for line in lines if int(line.split()[-1]) == 1])
            #print(temp)
            label_id = [f_ind for f_ind, filt in enumerate(filter) if filt in f_cf][0]
            print(label_id)
            train_labels.append(len(temp[-1]) * [label_id])
            #print(train_labels)
    train_filter = [item for l in temp for item in l]
    print(train_filter)
    image_folder = os.path.join(voc_root_folder, "VOC2009/JPEGImages/")
    image_filenames = [os.path.join(image_folder, file) for f in train_filter for file in os.listdir(image_folder) if
                       f in file]
    x = np.array([resize(io.imread(img_f), (image_size, image_size, 3)) for img_f in image_filenames]).astype(
        'float32')
    # changed y to an array of shape (num_examples, num_classes) with 0 if class is not present and 1 if class is present
    y_temp = []
    for tf in train_filter:
        y_temp.append([1 if tf in l else 0 for l in temp])
    y = np.array(y_temp)

    return x, y

def build_segmentation_dataset(list_of_files):
    """ build training or validation set

    :param list_of_files: list of filenames to build trainset with
    :return: tuple with x np.ndarray of shape (n_images, image_size, image_size, 3) and  y np.ndarray of shape (n_images, n_classes)
    """
    temp = []
    train_labels = []
    for f_cf in list_of_files:
        with open(f_cf) as file:
            lines = file.read().splitlines()
            temp.append(lines)
            #label_id = [f_ind for f_ind, filt in enumerate(filter) if filt in f_cf][0]
            #train_labels.append(len(temp[-1]) * [label_id])
    train_filter = [item for l in temp for item in l]
    image_folder = os.path.join(voc_root_folder, "VOC2009/SegmentationClass/")
    image_filenames = [os.path.join(image_folder, file) for f in train_filter for file in os.listdir(image_folder) if
                       f in file]
    x = np.array([resize(cv2.cvtColor(cv2.imread(img_f),cv2.COLOR_BGR2RGB), (image_size, image_size, 3)) for img_f in image_filenames]).astype(
        'float32')
    # changed y to an array of shape (num_examples, num_classes) with 0 if class is not present and 1 if class is present
    y_temp = []
    for tf in train_filter:
        y_temp.append([1 if tf in l else 0 for l in temp])
    y = np.array(y_temp)
    return x, y

def build_segmentation(folder_type,root=voc_root_folder):
    txt_fname = '%s/VOC2009/ImageSets/Segmentation/%s' % (
        root, folder_type)
    with open(txt_fname, 'r') as f:
        images = f.read().split()
    features, labels = [None] * len(images), [None] * len(images)
    for i, fname in enumerate(images):
        features[i] = resize(io.imread('%s/VOC2009/JPEGImages/%s.jpg' % (root, fname)),(image_size, image_size, 3))
        labels[i] = resize(io.imread(
            '%s/VOC2009/SegmentationClass/%s.png' % (root, fname)),(image_size, image_size, 3))
    feat_array = np.array(features)
    label_array = np.array(labels)
    feat_array.reshape(-1,1)
    label_array.reshape(-1,1)
    return feat_array,label_array


x_train, y_train = build_classification_dataset(train_files)
io.imshow(x_train[0])
print('%i training images from %i classes' %(x_train.shape[0], y_train.shape[1]))
x_val, y_val = build_classification_dataset(val_files)
print('%i validation images from %i classes' %(x_val.shape[0],  y_train.shape[1]))

x_seg_train,y_seg_train = build_segmentation(folder_type = 'train.txt')
x_seg_val,y_seg_val = build_segmentation(folder_type = 'val.txt')

# from here, you can start building your model
# you will only need x_train and x_val for the autoencoder
# you should extend the above script for the segmentation task (you will need a slightly different function for building the label images)

def buildClassificationAutoEncoder(x_train,y_train,x_val,y_val,config):
    ##AutoEncoder w/ CNN
    autoencoder = Sequential()

    #Encoder part
    #autoencoder.add(Conv2D(128, (3, 3), activation='relu', padding='same', input_shape=x_train.shape[1:],trainable = config['trainable']))
    autoencoder.add(Conv2D(64, (3, 3), activation='relu', padding='same',trainable = config['trainable']))
    autoencoder.add(Conv2D(64, (3, 3), activation='relu',padding='same',trainable = config['trainable']))
    autoencoder.add(BatchNormalization(axis=-1))
    autoencoder.add(MaxPooling2D((2, 2),trainable = config['trainable']))
    autoencoder.add(Conv2D(32, (3, 3), activation='relu', padding='same',trainable = config['trainable']))
    autoencoder.add(Conv2D(32, (3, 3), activation='relu', padding='same',trainable = config['trainable']))
    autoencoder.add(BatchNormalization(axis=-1))
    #autoencoder.add(Conv2D(16, (3, 3), activation='relu',trainable = config['trainable']))
    autoencoder.add(MaxPooling2D((2, 2),trainable = config['trainable']))
    autoencoder.add(Conv2D(16, (3, 3), activation='relu', padding='same',trainable = config['trainable']))
    autoencoder.add(Conv2D(16, (3, 3), activation='relu',padding = 'same',trainable = config['trainable']))
    #autoencoder.add(Conv2D(8, (3, 3), activation='relu',trainable = config['trainable']))
    autoencoder.add(MaxPooling2D((2, 2),trainable = config['trainable']))
    autoencoder.add(Dropout(0.25))
    autoencoder.add(Conv2D(8, (3, 3), activation='relu', padding='same',trainable = config['trainable']))
    autoencoder.add(Conv2D(8, (3, 3), activation='relu',padding = 'same',trainable = config['trainable']))
    #autoencoder.add(Conv2D(8, (3, 3), activation='relu',trainable = config['trainable']))
    autoencoder.add(MaxPooling2D((2, 2), name = 'encode_layer',trainable = config['trainable']))
    autoencoder.add(Dropout(0.25))

    #Decoder part
    autoencoder.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
    autoencoder.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
    #autoencoder.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    #autoencoder.add(Conv2D(16, (3, 3), activation='relu'))
    autoencoder.add(UpSampling2D((2, 2)))
    autoencoder.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
    autoencoder.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
    #autoencoder.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    #autoencoder.add(Conv2D(16, (3, 3), activation='relu'))
    autoencoder.add(UpSampling2D((2, 2)))
    autoencoder.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    autoencoder.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    #autoencoder.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    #autoencoder.add(Conv2D(32, (3, 3), activation='relu'))
    autoencoder.add(UpSampling2D((2, 2)))
    autoencoder.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    autoencoder.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    #autoencoder.add(Conv2D(128, (3, 3), activation='relu'))
    autoencoder.add(UpSampling2D((2, 2)))
    autoencoder.add(Dropout(0.25))
    autoencoder.add(Conv2D(3,(1,1),activation='sigmoid',name = 'decode_layer'))
    #Train the autoencoder itself w/ output classification layer
    autoencoder.add(Flatten())
    #autoencoder.add(Dense(32))
    #autoencoder.add(Activation('relu'))
    #autoencoder.add(Dropout(0.25))
    autoencoder.add(Dense(y_train.shape[1],name= 'output_layer'))
    autoencoder.add(Activation('softmax'))

    autoencoder.compile(optimizer = 'adam', loss = 'categorical_crossentropy')

    checkpoint = ModelCheckpoint(filepath = config['filename'],monitor='val_loss'
                                ,verbose=1,save_best_only= True)
    earlystop = EarlyStopping(monitor='val_loss', min_delta=0,
                                patience=10, mode='auto', restore_best_weights=True)
    tensorboard = TensorBoard(log_dir=config['logdir'], histogram_freq=0,
                                write_graph=True, write_images=True)
    autoencoder.fit(x_train,y_train,validation_data =(x_val,y_val),epochs = config['epoch'], batch_size = config['batch_size'],callbacks= [checkpoint,earlystop,tensorboard])
    #autoencoder.fit(x_train,x_val,validation_split=0.15,epochs = config['epoch'], batch_size = config['batch_size'],callbacks= [checkpoint,earlystop,tensorboard])

    autoencoder.summary()
    return autoencoder

def buildReconstructionAutoEncoder(x_train,y_train,x_val,y_val,config):
    ##AutoEncoder w/ CNN
    autoencoder = Sequential()

    #Encoder part
    #autoencoder.add(Conv2D(128, (3, 3), activation='relu', padding='same', input_shape=x_train.shape[1:],trainable = config['trainable']))
    autoencoder.add(Conv2D(64, (3, 3), activation='relu', padding='same',trainable = config['trainable']))
    #autoencoder.add(Conv2D(64, (3, 3), activation='relu',padding='same',trainable = config['trainable']))
    autoencoder.add(BatchNormalization(axis=-1))
    autoencoder.add(MaxPooling2D((2, 2),trainable = config['trainable']))
    #autoencoder.add(Conv2D(32, (3, 3), activation='relu', padding='same',trainable = config['trainable']))
    autoencoder.add(Conv2D(32, (3, 3), activation='relu', padding='same',trainable = config['trainable']))
    autoencoder.add(BatchNormalization(axis=-1))
    #autoencoder.add(Conv2D(16, (3, 3), activation='relu',trainable = config['trainable']))
    autoencoder.add(MaxPooling2D((2, 2),trainable = config['trainable']))
    #autoencoder.add(Conv2D(16, (3, 3), activation='relu', padding='same',trainable = config['trainable']))
    autoencoder.add(Conv2D(16, (3, 3), activation='relu',padding = 'same',trainable = config['trainable']))
    #autoencoder.add(Conv2D(8, (3, 3), activation='relu',trainable = config['trainable']))
    autoencoder.add(MaxPooling2D((2, 2), name = 'encode_layer',trainable = config['trainable']))
    autoencoder.add(Dropout(0.25))

    #Decoder part
    autoencoder.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
    #autoencoder.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
    #autoencoder.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    #autoencoder.add(Conv2D(16, (3, 3), activation='relu'))
    autoencoder.add(UpSampling2D((2, 2)))
    autoencoder.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    #autoencoder.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    #autoencoder.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    #autoencoder.add(Conv2D(32, (3, 3), activation='relu'))
    autoencoder.add(UpSampling2D((2, 2)))
    autoencoder.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    #autoencoder.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    #autoencoder.add(Conv2D(128, (3, 3), activation='relu'))
    autoencoder.add(UpSampling2D((2, 2)))
    autoencoder.add(Dropout(0.25))
    autoencoder.add(Conv2D(3,(1,1),activation='relu',name = 'decode_layer'))
    #Train the autoencoder itself w/ output classification layer
    #autoencoder.add(Flatten())
    #autoencoder.add(Dense(32))
    #autoencoder.add(Activation('relu'))
    #autoencoder.add(Dropout(0.25))
    #autoencoder.add(Dense(y_train.shape[1],name= 'output_layer'))
    #autoencoder.add(Activation('softmax'))

    autoencoder.compile(optimizer = 'adam', loss = 'binary_crossentropy')

    checkpoint = ModelCheckpoint(filepath = config['filename'],monitor='val_loss'
                                ,verbose=1,save_best_only= True)
    earlystop = EarlyStopping(monitor='val_loss', min_delta=0,
                                patience=10, mode='auto', restore_best_weights=True)
    tensorboard = TensorBoard(log_dir=config['logdir'], histogram_freq=0,
                                write_graph=True, write_images=True)
    autoencoder.fit(x_train,y_train,validation_data =(x_val,y_val),epochs = config['epoch'], batch_size = config['batch_size'],callbacks= [checkpoint,earlystop,tensorboard])
    #autoencoder.fit(x_train,x_val,validation_split=0.15,epochs = config['epoch'], batch_size = config['batch_size'],callbacks= [checkpoint,earlystop,tensorboard])

    autoencoder.summary()
    return autoencoder

def conv2d_block(input,num_filters,kernel_size = 3,batchnorm = True):
    # first layer
    x = Conv2D(filters=num_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
    padding="same")(input)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    # second layer
    x = Conv2D(filters=num_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
    padding="same")(x)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x

def build_unet(input_img,num_filters = 16,dropout = 0.5,batchnorm = True):
    # contracting path
    c1 = conv2d_block(input_img, num_filters=num_filters*1, kernel_size=3, batchnorm=batchnorm)
    p1 = MaxPooling2D((2, 2)) (c1)
    p1 = Dropout(dropout*0.5)(p1)
    c2 = conv2d_block(p1, num_filters=num_filters*2, kernel_size=3, batchnorm=batchnorm)
    p2 = MaxPooling2D((2, 2)) (c2)
    p2 = Dropout(dropout)(p2)
    c3 = conv2d_block(p2, num_filters=num_filters*4, kernel_size=3, batchnorm=batchnorm)
    p3 = MaxPooling2D((2, 2)) (c3)
    p3 = Dropout(dropout)(p3)
    c4 = conv2d_block(p3, num_filters=num_filters*8, kernel_size=3, batchnorm=batchnorm)
    p4 = MaxPooling2D(pool_size=(2, 2)) (c4)
    p4 = Dropout(dropout)(p4)
    c5 = conv2d_block(p4, num_filters=num_filters*16, kernel_size=3, batchnorm=batchnorm)
    # expansive path
    u6 = Conv2DTranspose(num_filters*8, (3, 3), strides=(2, 2), padding='same') (c5)
    u6 = concatenate([u6, c4])
    u6 = Dropout(dropout)(u6)
    c6 = conv2d_block(u6, num_filters=num_filters*8, kernel_size=3, batchnorm=batchnorm)
    u7 = Conv2DTranspose(num_filters*4, (3, 3), strides=(2, 2), padding='same') (c6)
    u7 = concatenate([u7, c3])
    u7 = Dropout(dropout)(u7)
    c7 = conv2d_block(u7, num_filters=num_filters*4, kernel_size=3, batchnorm=batchnorm)
    u8 = Conv2DTranspose(num_filters*2, (3, 3), strides=(2, 2), padding='same') (c7)
    u8 = concatenate([u8, c2])
    u8 = Dropout(dropout)(u8)
    c8 = conv2d_block(u8, num_filters=num_filters*2, kernel_size=3, batchnorm=batchnorm)
    u9 = Conv2DTranspose(num_filters*1, (3, 3), strides=(2, 2), padding='same') (c8)
    u9 = concatenate([u9, c1], axis=3)
    u9 = Dropout(dropout)(u9)
    c9 = conv2d_block(u9, num_filters=num_filters*1, kernel_size=3, batchnorm=batchnorm)
    outputs = Conv2D(3, (1, 1), activation='sigmoid') (c9)
    #flat = Flatten()(outputs)
    #out_real = Dense(5,activation='softmax')(flat)
    model = Model(inputs=[input_img], outputs=[outputs])
    return model

def trainModel(x_train,y_train,x_val,y_val,config,input_img,num_filters,dropout = 0.05,batchnorm = True):
    autoencoder = build_unet(input_img, num_filters=num_filters, dropout=dropout, batchnorm=True)
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy', metrics=[dice_coef])
    autoencoder.summary()
    checkpoint = ModelCheckpoint(filepath = config['filename'],monitor='val_dice_coef'
                                ,verbose=1,save_best_only= True,mode = 'max')
    earlystop = EarlyStopping(monitor='val_dice_coef', min_delta=0,
                                patience=10, mode='max', restore_best_weights=True)
    tensorboard = TensorBoard(log_dir=config['logdir'], histogram_freq=0,
                                write_graph=True, write_images=True)
    results = autoencoder.fit(x_train,y_train,validation_data=(x_val,y_val),batch_size=config['batch_size'],epochs=config['epoch'],callbacks= [checkpoint,earlystop,tensorboard])
    return results,autoencoder

def dice_coef(y_true, y_pred):
    smooth = 1e-5

    y_true = tf.round(tf.reshape(y_true, [-1]))
    y_pred = tf.round(tf.reshape(y_pred, [-1]))

    isct = tf.reduce_sum(y_true * y_pred)

    return 2 * isct / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred))

keras.metrics.dice_coef = dice_coef

config = {
'trainable' : True,
'epoch' : 10,
'batch_size' : 256,
'filename':'D:/Sem 2/CV/Project/autoencoder-recon.h5',
'logdir': 'D:/Sem 2/CV/Project/logs/segmentation'
}

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Wistia):
 
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    
    
    

autoencoder_recon = buildReconstructionAutoEncoder(x_train,x_train,x_val,x_val,config)
config['filename'] = 'D:/Sem 2/CV/Project/autoencoder-scratchxxx.h5'
autoencoder_scratch = buildClassificationAutoEncoder(x_train,y_train,x_val,y_val,config)

yhat_probs = autoencoder_scratch.predict(x_val, verbose=0)
yhat_classes = autoencoder_scratch.predict_classes(x_val, verbose=0)

true_class = []
for imgs in range(len(y_val)):
    idx = np.where(y_val[imgs]==1)
    idx = idx[0][0]
    true_class.append(idx)

matrix = confusion_matrix(true_class, yhat_classes)

plot_confusion_matrix(matrix, classes=filter)
autoencoder_finetune = buildClassificationAutoEncoder(x_train,y_train,x_val,y_val,config)
results,autoencoder_segmentation = trainModel(x_seg_train,y_seg_train,x_seg_val,y_seg_val,config,Input((64,64,3)),num_filters = 16)
