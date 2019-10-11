import tensorflow as tf
import matplotlib.pyplot as plt
#matplotlib inline
from tqdm import tqdm
import numpy as np
import os 
from random import shuffle
import cv2
import tflearn
# conv_2d: CNN model
# max_pool_2d: reduce overfitting
from tflearn.layers.conv import conv_2d,max_pool_2d
# dropout: randomly switches ff neurons during training (improves accuracy)
# fully_connected: allows use of softmax activation function
from tflearn.layers.core import input_data,dropout,fully_connected
from tflearn.layers.estimator import regression
import tensorflow as tf
import matplotlib.pyplot as plt

training_accuracies = []
validation_accuracies = []
test_accuracies = []

# tflearn specific callback format
class accuracyCallback(tflearn.callbacks.Callback):
    def on_epoch_end(self, training_state):
        print(training_state.acc_value,training_state.val_acc,training_state.best_accuracy)
        training_accuracies.append(training_state.acc_value)
        validation_accuracies.append(training_state.val_acc)

#def label_image(img):
#    # use first three letters of filename to classify data
#    img_name = img.split("_")[0]
#   print(img,' -> ', img_name)
#    if img_name == "Prx1":
#        return [1,0]
#    elif img_name == "PrxQ":
#        return [0,1]

def training_data_loader(path_arg, class_arg):
    training_data = []
    if (class_arg == 0):
        img_label = [1,0]
    else:
        img_label = [0,1]
    for img in tqdm(os.listdir(path=path_arg)):
        path_to_img = os.path.join(path_arg,img)
        img = cv2.resize(cv2.imread(path_to_img,cv2.IMREAD_COLOR),(IMAGE_SIZE,IMAGE_SIZE))
        training_data.append([np.array(img),np.array(img_label)])
    shuffle(training_data) # randomize training data order
    #np.save("training_data_new.npy",training_data) #
    return training_data

def testing_data_loader(path_arg, class_arg):
    test_data = []
    if (class_arg == 0):
        img_label = [1,0]
    else:
        img_label = [0,1]
    for img in tqdm(os.listdir(path=path_arg)):
        path_to_img = os.path.join(path_arg,img)
        img = cv2.resize(cv2.imread(path_to_img,cv2.IMREAD_COLOR),(IMAGE_SIZE,IMAGE_SIZE))
        test_data.append([np.array(img),np.array(img_label)])
    #shuffle(test_data) # randomize test data order
    #np.save("test_dataone.npy",test_data)
    return test_data


#TRAIN_DIR = 'train'
#TEST_DIR = 'test
CLASS0_TRAIN_DIR = 'train/Prx1'
CLASS1_TRAIN_DIR = 'train/PrxQ'
CLASS0_TEST_DIR = 'test/Prx1'
CLASS1_TEST_DIR = 'test/PrxQ'


MAX_EPOCHS = 150
LEARNING_RATE = 1e-3
# save model data so you don't have to re-run it.
MODEL_NAME = "Prx1PrxQ_{}_{}.model".format(LEARNING_RATE,"6conv-fire")

IMAGE_SIZE = 96

# load the training data
class0_train_data_g = training_data_loader(CLASS0_TRAIN_DIR,0)
class1_train_data_g = training_data_loader(CLASS1_TRAIN_DIR,1)
#train_data_g = np.load('training_data_new.npy')

tf.reset_default_graph()

# input function reads in the data
convnet = input_data(shape=[None, IMAGE_SIZE, IMAGE_SIZE, 3], name='input')

# convnet layer 1
convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

# convnet layer 2
convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

# convnet layer 3
convnet = conv_2d(convnet, 128, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

# convnet layer 4
convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

# convnet layer 5
convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

# convnet layer 6: the purpose of this layer is just to prepare for softmax
convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.8) # randomly remove neurons

# fully connected layer with softmax as activation function
convnet = fully_connected(convnet, 2, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=LEARNING_RATE, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet, tensorboard_dir='log')

class0_train = class0_train_data_g[:-20]
class0_validation = class0_train_data_g[-20:]
class1_train = class1_train_data_g[:-20]
class1_validation = class1_train_data_g[-20:]

train = class0_train + class1_train
validation = class0_validation + class1_validation

np.save("validation_data_new",validation)

# This is our training data
X = np.array([i[0] for i in train]).reshape(-1,IMAGE_SIZE,IMAGE_SIZE,3)
Y = [i[1] for i in train]
# This is our validation data
validation_x = np.array([i[0] for i in validation]).reshape(-1,IMAGE_SIZE,IMAGE_SIZE,3)
validation_y = [i[1] for i in validation]

class0_test_data = testing_data_loader(CLASS0_TEST_DIR,0)
class1_test_data = testing_data_loader(CLASS1_TEST_DIR,1)
test_data = class0_test_data + class1_test_data
#test_data = np.load("test_dataone.npy")

#model.fit(X, Y, n_epoch=MAX_EPOCHS, validation_set=(validation_x,  validation_y), snapshot_step=20, show_metric=True, run_id=MODEL_NAME, callbacks=accuracyCallback())
model.fit(X, Y, n_epoch=MAX_EPOCHS, validation_set=(validation_x,  validation_y), show_metric=True, run_id=MODEL_NAME, callbacks=accuracyCallback())

model.save(MODEL_NAME)

print(training_accuracies)
print(validation_accuracies)


epochValues = range(MAX_EPOCHS)
plt.plot(epochValues, training_accuracies, color='black', label='training')
plt.plot(epochValues, validation_accuracies, color='blue', label='validation')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.show()

#figs = plt.figure()
#for num,data in enumerate(test_data[:40]):
#    test_img = data[0]
#    test_label = data[1]
#    test_img_feed = test_img.reshape(IMAGE_SIZE,IMAGE_SIZE,3)
#    t = figs.add_subplot(8,5,num+1)
#    ores = test_img
#    model_pred = model.predict([test_img_feed])[0]
#    if np.argmax(model_pred) == 1:
#        pred_val = "PrxQ"
#    else:
#        pred_val = "Prx1"
#        
#    t.imshow(ores,cmap="gray")
#    plt.title("" + str(test_label) + " " +  pred_val)
#plt.show()

