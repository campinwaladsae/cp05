
import mahotas as mh
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import image as mh
from glob import glob
import os
import numpy as np
import pandas as pd
from joblib import load


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

#Classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay





IMM_SIZE = 50

def get_data(data_dir):
    # get a list of the file paths to all the images in the data_dir
    image_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(('.jpg', '.jpeg', '.png', '.gif'))]
    
    # sort the list of file paths to make sure they are processed in a consistent order
    image_paths.sort()
    
    # read the images and resize them
    images = []
    for im_path in image_paths:
        image = mh.imread(im_path)
        if len(image.shape) > 2:
            image = mh.resize_to(image, [IMM_SIZE, IMM_SIZE, image.shape[2]])
        else:
            image = mh.resize_to(image, [IMM_SIZE, IMM_SIZE])
        if len(image.shape) > 2:
            image = mh.colors.rgb2grey(image[:,:,:3], dtype = np.uint8)
        images.append(image)
    
    # plot the images
    plt.figure(figsize=(20,20))
    for im_n, im in enumerate(images):
        plt.subplot(int(len(images)/5)+1,5,im_n+1)
        plt.imshow(im)
    plt.show()
    
    # create the dataset
    data = []
    for i in range(len(images)):
        data.append([images[i], 0])
    
    return np.array(data)

d = "C:/Users/Campin Waladsae A/Downloads/cp05-main/Covid19-dataset/train"
train = get_data(d)

d = "C:/Users/Campin Waladsae A/Downloads/cp05-main/Covid19-dataset/test"
val = get_data(d)

print("Train shape", train.shape) # Size of the training DataSet
print("Test shape", val.shape) # Size of the test DataSet
print("Image size", train[0][0].shape) # Size of image



l = []
for i in train:
    l.append(i[1])
sns.set_style('darkgrid')
sns.countplot(l)



plt.figure(figsize = (5,5))
plt.imshow(train[np.where(train[:,1] == 'Viral Pneumonia')[0][0]][0])
plt.title('Viral Pneumonia')

plt.figure(figsize = (5,5))
plt.imshow(train[np.where(train[:,1] == 'Covid')[0][0]][0])
plt.title('Covid')



def create_features(data):
    features = []
    labels = []
    for image, label in data:
        features.append(mh.features.haralick(image).ravel())
        labels.append(label)
    features = np.array(features)
    labels = np.array(labels)
    return (features, labels)

features_train, labels_train = create_features(train)
features_test, labels_test = create_features(val)



from sklearn.metrics import plot_confusion_matrix

clf = Pipeline([('preproc', StandardScaler()), ('classifier', LogisticRegression())])
clf.fit(features_train, labels_train)
scores_train = clf.score(features_train, labels_train)
scores_test = clf.score(features_test, labels_test)
print('Training DataSet accuracy: {: .1%}'.format(scores_train), 'Test DataSet accuracy: {: .1%}'.format(scores_test))
plot_confusion_matrix(clf, features_test, labels_test)
plt.show()



names = ["Logistic Regression", "Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA"]

classifiers = [
    LogisticRegression(),
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1, max_iter=1000),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis()]
scores_train = []
scores_test = []
for name, clf in zip(names, classifiers):
    clf = Pipeline([('preproc', StandardScaler()), ('classifier', clf)])
    clf.fit(features_train, labels_train)
    score_train = clf.score(features_train, labels_train)
    score_test = clf.score(features_test, labels_test)
    scores_train.append(score_train)
    scores_test.append(score_test)



res = pd.DataFrame(index = names)
res['scores_train'] = scores_train
res['scores_test'] = scores_test
res.columns = ['Test','Train']
res.index.name = "Classifier accuracy"
pd.options.display.float_format = '{:,.2f}'.format
print(res)



x = np.arange(len(names))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, scores_train, width, label='Train')
rects2 = ax.bar(x + width/2, scores_test, width, label='Test')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Accuracy')
ax.set_title('Accuracy of classifiers')
ax.set_xticks(x)
plt.xticks(rotation = 90)
ax.set_xticklabels(names)
ax.legend()

fig.tight_layout()

plt.show()



import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D , MaxPool2D , Flatten , Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
import tensorflow as tf
from sklearn.metrics import classification_report,confusion_matrix



x_train = []
y_train = []
x_val = []
y_val = []

for feature, label in train:
    x_train.append(feature)
    y_train.append(label)

for feature, label in val:
    x_val.append(feature)
    y_val.append(label)

# Normalize the data
x_train = np.array(x_train) / 255
x_val = np.array(x_val) / 255

# Reshaping input images
x_train = x_train.reshape(-1, IMM_SIZE, IMM_SIZE, 1)
x_val = x_val.reshape(-1, IMM_SIZE, IMM_SIZE, 1)

# Creating a dictionary of clases
lab = {}
for i, l in enumerate(set(y_train)):
    lab[l] = i


y_train = np.array([lab[l] for l in y_train])
y_val = np.array([lab[l] for l in y_val])

print("Shape of the input DataSet:", x_train.shape)
print("Shape of the output DataSet:", y_train.shape)
print("Dictionary of classes:", lab)



datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range = 30,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.2, # Randomly zoom image
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip = True,  # randomly flip images
        vertical_flip=False)  # randomly flip images


datagen.fit(x_train)

"""## Model defining

Let’s define a simple CNN model with 3 Convolutional layers followed by max-pooling layers. A dropout layer is added after the 3rd maxpool operation to avoid overfitting.
"""

model = Sequential()
model.add(Conv2D(32,1,padding="same", activation="relu", input_shape=(IMM_SIZE,IMM_SIZE,1)))
model.add(MaxPool2D())

model.add(Conv2D(32, 1, padding="same", activation="relu"))
model.add(MaxPool2D())

model.add(Conv2D(64, 1, padding="same", activation="relu"))
model.add(MaxPool2D())
model.add(Dropout(0.4))

model.add(Flatten())
model.add(Dense(128,activation="relu"))
model.add(Dense(3, activation="softmax"))

model.summary()

"""Let’s compile the model now using Adam as our optimizer and SparseCategoricalCrossentropy as the loss function. We are using a lower learning rate of 0.000001 for a smoother curve.

"""

opt = Adam(lr=0.000001)
model.compile(optimizer = opt , loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True) , metrics = ['accuracy'])

"""Now, let’s train our model for **2000** epochs.
Admittedly, the fitting process is very slow. Therefore, we saved the fitted model to a file.
To save time, we will upload the fitted model.
If you would like, you can change the parameter **fitting to True** in order to refit the model.

"""

fitting = True
fitting_save = True
epochs = 5

import pickle

if fitting:
    history = model.fit(x_train,y_train,epochs = epochs , validation_data = (x_val, y_val), shuffle = True)
    if fitting_save:
    # serialize model to JSON
        model_json = model.to_json()
        with open("model.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights("model.h5")
        print("Saved model to disk")
        with open('history.pickle', 'wb') as f:
            pickle.dump(history.history, f)
        with open('lab.pickle', 'wb') as f:
            pickle.dump(lab, f)
# load model
from keras.models import model_from_json
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into a new model
model.load_weights("model.h5")
with open('history.pickle', 'rb') as f:
    history = pickle.load(f)
print("Loaded model from disk")

"""## Results evaluation

We will plot our training and validation accuracy along with the training and validation loss.
"""

acc = history['accuracy']
val_acc = history['val_accuracy']
loss = history['loss']
val_loss = history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(15, 15))
plt.subplot(2, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(2, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

"""Let’s see what the curve looks like.
You can see that the accuracy of the training and validation sets is the same. The loss function of validation and training sets is stable. It means that our CNN is fitted well and can be used for classification.

We can print out the classification report to see the precision and accuracy using [model.predict_classes()]() and [classification_report()]().

Also we can create a confusion matrix. Unfortunately, Keras framework does not have plot_confusion_matrix() function. Therefore, we have to create it using Pandas and [Seaborn.heatmap()]()

"""

# Classification report
#predictions = model.predict_classes(x_val)
predict_x=model.predict(x_val)
predictions=np.argmax(predict_x,axis=1)
predictions = predictions.reshape(1,-1)[0]
print(classification_report(y_val, predictions, target_names = lab.keys()))

# Confusion matrix
cm = pd.DataFrame(confusion_matrix(y_val, predictions))
cm.index = ["Predicted " + s for s in lab.keys()]
cm.columns = ["True  " + s for s in lab.keys()]
print(cm)

sns.heatmap(confusion_matrix(y_val, predictions), annot=True,
            xticklabels = list(lab.keys()), yticklabels = list(lab.keys()))
plt.xlabel("True labels")
plt.ylabel("Predicted labels")
plt.show()

# Accuracy
predict_x=model.predict(x_train)
predictions=np.argmax(predict_x,axis=1)
z = predictions == y_train
scores_train = sum(z+0)/len(z)
predict_x=model.predict(x_val)
predictions=np.argmax(predict_x,axis=1)

z = predictions == y_val
scores_test = sum(z+0)/len(z)
print('Training DataSet accuracy: {: .1%}'.format(scores_train), 'Test DataSet accuracy: {: .1%}'.format(scores_test))

"""As you can see, the CNN shows better results than classical models. However, fitting takes much longer.

And now, on your own, try to create a function that will establish a diagnosis based on the CNN.
**Project Tasks:**
"""

def diagnosis(file):
    # Download image
    ##YOUR CODE GOES HERE##
    try:
        image = mh.imread(file)
    except:
        print("Cannot download image: ", file)
        return

    # Prepare image to classification
    ##YOUR CODE GOES HERE##
    if len(image.shape) > 2:
        image = mh.resize_to(image, [IMM_SIZE, IMM_SIZE, image.shape[2]]) #resize of images RGB and png
    else:
        image = mh.resize_to(image, [IMM_SIZE, IMM_SIZE]) #resize of grey images
    if len(image.shape) > 2:
        image = mh.colors.rgb2grey(image[:,:,:3], dtype = np.uint8)  #change of colormap of images alpha chanel delete

    # Show image
    ##YOUR CODE GOES HERE##
    plt.gray()
    plt.imshow(image)
    plt.show()


    # Load model
    ##YOUR CODE GOES HERE##
    from keras.models import model_from_json
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights("model.h5")
    with open('history.pickle', 'rb') as f:
        history = pickle.load(f)
    with open('lab.pickle', 'rb') as f:
        lab = pickle.load(f)


    # Normalize the data
    ##YOUR CODE GOES HERE##
    image = np.array(image) / 255

    # Reshape input images
    ##YOUR CODE GOES HERE##
    image = image.reshape(-1, IMM_SIZE, IMM_SIZE, 1)


    # Predict the diagnosis
    ##YOUR CODE GOES HERE##
    diag = model.predict_classes(image)


    # Find the name of the diagnosis
    ##YOUR CODE GOES HERE##
    diag =list(lab.keys())[list(lab.values()).index(diag[0])]


    return diag

print ("Diagnosis is:", diagnosis("Covid19-dataset/test/Covid/0120.jpg"))
print ("Diagnosis is:", diagnosis("Covid19-dataset/test/Normal/0105.jpeg"))
print ("Diagnosis is:", diagnosis("Covid19-dataset/test/Viral Pneumonia/0111.jpeg"))

