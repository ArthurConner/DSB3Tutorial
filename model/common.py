import numpy as np
import pandas as pd
import os
import bcolz
import itertools
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from datetime import datetime
from skimage import measure, feature
import json

def save_array(fname, arr): c=bcolz.carray(arr, rootdir=fname, mode='w'); c.flush()
def load_array(fname): return bcolz.open(fname)[:]


def plot_3d(image, threshold=-300):
    
    # Position the scan upright, 
    # so the head of the patient would be at the top facing the camera
    p = image.transpose(2,1,0)
    p = p[:,:,::-1]
    
    verts, faces = measure.marching_cubes(p, threshold)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], alpha=0.1)
    face_color = [0.5, 0.5, 1]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)

    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])

    plt.show()

def makeKaggeDataArray(images_path):
    patients = os.listdir(images_path)
    patients.sort()

    labels_csv = pd.read_csv('/data1/cancer/KAG2017/stage1_labels.csv', index_col='id')

    #get pats from labels
    lookup = {}
    for x in patients:
        lookup[x] = -1

    for index, row in labels_csv.iterrows():
        if index in  lookup :
            lookup[index] = row["cancer"]

    #find classes
    totals = {-1:0,0:0,1:0}
    bad = []
    good = []

    for k , v in lookup.items():
        totals[v] = totals[v] + 1
        if v < 0:
            bad.append(k)
        else:
            good.append(k)

    good.sort()

    y = []
    x = []

    for patid in good:
        y.append(lookup[patid])
        path = os.path.join(images_path,patid,"")
        getImage = load_array(path)
        x.append(getImage)
        
    y = np.array(y)
    x = np.array(x)

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
    return X_train, X_test, y_train, y_test

class TrainSet():
    """The Vg3 16 Imagenet model"""


    def __init__(self,images_path,is3D=True):

        X_train, X_test , y_train, y_test = makeKaggeDataArray(images_path)
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.is3D = is3D
        if is3D:
            self.X_train3d = self.colorPad(X_train)
            self.X_test3d = self.colorPad(X_test)

    def trainX(self):
        if self.is3D:
           return self.X_train3d
        return self.X_train

    def testX(self):
        if self.is3D:
           return self.X_test3d
        return self.X_test




    def process_image(self,img):
        #img = scipy.ndimage.zoom(img.astype(np.float), 0.25)
        img_std = np.std(img)
        img_avg = np.average(img)
        return np.clip((img - img_avg + img_std) / (img_std * 2), 0, 1).astype(np.float16)

    def colorPad(self,patients):
        hort = patients[0].shape[0]
        vert = patients[0].shape[1]
        zdir = patients[0].shape[2]
        
        train_features = np.zeros([len(patients), 1, hort, vert, zdir], np.float16)
        for i in range(len(patients)):
            f = patients[i]
            f = self.process_image(f)
            f = np.concatenate([f, np.zeros([hort - f.shape[0], vert, zdir], np.float16)]) # Pads the image
            f = f.reshape([1, hort, vert, zdir]) # add an extra dimension for the color channel
            train_features[i] = f
            
        return train_features

   




def plot_confusion_matrix(y_train, y_preds, classes=["No","Yes"],
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    cm = confusion_matrix(y_train, y_preds)

    #good = cm[0] + cm[2]
    #total = good + cm[1] + cm[3]
    print("we have accuracy " + str( accuracy_score(y_train, y_preds)))
    #print("we have " + str(good) + " out of " + str(total) + ".")
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title )
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    #print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')




def evaluate(vg,ts,label="model",nb_epoch=1):
    model = vg.model

    train = ts.trainX()
    test = ts.testX()

    model.fit(train, ts.y_train, batch_size=32, validation_split=0.1, nb_epoch=nb_epoch)
    
    y_predProb = model.predict(test)
    ts.y_predProb = y_predProb
    y_pred =  np.rint(y_predProb)

    cm = confusion_matrix(ts.y_test, y_pred)

    runname = label + "." + datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    save_array("./runs/" + runname + ".confusion",cm)
    model.save("./runs/" + runname + ".model") 
    

    with open("./runs" + "/" + runname + ".meta", 'w') as f:
     json.dump(vg.meta, f)

        # save as JSON
    json_string = model.to_json()
    with open("./runs" + "/" + runname + ".model", 'w') as f:
        f.write(json_string)
   

    # save as YAML
    yaml_string = model.to_yaml()

    with open("./runs" + "/" + runname + ".yaml", 'w') as f:
        f.write(yaml_string)

   
    print("ran " + str(vg.meta))


    plot_confusion_matrix(ts.y_test, np.rint(y_pred), title='Confusion matrix, without normalization')
   
    y_predProb = model.predict(train)
    y_pred =  np.rint(y_predProb)
    plot_confusion_matrix(ts.y_train, np.rint(y_pred), title='Again Training Set')

