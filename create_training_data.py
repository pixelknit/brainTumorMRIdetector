import cv2
import pickle
import numpy as np
from tqdm import tqdm
import random
import os

DATADIR = r"C:\Users\gabriel\Documents\development\datasets\archive\brain_tumor_dataset"
CATEGORIES = ["yes","no"]
IMG_SIZE = 100

training_data = []

def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR,category)
        class_num = CATEGORIES.index(category)

        for img in tqdm(os.listdir(path)):
            try:
                img_array = cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))
                training_data.append([new_array, class_num])
            except Exception as e:
                pass

create_training_data()
print(len(training_data))

random.shuffle(training_data)

X = []
y = []

for features, label in training_data:
    X.append(features)
    y.append(label)

X = np.array(X).reshape(-1,IMG_SIZE, IMG_SIZE,1)
y = np.array(y)

with open("X.pickle","wb") as xp:
    pickle.dump(X,xp)
with open("y.pickle","wb") as yp:
    pickle.dump(y,yp)
