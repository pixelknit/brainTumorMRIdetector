import tensorflow as tf
import cv2

CATEGORIES = ["yes","no"]

def prepare(filepath):
    IMG_SIZE = 100
    img_array = cv2.imread(filepath,cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))
    return new_array.reshape(-1,IMG_SIZE,IMG_SIZE,1)

model = tf.keras.models.load_model("brain.model")

prediction = model.predict([prepare("brain_no2.jpeg")])

print(CATEGORIES[int(prediction[0][0])])
