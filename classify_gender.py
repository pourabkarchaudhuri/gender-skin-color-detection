# import necessary packages
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
# import argparse
import cv2
import os

MODEL_DIR = os.path.join(os.getcwd(), 'models', 'gender_classification.model')
IMAGE_PATH = os.path.join(os.getcwd(), 'uploads', 'sample_input1.jpg')
# read input image
image = cv2.imread(IMAGE_PATH)

if image is None:
    print("Could not read input image")
    exit()

# preprocessing
output = np.copy(image)
image = cv2.resize(image, (96,96))
image = image.astype("float") / 255.0
image = img_to_array(image)
image = np.expand_dims(image, axis=0)

# load pre-trained model
model = load_model(MODEL_DIR)

# run inference on input image
confidence = model.predict(image)[0]

# write predicted gender and confidence on image (top-left corner)
classes = ["man", "woman"]    
idx = np.argmax(confidence)
label = classes[idx]
# label = "{}: {:.2f}%".format(label, confidence[idx] * 100)

print("Label : {}".format(label))
print("COnfidence : {:.2f}".format(confidence[idx] * 100))
