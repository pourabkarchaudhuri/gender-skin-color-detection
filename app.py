# import classify_gender
import detect_skintone
from flask import Flask, request, jsonify
import os
from werkzeug.utils import secure_filename
import time
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import mtcnn
import tensorflow as tf
# import argparse
import cv2
import base64
os.environ["CUDA_VISIBLE_DEVICES"]="-1"



MODEL_DIR = os.path.join(os.getcwd(), 'models', 'gender_classification.model')
TEST_PATH = os.path.join(os.getcwd(), 'uploads', 'sample_input1.jpg')

dirname = os.path.dirname(__file__)

UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')

image = cv2.imread(TEST_PATH)
output = np.copy(image)
image = cv2.resize(image, (96,96))
image = image.astype("float") / 255.0
image = img_to_array(image)
image = np.expand_dims(image, axis=0)

model = load_model(MODEL_DIR)
classes = ["male", "female"] 
confidence = model.predict(image)[0]

   
idx = np.argmax(confidence)
label = classes[idx]

minsize = 25 # minimum size of face
threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
factor = 0.709 # scale factor

# Flask App Deifinition

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = set(['jpeg', 'jpg', 'png'])
current_milli_time = lambda: int(round(time.time() * 1000))
@app.route('/detect', methods=['POST'])
def process_image():
    count = 0
    stamp = str(current_milli_time())
    global model, classes, minsize, threshold, factor
    if not request.headers.get('Content-type') is None:
        if(request.headers.get('Content-type').split(';')[0] == 'multipart/form-data'):
            if 'image' in request.files.keys():
                print("Form Data, Multipart upload")
                file = request.files['image']
                # if user does not select file, browser also
                # submit a empty part without filename
                if file.filename == '':
                    result = {
                        "error":True,
                        "message": "No Image Recieved",
                        "gender":None,
                        "skin": None
                    }
                    return jsonify(result)
                filename = secure_filename(file.filename)
                # file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], "photo_"+stamp+".jpg"))

            else:
                return jsonify(get_status_code("Invalid body", "Please provide valid format for Image 2")), 415
        elif(request.headers.get('Content-type') == 'application/json'):
            if(request.data == b''):
                return jsonify(get_status_code("Invalid body", "Please provide valid format for Image")), 415
            else:
                print("Application/Json upload as base64")
                body = request.get_json()
                if 'image_string' in body.keys():
                    img_string = body['image_string']
                    try:
                        str_image = img_string.split(',')[1]
                        imgdata = base64.b64decode(str_image)
                    
                        with open(os.path.join(UPLOAD_FOLDER, "photo_"+stamp+".jpg"), 'wb') as f:
                            f.write(imgdata)
                        
                    except IndexError:
                        result = {
                            "error":True,
                            "message": "Invalid base64 string",
                            "gender":None,
                            "skin": None
                        }
                        return jsonify(result)
                    
                    
                    
                else:
                    result = {
                        "error":True,
                        "message": "Put 'image_string' as key in input payload",
                        "gender":None,
                        "skin": None
                    }
                    return jsonify(result)

        else:
            return jsonify(get_status_code("Invalid header", "Please provide correct header with correct data")), 415
    
    else:
        return jsonify(get_status_code("Invalid Header", "Please provide valid header")), 401



    print("File to be processed : {}".format(os.path.join(UPLOAD_FOLDER, "photo_"+stamp+".jpg")))
    root_image = cv2.imread(os.path.join(UPLOAD_FOLDER, "photo_"+stamp+".jpg"))
    
    if root_image is None:
        print("Could not read input image")
    
    # output = np.copy(image)
    image = cv2.resize(root_image, (96,96))
    image = image.astype("float") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)

    # run inference on input image
    confidence = model.predict(image)[0]
   
    idx = np.argmax(confidence)
    label = classes[idx]
    # label = "{}: {:.2f}%".format(label, confidence[idx] * 100)
    face_count = 0
    # print("Label : {}".format(label))
    # print("Confidence : {:.2f}".format(confidence[idx] * 100))
    sess = tf.Session()
    with sess.as_default():
        pnet, rnet, onet = mtcnn.create_mtcnn(sess, None)
        # img = cv2.imread(os.path.join(os.getcwd(), 'uploads', 'sample_input1.jpg'))
        boxes, _ = mtcnn.detect_face(root_image, minsize, pnet, rnet, onet, threshold, factor)
        for i in range(boxes.shape[0]):
            # pt1 = (int(boxes[i][0]), int(boxes[i][1]))
            # pt2 = (int(boxes[i][2]), int(boxes[i][3]))
            x = int(boxes[i][0])
            y = int(boxes[i][1])
            w = int(boxes[i][2])
            h = int(boxes[i][3])
            p1 = int(boxes[0][2])
            p2 = int(boxes[0][3])
            
            if(float(boxes[i][4]) >= 0.95):
                sub_faces = root_image[y:h, x:w]
                count = count + 1
                cv2.imwrite(os.path.join(os.getcwd(), 'uploads', "photo_"+stamp+".jpg"), sub_faces)
            else:
                print("No faces detected")
                count = 0
    
    print("Count : ".format(count))
    if(count == 0):
        result = {
            "error":True,
            "message": "No faces found",
            "gender":None,
            "skin": None
        }
    elif(count>1):
        result = {
            "error":True,
            "message": "Too many faces detected",
            "gender":None,
            "skin": None
        }
    else:
        skin_tone = detect_skintone.get_skin_tone(root_image)

        result = {
            "error":False,
            "message":None,
            "gender":{
                "type":label,
                "confidence":confidence[idx] * 100
            },
            "skin": skin_tone
        }
    
    return jsonify(result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3002)

