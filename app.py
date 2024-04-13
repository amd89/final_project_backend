from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
from PIL import Image
import base64
from io import BytesIO
import requests
import numpy as np
import cv2
import pickle
from keras.models import load_model
from keras.layers import Rescaling
import matplotlib.pyplot as plt
labels = pickle.load(open("LE.pkl", "rb"))
model = load_model("model.h5")
app = Flask(__name__)
CORS(app)

lesion_types = {
    'akiec': 'Actinic Keratoses',
    'bcc': 'Basal Cell Carcinoma',
    'bkl': 'Benign Keratosis',
    'df': 'Dermatofibroma',
    'mel': 'Melanoma',
    'nv': 'Melanocytic Nevi',
    'vasc': 'Vascular Lesions'
}

@app.route("/classify", methods=["POST"])
def image_check():
    print("Connected to the server.")
    print(request.get_json())
    url = request.get_json()['image']
    img = base64.b64decode(url)
    img = Image.open(BytesIO(img))
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    cv2.imwrite("test1.jpg", img)
    # Grabbing image data from base64 string or URL
    try:
        # Base64 DATA
        if "data:image/jpeg;base64," in url:
            base_string = url.replace("data:image/jpeg;base64,", "")
            decoded_img = base64.b64decode(base_string)
            img = Image.open(BytesIO(decoded_img))
            img = np.array(img)
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (224, 224))
            print(img.shape)

        # Base64 DATA
        elif "data:image/png;base64," in url:
            base_string = url.replace("data:image/png;base64,", "")
            decoded_img = base64.b64decode(base_string)
            img = Image.open(BytesIO(decoded_img))
            img = np.array(img)
            img = cv2.convertColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (224, 224))
            print(img.shape)

        # Regular URL Form DATA
        else:
            response = requests.get(url)
            img = Image.open(BytesIO(response.content)).convert("RGB")
            img = np.array(img)
            img = cv2.resize(img, (224, 224))
    
        print("model")
        print(model.predict(img/255))
    # ----- SECTION 3 -----    
        result = "Image has been succesfully sent to the server."
    except Exception as e:
        result = "Error: " + str(e)
    cv2.imwrite("test1.jpg", img)
    plt.imshow(img)
    normalization = Rescaling(1./255)
    #img = img/255
    plt.imshow(img)
    cv2.imwrite("test.jpg", img)
    predictions = model.predict(img[None, ...])
    index = np.argmax(predictions)
    print(predictions[0][index])
    print(predictions)
    print(len(predictions))
    print(index)
    type_of_lesion = labels.inverse_transform(np.argmax(predictions, axis=1))
    print(type_of_lesion)
    print(type_of_lesion[0])

    #print(classes[index])
    return jsonify({'result': lesion_types[type_of_lesion[0]]})