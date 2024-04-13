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

@app.route('/', methods=["GET"])
@cross_origin()
def serve():
    return jsonify("Backend for Group 7's LesionCheckr App.")

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
    predictions = model.predict(img[None, ...])
    index = np.argmax(predictions)
    type_of_lesion = labels.inverse_transform(np.argmax(predictions, axis=1))

    #print(classes[index])
    return jsonify({'result': lesion_types[type_of_lesion[0]]})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)