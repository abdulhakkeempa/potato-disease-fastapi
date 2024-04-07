from fastapi import FastAPI, UploadFile
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import io

app = FastAPI()

model = load_model('./model/model.h5')
model.load_weights('./model/easy_checkpoint')

CLASSES = ['Potato___Bacteria', 'Potato___Early_blight', 'Potato___Fungi', 'Potato___Late_blight', 'Potato___Pest', 'Potato___healthy']

IMAGE_SIZE = 200
CHANNELS = 3

def predict(image):
    image = image / 255.0
    image = np.resize(image, (IMAGE_SIZE, IMAGE_SIZE, CHANNELS))
    image = np.expand_dims(image, axis=0)
    preds = model.predict(image)
    print(preds)
    class_idx = np.argmax(preds[0])
    return CLASSES[class_idx]

@app.post("/predict")
async def predict_image(file: UploadFile):
    image = Image.open(io.BytesIO(await file.read()))
    image = img_to_array(image)

    class_label = predict(image)

    return {"class_label": class_label}
