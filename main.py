from fastapi import FastAPI, File
from pydantic import BaseModel
import numpy as np
import pickle
import warnings
import base64
from PIL import Image
import io

warnings.simplefilter(action='ignore',category=DeprecationWarning)

app = FastAPI()

class PredictionResponse(BaseModel):
    prediction: float

class ImageRequest(BaseModel):
    image: str

def load_model():
    global dec_tree_model
    with open("models/dec_tree.pkl","rb") as f:
        dec_tree_model = pickle.load(f)

@app.on_event("startup")
async def startup_event():
    load_model()

@app.post("/predict",response_model=PredictionResponse)
async def predict(request:ImageRequest):
    img_bytes = base64.b64decode(request.image)
    img = Image.open(io.BytesIO(img_bytes))
    img = img.resize((8,8))
    img_array = np.array(img)

    img_array = np.dot(img_array[...,:3],[0.2989,0.5870,0.1140])
    img_array = img_array.reshape(1,-1)

    prediction = dec_tree_model.predict(img_array)
    return{"prediction":prediction}

@app.get("/healthcheck")
async def healthcheck():
    return {"status": "ok"}