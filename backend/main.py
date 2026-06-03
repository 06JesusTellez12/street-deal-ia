from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

import os
os.environ["KERAS_BACKEND"] = "tensorflow"

import keras
import numpy as np
from PIL import Image
import io

app = FastAPI()

# ==========================
# CORS
# ==========================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================
# FRONTEND
# ==========================
app.mount(
    "/static",
    StaticFiles(directory="/app/frontend"),
    name="static"
)

# ==========================
# MODELO IA
# ==========================
MODEL_PATH = "STREET_DEAL_PRO_LIMPIO.h5"

print("Instanciando arquitectura base limpia de MobileNetV2...")

base_model = keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights=None
)

x = keras.layers.GlobalAveragePooling2D()(base_model.output)
x = keras.layers.Dropout(0.4)(x)

feature_extractor = keras.models.Model(
    inputs=base_model.input,
    outputs=x,
    name="mobilenetv2"
)

model = keras.models.Sequential([
    feature_extractor,
    keras.layers.Dense(24, activation="softmax")
])

model.build((None, 224, 224, 3))

print("Cargando pesos...")
model.load_weights(MODEL_PATH)

print("STREET DEAL PRO LISTO")

# ==========================
# CLASES
# ==========================
CLASSES = [
    'Audi',
    'Benz',
    'Bmw',
    'Cadillac',
    'Dodge',
    'Ferrari',
    'Ford',
    'Ford mustang',
    'GMC',
    'Jeep',
    'Kia',
    'Lamborghini',
    'Maserati',
    'Mitsubishi',
    'Nissan',
    'Porsche',
    'RAM',
    'Rolls royce',
    'Tesla',
    'Toyota',
    'alfa romeo',
    'hyundai',
    'mercedes',
    'volkswagen'
]

# ==========================
# PAGINA PRINCIPAL
# ==========================
@app.get("/")
async def home():
    return FileResponse("/app/frontend/index.html")

# ==========================
# PREDICCION
# ==========================
@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    content = await file.read()

    image = Image.open(
        io.BytesIO(content)
    ).convert("RGB")

    image = image.resize((224, 224))

    input_array = (
        np.array(image, dtype=np.float32)
        / 255.0
    )

    input_array = np.expand_dims(
        input_array,
        axis=0
    )

    predictions = model.predict(input_array)

    best_match_index = np.argmax(predictions[0])

    confidence = float(
        predictions[0][best_match_index]
    )

    return {
        "class_index": int(best_match_index),
        "label": CLASSES[best_match_index],
        "confidence": confidence
    }

if __name__ == "__main__":
    import uvicorn

    port = int(
        os.environ.get("PORT", 10000)
    )

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port
    )