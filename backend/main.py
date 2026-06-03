from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import os

# Forzar el backend de TensorFlow para Keras
os.environ["KERAS_BACKEND"] = "tensorflow"

import keras
import numpy as np
from PIL import Image
import io

app = FastAPI()

# Permitir conexiones de la página web (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- RECONSTRUCCIÓN DE LA ARQUITECTURA E INYECCIÓN DE PESOS ---
MODEL_PATH = "STREET_DEAL_PRO_LIMPIO.h5"

print("Instanciando arquitectura base limpia de MobileNetV2...")
base_model = keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights=None
)

# Acoplamos el bloque extractor exacto de tu Sequential
x = keras.layers.GlobalAveragePooling2D()(base_model.output)
x = keras.layers.Dropout(0.4)(x)
feature_extractor = keras.models.Model(inputs=base_model.input, outputs=x, name="mobilenetv2")

# Tu modelo Sequential compacto de 2 bloques tal cual se guardó en el H5
model = keras.models.Sequential([
    feature_extractor,
    keras.layers.Dense(24, activation="softmax")
])

# Construir dimensiones antes de inyectar
model.build((None, 224, 224, 3))

print("Cargando pesos de STREET_DEAL_PRO_LIMPIO.h5...")
model.load_weights(MODEL_PATH)
print("¡SISTEMA STREET DEAL PRO PREPARADO EXITOSAMENTE!")

# --- LISTA OFICIAL EN EL ORDEN EXACTO DE TU DATASET ---
CLASSES = [
    'Audi', 'Benz', 'Bmw', 'Cadillac', 'Dodge', 'Ferrari', 'Ford', 
    'Ford mustang', 'GMC', 'Jeep', 'Kia', 'Lamborghini', 'Maserati', 
    'Mitsubishi', 'Nissan', 'Porsche', 'RAM', 'Rolls royce', 'Tesla', 
    'Toyota', 'alfa romeo', 'hyundai', 'mercedes', 'volkswagen'
]

@app.get("/")
def home():
    return {"status": "Backend de STREET_DEAL corriendo al 100% con normalización fija"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # 1. Leer la imagen enviada desde el navegador
    request_object_content = await file.read()
    image = Image.open(io.BytesIO(request_object_content)).convert("RGB")
    
    # 2. Redimensionar al tamaño estándar del entrenamiento (224x224)
    image = image.resize((224, 224))
    
    # 3. Convertir a array numérico y aplicar la normalización EXACTA de tu Colab (0 a 1)
    input_array = np.array(image, dtype=np.float32) / 255.0
    
    # 4. Expandir dimensiones para el formato de lote: [1, 224, 224, 3]
    input_array = np.expand_dims(input_array, axis=0)
    
    # 5. Ejecutar la inferencia con la IA
    predictions = model.predict(input_array)
    best_match_index = np.argmax(predictions[0])
    confidence = float(predictions[0][best_match_index])
    
    return {
        "class_index": int(best_match_index),
        "label": CLASSES[best_match_index],
        "confidence": confidence
    }

if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)