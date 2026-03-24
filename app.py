import streamlit as st
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input
from PIL import Image, ImageOps
import cv2

st.title('Clasificador MNIST de Dígitos')
st.write('Sube una imagen de un número manuscrito y la CNN predecirá qué dígito es.')

@st.cache_resource
def get_model():
    # 1. Recreamos la estructura exacta del modelo ganador a mano
    # ¡Esto evita por completo el bug de lectura de configuración de Keras!
    model = Sequential([
        Input(shape=(28, 28, 1)),
        Conv2D(32, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(10, activation='softmax')
    ])
    
    # 2. Le inyectamos únicamente los pesos (lo que aprendió en Colab)
    model.load_weights('mejor_modelo.keras')
    return model

model = get_model()

uploaded_file = st.file_uploader("Elige una imagen (png, jpg)...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('L')
    st.image(image, caption='Imagen subida', width=150)
    
    # Redimensionar a 28x28 y preparar
    img_array = np.array(image)
    img_array = cv2.resize(img_array, (28, 28))
    
    # MNIST usa fondo negro y números blancos. Si el fondo es blanco, se invierte
    if np.mean(img_array) > 127:
        img_array = 255 - img_array
        
    img_array = img_array.astype('float32') / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)
    
    # Predicción
    prediccion = model.predict(img_array)
    digito = np.argmax(prediccion)
    confianza = np.max(prediccion) * 100
    
    st.success(f'Predicción: El número es un {digito} (Confianza: {confianza:.2f}%)')
