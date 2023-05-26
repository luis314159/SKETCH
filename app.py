from flask import Flask, render_template, request, redirect, url_for, jsonify
from PIL import Image
import tensorflow as tf
import io
import base64
import numpy as np


def procesar_prediccion(prediction):
    # Obtén el índice del valor más alto
    predicted_class = np.argmax(prediction)
    print(predicted_class)
    if predicted_class == 0:
        return "grisley"
    if predicted_class == 1:
        return "panda"
    if predicted_class == 2:
        return "polar"
   



app = Flask(__name__)
model_path = "./callback-001/callback"
# Carga el modelo de TensorFlow.
modelo = tf.keras.models.load_model(model_path)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Obtenemos la imagen del formulario
    data = request.form['imagen']
    # Quitamos la cabecera de la imagen en base64
    data = data.split(",")[1]
    # Convertimos la imagen en base64 a bytes
    decoded = base64.b64decode(data)
    image = Image.open(io.BytesIO(decoded))
    image = image.convert('RGB')  # Convertir a RGB
    # Reescalamos la imagen a 224x224
    image = image.resize((224, 224))

    # Convertimos la imagen a un array de numpy y normalizamos
    image = np.array(image) / 255.0
    # Agregamos una dimensión extra ya que el modelo lo espera
    image = np.expand_dims(image, axis=0)
    
    # Hacemos la predicción
    pred = modelo.predict(image)


    
    # Aquí puedes agregar una lógica para convertir el valor predicho a una cadena que representa la clase
    # Por ejemplo, si el modelo predice frutas, podrías tener un diccionario de {0: "manzana", 1: "banano", ...}

    pred = procesar_prediccion(pred)
    #fruta_predicha = "manzana"  # Esto es solo un ejemplo
    
    return render_template('index.html', prediccion=pred)

@app.route('/restart', methods=['POST'])
def restart():
    return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(debug=True)
