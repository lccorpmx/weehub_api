from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
CORS(app) 

# Ruta al archivo CSV en Google Drive
file_path = 'dataset.csv'

# Cargar el archivo CSV en un DataFrame de pandas
data = pd.read_csv(file_path)

# Lista de terpenos deseados
terpenos_deseados = ['relaxed', 'happy', 'euphoric', 'uplifted', 'sleepy', 'creative', 'energetic', 'focused', 'hungry']

X = data[terpenos_deseados]

# Normalizar los datos
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)
# Ajustar el número de clústeres
n_clusters = 100  # Puedes ajustar este valor
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
data['cluster'] = kmeans.fit_predict(X_normalized)

@app.route('/recomendar_cepa', methods=['POST'])
def recomendar_cepa():
    try:
        # Obtener preferencias de efectos del usuario desde el JSON de entrada en la API
        preferencias_usuario = request.json['preferencias_usuario']
        print(preferencias_usuario)

        # Normalizar las preferencias del usuario
        preferencias_usuario_normalized = scaler.transform([preferencias_usuario])

        # Asignar la cepa más cercana al cluster de preferencias del usuario
        cluster_recomendado = kmeans.predict(preferencias_usuario_normalized)[0]

        # Obtener las cepas en el mismo cluster
        cepas_en_cluster = data[data['cluster'] == cluster_recomendado]

        # Seleccionar las 5 primeras cepas del cluster
        cepas_recomendadas = cepas_en_cluster.head(5)[['name','img_url','relaxed', 'happy', 'euphoric', 'uplifted', 'sleepy', 'creative', 'energetic', 'focused', 'hungry']].to_dict(orient='records')

        # Devolver las cepas recomendadas como JSON
        resultado = {
            'cepas_recomendadas': cepas_recomendadas,
            'mensaje': 'Predicción Exitosa'
        }
        print(resultado)

    except Exception as e:
        resultado = {'mensaje': f'Error: {str(e)}'}

    return jsonify(resultado)

if __name__ == '__main__':
    app.run(debug=True)
