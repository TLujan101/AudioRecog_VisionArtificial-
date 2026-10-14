import os
import librosa
import numpy as np
import matplotlib.pyplot as plt

# Configurar las carpetas de origen y destino
CARPETA_AUDIOS = "audios"
CARPETA_ESPECTROGRAMAS = "spectrograms"
os.makedirs(CARPETA_ESPECTROGRAMAS, exist_ok=True)

# Extensiones de audio válidas
EXTENSIONES_PERMITIDAS = ('.wav', '.mp3', '.aiff', '.flac')

print("Iniciando la conversion de archivos en la carpeta audios.")

for archivo in os.listdir(CARPETA_AUDIOS):
    if archivo.lower().endswith(EXTENSIONES_PERMITIDAS):
        ruta_audio = os.path.join(CARPETA_AUDIOS, archivo)
        
        ruta_audio = os.path.join(CARPETA_AUDIOS, archivo)
        
        # 1. Definir el nombre del archivo de imagen de salida primero
        nombre_base = os.path.splitext(archivo)[0]
        ruta_imagen = os.path.join(CARPETA_ESPECTROGRAMAS, f"{nombre_base}.png")
        
        # 2. COMPROBACIÓN: Si la imagen ya existe, saltar al siguiente audio
        if os.path.exists(ruta_imagen):
            print(f"Omitido (Ya convertido): {archivo}")
            continue
        
        try:
            # 1. Cargar el audio (remuestreado automáticamente a 22050 Hz)
            y, sr = librosa.load(ruta_audio, sr=None) 
            
            # 2. Calcular el Espectrograma de Mel
            # n_mels=128 es el estándar ideal para la resolución de una CNN
            mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=256, n_fft=4096, hop_length=1024)
            
            # 3. Escalar la amplitud a logarítmica (Decibelios)
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            # 4. Definir el nombre del archivo de imagen de salida
            nombre_base = os.path.splitext(archivo)[0]
            ruta_imagen = os.path.join(CARPETA_ESPECTROGRAMAS, f"{nombre_base}.png")
            
            # 5. Guardar el espectrograma como imagen pura (sin bordes, ejes ni marcos)
            plt.imsave(ruta_imagen, mel_spec_db, cmap='magma')
            print(f"Convertido con exito: {archivo} -> {nombre_base}.png")
            
        except Exception as e:
            print(f"Error procesando {archivo}: {e}")

print("Proceso Terminado.")
