import os
import glob
from PIL import Image
import librosa

# Definir rutas relativas según la estructura de tu proyecto
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CARPETA_ESPECTROGRAMAS = os.path.join(BASE_DIR, "spectrograms")
CARPETA_AUDIOS = os.path.join(BASE_DIR, "audios")
CARPETA_SALIDA = os.path.join(BASE_DIR, "audios_div")

# Duración deseada de cada segmento en segundos
DURACION_SEGMENTO = 4.0 

# Porcentaje de solapamiento (overlap) para Data Augmentation (ej. 0.5 = 50% de solapamiento)
# Cambia a 0.0 si NO quieres solapamiento entre subdivisiones.
OVERLAP = 0.5 

def procesar_divisio_y_augmentation():
    # Crear carpeta de salida si no existe
    if not os.path.exists(CARPETA_SALIDA):
        os.makedirs(CARPETA_SALIDA)

    # Buscar todas las imágenes PNG en 'spectrograms'
    imagenes = glob.glob(os.path.join(CARPETA_ESPECTROGRAMAS, "*.png"))

    if not imagenes:
        print("No se encontraron imágenes en la carpeta 'spectrograms'.")
        return

    for ruta_img in imagenes:
        nombre_base = os.path.splitext(os.path.basename(ruta_img))[0]
        
        # Buscar el audio correspondiente en la carpeta 'audios' para conocer su duración real
        ruta_audio = os.path.join(CARPETA_AUDIOS, f"{nombre_base}.wav")
        
        if not os.path.exists(ruta_audio):
            print(f"[ADVERTENCIA] No se encontró el audio equivalente '{nombre_base}.wav'. Omitiendo...")
            continue

        # 1. Obtener duración del audio en segundos
        duracion_total = librosa.get_duration(path=ruta_audio)

        # 2. Leer la imagen del espectrograma
        img = Image.open(ruta_img)
        ancho_total, alto_total = img.size

        # 3. Calcular cuántos píxeles corresponden a 4 segundos de audio
        px_por_segundo = ancho_total / duracion_total
        ancho_ventana = int(px_por_segundo * DURACION_SEGMENTO)

        if ancho_ventana > ancho_total:
            print(f"[OMITIDO] '{nombre_base}.png' dura menos de 4 segundos ({duracion_total:.2f}s).")
            continue

        # 4. Definir el avance (paso) según el overlap deseado
        paso = int(ancho_ventana * (1 - OVERLAP))
        if paso < 1:
            paso = 1

        sub_idx = 1
        # Recorrer el eje horizontal (eje del tiempo)
        for x_inicio in range(0, ancho_total - ancho_ventana + 1, paso):
            x_fin = x_inicio + ancho_ventana
            
            # Recorte vertical completo (manteniendo todo el rango de frecuencia)
            caja_corte = (x_inicio, 0, x_fin, alto_total)
            sub_img = img.crop(caja_corte)

            # Nombre formateado: nombreOriginal_1.png, nombreOriginal_2.png, ..., n.png
            nombre_salida = f"{nombre_base}_{sub_idx}.png"
            ruta_guardado = os.path.join(CARPETA_SALIDA, nombre_salida)
            
            sub_img.save(ruta_guardado)
            sub_idx += 1

        print(f"[OK] {nombre_base}.png -> {sub_idx - 1} segmentos creados en 'audios_div'.")

if __name__ == "__main__":
    procesar_divisio_y_augmentation()