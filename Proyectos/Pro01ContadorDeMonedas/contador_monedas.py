# Uso
# python contador_monedas.py --imagen imagenes/coins.png
import numpy as np
import argparse
import imutils
import cv2

# Construye el analizador de argumentos y analiza los argumentos
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--imagen", required = True,
	help = "Ruta de la imagen")
args = vars(ap.parse_args())

# Carga la imagen, la conviérte a escala de grises y la difumína
# ligeramente
imagen = cv2.imread(args["imagen"])
gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
difuminada = cv2.GaussianBlur(gris, (11, 11), 0)
cv2.imshow("Imagen", imagen)
#cv2.imshow("Gris", gris)
#cv2.imshow("Difuminada", difuminada)

# Se aplica detección de bordes a la imagen para revelar los
#  contornos de las monedas.
bordes = cv2.Canny(difuminada, 30, 150)
cv2.imshow("Bordes", bordes)

# Encuentra contornos en la imagen con bordes.
# NOTA: El método cv2.findContours es DESTRUCTIVO para la imagen
# que ingresa. Si tiene la intención de reutilizar su imagen con
# bordes, asegúrese de copiarla antes de llamar a cv2.findContours
contornos = cv2.findContours(bordes.copy(), cv2.RETR_EXTERNAL,
 cv2.CHAIN_APPROX_SIMPLE)
contornos = imutils.grab_contours(contornos)

# ¿Cuántos contornos encontramos?
print("Se cuentan {} monedas en esta imagen".format(len(contornos)))

# Resaltemos las monedas en la imagen original dibujando un
# círculo verde alrededor de ellas
monedas = imagen.copy()
cv2.drawContours(monedas, contornos, -1, (0, 255, 0), 2)
cv2.imshow("Monedas", monedas)
cv2.waitKey(0)

# Ahora, recorramos cada contorno
for (i, c) in enumerate(contornos):
    # Podemos calcular el 'cuadro delimitador' para cada contorno,
    # que es el rectángulo que encierra el contorno
    (x, y, w, h) = cv2.boundingRect(c)

    # Ahora que tenemos el contorno, vamos a extraerlo usando
    # cortes con matrices.
    print("Moneda #{}".format(i + 1))
    moneda = imagen[y:y + h, x:x + w]
    cv2.imshow("Moneda", moneda)
    
    # Solo por diversión, construyamos una máscara para la moneda 
    # encontrando el círculo envolvente mínimo del contorno
    mascara = np.zeros(imagen.shape[:2], dtype = "uint8")
    ((centerX, centerY), radius) = cv2.minEnclosingCircle(c)
    cv2.circle(mascara, (int(centerX), int(centerY)), int(radius), 255, -1)
    mascara = mascara[y:y + h, x:x + w]
    cv2.imshow("Moneda con mascara", cv2.bitwise_and(moneda, moneda, mask = mascara))
    cv2.waitKey(0)