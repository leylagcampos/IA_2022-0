# Uso:
# python deteccion_rostros.py --rostro cascadas/haarcascade_frontalface_default.xml 
#    --imagen imagenes/obama.png

from detector.detector_de_rostros import DetectorRostro
import argparse
import cv2

# construir el analizador de argumentos
ap = argparse.ArgumentParser()
ap.add_argument("-r", "--rostro", required=True,
    help="ruta donde se encuentra la cascada de rostros")
ap.add_argument("-i", "--imagen", required=True,
    help="ruta donde se encuentra la imagen")
args = vars(ap.parse_args())

# cargar la imagen y convertirla a escala de grises
imagen = cv2.imread(args["imagen"])
gris = cv2.cvtColor(imagen,cv2.COLOR_BGR2GRAY)

# encontrar los rostros en la imagen
dr = DetectorRostro(args["rostro"])
rectRostros = dr.detectar(gris,factorEscala=1.1, vecMin=5, tamMin=(30, 30))
print("Se encontró {} rostro(s)".format(len(rectRostros)))

# dibujar los rectángulos sobre los rostros
for(x , y, w, h) in rectRostros:
    cv2.rectangle(imagen,(x,y),(x+w,y+h),(0,255,0),2)

cv2.imshow("Rostros",imagen)
cv2.waitKey(0)
