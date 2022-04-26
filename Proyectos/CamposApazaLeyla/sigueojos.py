
from seguidor.seguidordeojos import SeguidorDeOjos
import numpy as np
import imutils
import cv2

args = { "face":"cascadas/haarcascade_frontalface_default.xml","eye":"cascadas/haarcascade_eye.xml"}

et = SeguidorDeOjos(args["face"], args["eye"])

rojoMin = np.array([0, 0, 155], dtype = "uint8")
rojoMax = np.array([50,55,255], dtype = "uint8")

captura = cv2.VideoCapture(0)
salida = cv2.VideoWriter('video/CamposApazaLeyla.avi',cv2.VideoWriter_fourcc(*'XVID'),20.0,(640,480))
while (captura.isOpened()):
    ret, fotograma = captura.read()
    if ret == True:
        gray = cv2.cvtColor(fotograma, cv2.COLOR_BGR2GRAY)
        rects = et.seguir(gray)
        
        for rect in rects:
            cv2.rectangle(fotograma, (rect[0], rect[1]), (rect[2], rect[3]), (0, 255, 0), 2)
        
        #Determinar los pixeles que caen dentro de los límites y luego difuminar la imagen binaria
        rojo = cv2.inRange(fotograma, rojoMin, rojoMax)
        rojo = cv2.GaussianBlur(rojo, (3, 3), 0)
        # encontrar los contornos en la imagen
        cnts = cv2.findContours(rojo.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
    
        # revisar si se encontro algun contorno 
        if len(cnts) > 0:
            # ordena los contornos y encuentra el más grande
            cnt = sorted(cnts, key = cv2.contourArea, reverse = True)[0]
            # calcular el cuadro delimitador (girado) alrededor del contorno y luego dibujarlo
            rect = np.int32(cv2.boxPoints(cv2.minAreaRect(cnt)))
            cv2.drawContours(fotograma, [rect], -1, (0, 255, 0), 2)
        
        cv2.imshow("Fotograma", fotograma)
        #cv2.imshow("Binaria", rojo)
        salida.write(fotograma)
        
        if cv2.waitKey(1) & 0xFF == ord('s'):
            break
    else:
        break
captura.release()
salida.release()
cv2.destroyAllWindows()

#Recuperar el video

captura = cv2.VideoCapture('video/CamposApazaLeyla.avi')
while (captura.isOpened()):
    ret, imagen = captura.read()
    if ret == True:
        cv2.imshow('Video',imagen)
        if cv2.waitKey(1) == ord('s'):
            break
    else:
        break
captura.release()
cv2.destroyAllWindows()