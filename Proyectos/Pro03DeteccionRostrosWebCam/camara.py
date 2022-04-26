# Uso:
#
# python camara.py --rostro cascadas/haarcascade_frontalface_default.xml 
#
# python camara.py --rostro cascadas/haarcascade_frontalface_default.xml 
#    --video video/videoSalida.avi

from detector.detector_de_rostros import DetectorRostro
import argparse
import imutils
import cv2

# construir el analizador de argumentos
ap = argparse.ArgumentParser()
ap.add_argument("-r", "--rostro", required=True,
    help="ruta donde se encuentra la cascada de rostros")
ap.add_argument("-v", "--video",
    help="ruta donde se encuentra el vídeo (opcional)")
args = vars(ap.parse_args())

# construir el detector de rostros
dr = DetectorRostro(args["rostro"])

# En caso no exista una ruta para el video, se usa la
# camara
if not args.get("video",False):
    camara = cv2.VideoCapture(0)
# en caso contrario, cargar el video
else:
    camara = cv2.VideoCapture(args["video"])

while True:
    # recuperar el frame actual
    (grabado,frame) = camara.read()

    # en el caso se este peocesando un video y ya no se
    # grabe un frame, se alcanzó el final del vídeo
    if args.get("video") and not grabado:
        break

    # redimensionar el frame y se convierte a escala de
    # grises
    frame = imutils.resize(frame,width=300)
    gris = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    # detectar los rostros en el frame (imagen) y clonarla
    # para dibujar en esta
    rectsRostros = dr.detectar(gris,factorEscala=1.1, vecMin=5,
        tamMin=(30, 30))
    clonFrame = frame.copy()

    # bucla para dibujar los rectángulos en el clon
    for (fX, fY, fW, fH) in rectsRostros:
        cv2.rectangle(clonFrame, (fX,fY),(fX + fW,fY + fH),(0,255,0))

    # mostrar los rostros detectados
    cv2.imshow("Rostro",clonFrame)

    # if se presiona 's', se detiene el bucle
    if cv2.waitKey(1) & 0xFF == ord("s"):
        break

# liberar la camara y cerrar cualquier ventana abierta
camara.release()
cv2.destroyAllWindows()
