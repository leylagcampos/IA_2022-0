import cv2

class DetectorRostro:
    def __init__(self,rutaCascadaRostros):
        # cargar el detector de rostros
        self.cascadaRostro = cv2.CascadeClassifier(rutaCascadaRostros)
    
    def detectar(self,imagen,factorEscala = 1.1, vecMin = 5,tamMin = (30,30)):
        # detectar los rostros en la imagen
        rects = self.cascadaRostro.detectMultiScale(imagen, scaleFactor = factorEscala,
            minNeighbors = vecMin, minSize = tamMin, flags = cv2.CASCADE_SCALE_IMAGE)
        
        # devolver los rectangulos representando los cuadros en los
        # que se enmarcan la imagen
        return rects
