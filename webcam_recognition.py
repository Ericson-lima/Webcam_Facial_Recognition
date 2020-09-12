import os
import dlib
import cv2
import numpy as np
import serial

ser = serial.Serial('COM5', 9600, timeout=0)
detectorFace = dlib.get_frontal_face_detector()
detectorPontos = dlib.shape_predictor('recursos/shape_predictor_68_face_landmarks.dat')
reconhecimentoFacial = dlib.face_recognition_model_v1('recursos/dlib_face_recognition_resnet_model_v1.dat')
indices = np.load('recursos/indices_rn.pickle', allow_pickle=True)
descritoresFaciais = np.load('recursos/descritores_rn.npy')
limiar = 0.5

video = cv2.VideoCapture(0)
key = None;
while True:

    check,frame = video.read()
    facesDetectadas = detectorFace(frame,2)
    numeroFacesDetectadas = len(facesDetectadas)

    for face in facesDetectadas:
        l,t,r,b = (int(face.left()),int(face.top()),int(face.right()),int(face.bottom()))
        pontosFaciais = detectorPontos(frame,face)
        descritorFacial = reconhecimentoFacial.compute_face_descriptor(frame,pontosFaciais)
        listaDescritorFacial = [fd for fd in descritorFacial]

        npArrayDescritorFacial = np.asarray(listaDescritorFacial, dtype=np.float64)
        npArrayDescritorFacial = npArrayDescritorFacial[np.newaxis, :]
        distancias = np.linalg.norm(npArrayDescritorFacial - descritoresFaciais, axis=1)
        print('Distancias {}'.format(distancias))
        minimo = np.argmin(distancias)
        distanciaMinima = distancias[minimo]

        if distanciaMinima <= limiar:
            nome = os.path.split(indices[minimo])[1].split(".")[0]
        else:
            nome = "Desconhecido"

        cv2.rectangle(frame,(l,t),(r,b),(255,255,0),2)
        key = cv2.waitKey(1)
        #text = "{} {:.4f}".format(nome,distanciaMinima)

        text = '{}'.format(nome)
        ser.write(text.encode("utf-8"))
        cv2.putText(frame,nome, (r,t), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 255))
        cv2.imshow('Detector Hog', frame)

    try:
        resp = ser.read_all()
        if resp.decode() != "":
            print(resp.decode())
        
    except ser.SerialTimeoutException:
        print('Data could not be read')


video.release()
cv2.destroyAllWindows()
