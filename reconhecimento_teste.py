import os
import glob
import _pickle as cPickle
import dlib
import cv2
import numpy as np

detectorFace = dlib.get_frontal_face_detector()
detectorPontos = dlib.shape_predictor('recursos/shape_predictor_68_face_landmarks.dat')
reconhecimentoFacial = dlib.face_recognition_model_v1('recursos/dlib_face_recognition_resnet_model_v1.dat')
indices = np.load('recursos/indices_rn.pickle', allow_pickle=True)
descritoresFaciais = np.load('recursos/descritores_rn.npy')
limiar = 0.5



for arquivo in glob.glob(os.path.join('fotos','*.jpeg')):
    imagem = cv2.imread(arquivo)
    facesDetectadas = detectorFace(imagem,2)
    numeroFacesDetectadas = len(facesDetectadas)

    for face in facesDetectadas:
        l,t,r,b = (int(face.left()),int(face.top()),int(face.right()),int(face.bottom()))
        pontosFaciais = detectorPontos(imagem,face)
        descritorFacial = reconhecimentoFacial.compute_face_descriptor(imagem,pontosFaciais)
        listaDescritorFacial = [fd for fd in descritorFacial]

        npArrayDescritorFacial = np.asarray(listaDescritorFacial, dtype=np.float64)
        npArrayDescritorFacial = npArrayDescritorFacial[np.newaxis, :]
        distancias = np.linalg.norm(npArrayDescritorFacial - descritoresFaciais, axis=1)
        #print('Distancias {}'.format(distancias))
        minimo = np.argmin(distancias)
        distanciaMinima = distancias[minimo]

        if distanciaMinima <= limiar:
            nome = os.path.split(indices[minimo])[1].split(".")[0]
        else:
            nome = "Desconhecido"

        cv2.rectangle(imagem,(l,t),(r,b),(255,255,0),2)
        text = "{} {:.4f}".format(nome,distanciaMinima)
        cv2.putText(imagem,text, (r,t), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (0, 255, 255))

    cv2.imshow('Detector Hog',imagem)
    cv2.waitKey(0)
cv2.destroyAllWindows()
