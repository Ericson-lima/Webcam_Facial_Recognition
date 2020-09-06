import os
import glob
import _pickle as cPickle
import dlib
import cv2
import numpy as np

detectorFace = dlib.get_frontal_face_detector()
detectorPontos = dlib.shape_predictor('recursos/shape_predictor_68_face_landmarks.dat')
reconhecimentoFacial = dlib.face_recognition_model_v1('recursos/dlib_face_recognition_resnet_model_v1.dat')

indice={}
idx=0
descritoresFaciais = None

for arquivo in glob.glob(os.path.join('fotos/treinamento','*.jpeg')):
    imagem = cv2.imread(arquivo)
    facesDetectadas = detectorFace(imagem,1)
    numeroFacesDetectadas = len(facesDetectadas)
    if numeroFacesDetectadas > 1:
        print('Mais de uma face na imagem {}'.format(arquivo))
    elif numeroFacesDetectadas < 1:
        print('Nenhuma face detectada em {}'.format(arquivo))

    for face in facesDetectadas:
        pontosFaciais = detectorPontos(imagem, face)
        descritorFacial = reconhecimentoFacial.compute_face_descriptor(imagem,pontosFaciais)
        listaDescritorFacial = [df for df in descritorFacial]

        npArrayDescritorFacial = np.asarray(listaDescritorFacial, dtype=np.float64)
        npArrayDescritorFacial = npArrayDescritorFacial[np.newaxis, :]

        if descritoresFaciais is None:
                descritoresFaciais= npArrayDescritorFacial;
        else:
            descritoresFaciais = np.concatenate((descritoresFaciais,npArrayDescritorFacial),axis=0)
        indice[idx] = arquivo
        idx+=1
print("Tamanho: {} Formato: {}".format(len(descritoresFaciais), descritoresFaciais.shape))
np.save('recursos/descritores_rn.npy',descritoresFaciais)
with open("recursos/indices_rn.pickle",'wb') as f:
    cPickle.dump(indice, f)

