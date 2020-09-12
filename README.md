# Webcam_Facial_Recognition

In order to run propperly this code, follow the following steps:

- Create a folder named "fotos" and inside of it, create another named "treinamento"
- Inside "treinamento" add as much as you want photos for training your recognition model.
- Those photos must be named like: "Person Name"."Photo Number".jpeg
- Download dlib_face_recognition_resnet_model_v1.dat and add it into recursos' folder
- Run the code into reconhecimento_treinamento.py to generate you .xml file containing the faces parameters
- Start a serial connection and configure your port and baudrate
- Run webcam_recognition.py and try it out

If you want to recognize static images, add them into fotos' folder and run reconhecimento_teste.py
