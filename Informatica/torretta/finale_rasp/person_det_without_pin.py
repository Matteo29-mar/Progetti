import cv2
import datetime
import imutils
import numpy as np
#import RPi.GPIO as GPIO
import threading
import time  # Aggiunto l'importazione mancante


def funz():
    print("Ho finito!")


def timer1(tx):
    global t1
    dt = time.time() * 1000 - t1
    ret = 0
    if dt >= tx:
        t1 = time.time() * 1000
        ret = 1
    return ret

# Inizializzazione del timer
t1 = time.time() * 1000


def timer2(tx):
    global t2
    dt = time.time() * 1000 - t2
    ret = 0
    if dt >= tx:
        t2 = time.time() * 1000
        ret = 1
    return ret

# Inizializzazione del timer
t2 = time.time() * 1000

protopath = "MobileNetSSD_deploy.prototxt"
modelpath = "MobileNetSSD_deploy.caffemodel"
detector = cv2.dnn.readNetFromCaffe(prototxt=protopath, caffeModel=modelpath)

# Only enable it if you are using OpenVino environment
# detector.setPreferableBackend(cv2.dnn.DNN_BACKEND_INFERENCE_ENGINE)
# detector.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

def main():
    sierra = 0
    delta = 0
    cap = cv2.VideoCapture(0)

    fps_start_time = datetime.datetime.now()
    fps = 0
    total_frames = 0

    while True:
        ret, frame = cap.read()
        frame = imutils.resize(frame, width=200, height=600)
        total_frames = total_frames + 1

        (H, W) = frame.shape[0], frame.shape[1]  # Corretto l'accesso agli elementi della tupla

        blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)

        detector.setInput(blob)
        person_detections = detector.forward()

        person_detected = False  # Aggiunto per tenere traccia se è stata rilevata una persona

        for i in np.arange(0, person_detections.shape[2]):
            confidence = person_detections[0, 0, i, 2]
            if confidence > 0.5:
                idx = int(person_detections[0, 0, i, 1])

                if CLASSES[idx] != "person":
                    continue

                person_box = person_detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                (startX, startY, endX, endY) = person_box.astype("int")

                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)

                # Aggiungi questa linea per stampare "preso" quando una persona è rilevata
                print("preso")
                stop_movement()
                person_detected = True  # Imposta a True se è stata rilevata una persona

        if not person_detected:
            # Se nessuna persona è stata rilevata, stampa "niente"
            print("niente")

        fps_end_time = datetime.datetime.now()
        time_diff = fps_end_time - fps_start_time
        if time_diff.seconds == 0:
            fps = 0.0
        else:
            fps = (total_frames / time_diff.seconds)

        fps_text = "FPS: {:.2f}".format(fps)

        cv2.putText(frame, fps_text, (5, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)

        cv2.imshow("Application", frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    # Rilascia la webcam e chiudi la finestra fuori dal ciclo while
    cap.release()
    cv2.destroyAllWindows()

main()  # Corretto l'invocazione della funzione main()
