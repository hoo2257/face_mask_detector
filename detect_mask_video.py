# USAGE
# python detect_mask_video.py

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
from playsound import playsound
import numpy as np
import argparse
import imutils
import time
import cv2
import os

def detect_and_predict_mask(frame, faceNet, maskNet):

    (h, w) = frame.shape[:2]
    # create blob object from input frame
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300,300),
                                 (104.0, 177.0, 123.0))

    # input blob object to faceNet
    faceNet.setInput(blob)
    detections = faceNet.forward()

    faces = [] # list of faces
    locs = []  # list of face locations
    preds = [] # list of predictions from mask network

    for i in range(0, detections.shape[2]):
        # get confidence associated with the detection
        confidence = detections[0, 0, i, 2]

        if confidence > args["confidence"]:
            # compute the coordinates fo the bounding box for the object
            box = detections[0, 0, i, 3:7]*np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # ensure the bounding box fall within the dimension of the frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w-1, endX), min(h-1, endY))

            # extract the face ROI, convert it to BGR to RGB channel
            # resize it to 224 X 224 and preprocess for network input
            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face,(224,224))
            face = img_to_array(face)
            face = preprocess_input(face)

            # add the face and bounding box to list
            faces.append(face)
            locs.append((startX, startY, endX, endY))

    # make prediction only if the face is detected
    if len(faces) > 0:
        faces = np.array(faces, dtype="float32")
        # for faster inference make batch for prediction on all
        # detected faces
        preds = maskNet.predict(faces, batch_size=32)

    return (locs, preds)

ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face", type=str,
                default="face_detector",
                help="path to face detector model directory")
ap.add_argument("-m", "--model", type=str,
                default="mask_detector_model",
                help="path to mask detector model directory")
ap.add_argument("-c", "--confidence", type=float, default=0.8,
                help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# Load face detector model
print("[INFO] loading face detector model...")
prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
weightsPath = os.path.sep.join([args["face"],
                                "res10_300x300_ssd_iter_140000.caffemodel"])
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# Load face mask detector model
print("[INFO] loading face mask detector model...")
maskNet = load_model(args["model"])

print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

while True:
    # grad the frame from threaded video stream and resize it to have a
    # maximum width of 400 pixels
    frame = vs.read()
    frame = imutils.resize(frame, width=400)

    # detect faces in the face and determine if they are wearing a mask
    (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

    for (box, pred) in zip(locs, preds):

        (startX, startY, endX, endY) = box
        (mask, withoutMask) = pred

        label = "Mask" if mask > withoutMask else "No Mask"

        # if label == "No Mask":
        #     playsound('beep_1.mp3')
        # else:
        #     pass

        color = (0,255,0) if label == "Mask" else (0, 0, 255)

        label = "{}: {:.2f}%".format(label, max(mask, withoutMask)*100)

        cv2.putText(frame, label, (startX, startY-10),
                    cv2.FONT_HERSHEY_SIMPLEX,0.45, color,2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()






