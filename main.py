import cv2
import numpy as np
import imutils
from centroid_tracker import CentroidTracker

# Rutas a los modelos
prototxt_path = "models/MobileNetSSD_deploy.prototxt"
model_path   = "models/MobileNetSSD_deploy.caffemodel"
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant",
           "sheep", "sofa", "train", "tvmonitor"]

vs = cv2.VideoCapture(0)
tracker = CentroidTracker(maxDisappeared=40, maxDistance=50)

def send_control_command(objectID, centroid, frame_shape):
    h, w = frame_shape
    norm_x = centroid[0] / float(w)
    norm_y = centroid[1] / float(h)
    print(f"[Control] ID {objectID}: x={norm_x:.2f}, y={norm_y:.2f}")

while True:
    ret, frame = vs.read()
    if not ret: break
    frame = imutils.resize(frame, width=600)
    (h, w) = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    rects = []
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            idx = int(detections[0, 0, i, 1])
            if CLASSES[idx] != "person": continue
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            rects.append((startX, startY, endX, endY))
            cv2.rectangle(frame, (startX, startY), (endX, endY),
                          (0, 255, 0), 2)

    objects = tracker.update(rects)

    for (objectID, centroid) in objects.items():
        cv2.putText(frame, f"ID {objectID}",
                    (centroid[0] - 10, centroid[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.circle(frame, (centroid[0], centroid[1]), 4,
                   (0, 0, 255), -1)
        send_control_command(objectID, centroid, (h, w))

    cv2.imshow("Detección y Seguimiento", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

vs.release()
cv2.destroyAllWindows()
