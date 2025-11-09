# A Gender and Age Detection program by Aditya Pathania

import cv2
import time
import argparse
import sys

def highlightFace(net, frame, conf_threshold=0.7):
    frameOpencvDnn = frame.copy()
    frameHeight, frameWidth = frameOpencvDnn.shape[:2]
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300),
                                 [104, 117, 123], True, False)
    net.setInput(blob)
    detections = net.forward()
    faceBoxes = []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            faceBoxes.append([x1, y1, x2, y2])
            cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2),
                          (0, 255, 0), int(round(frameHeight / 150)), 2)
    return frameOpencvDnn, faceBoxes


# Model paths
faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"
ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"
genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"

# Labels and constants
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
genderList = ['Male', 'Female']
ageMidpoints = [1, 5, 10, 17, 28, 40, 50, 80]

# Load networks
faceNet = cv2.dnn.readNet(faceModel, faceProto)
ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)

padding = 20

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--image", help="Path to image file")
args = parser.parse_args()

# ---------- IMAGE MODE ----------
def process_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Could not read image: {image_path}")
        sys.exit()

    resultImg, faceBoxes = highlightFace(faceNet, image)
    for faceBox in faceBoxes:
        y1 = max(0, faceBox[1] - padding)
        y2 = min(faceBox[3] + padding, image.shape[0] - 1)
        x1 = max(0, faceBox[0] - padding)
        x2 = min(faceBox[2] + padding, image.shape[1] - 1)

        if y2 <= y1 or x2 <= x1:
            continue
        face = image[y1:y2, x1:x2]
        if face.size == 0:
            continue

        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227),
                                     MODEL_MEAN_VALUES, swapRB=False)

        genderNet.setInput(blob)
        genderPreds = genderNet.forward()
        gender = genderList[genderPreds[0].argmax()]

        ageNet.setInput(blob)
        agePreds = ageNet.forward()
        predicted_age = int(sum([p * a for p, a in zip(agePreds[0], ageMidpoints)]))

        label = f"{gender}, {predicted_age} yrs"
        cv2.putText(resultImg, label, (x1 + 5, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(resultImg, label, (x1 + 5, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow("Image - Age and Gender Detection", resultImg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ---------- VIDEO MODE ----------
if args.image:
    process_image(args.image)
else:
    video = cv2.VideoCapture(0)
    last_check_time = 0
    last_predictions = []

    while True:
        hasFrame, frame = video.read()
        if not hasFrame:
            break

        resultImg, faceBoxes = highlightFace(faceNet, frame)
        current_time = time.time()

        # Update predictions every 2 seconds
        if current_time - last_check_time >= 2 or not last_predictions:
            last_predictions = []
            for faceBox in faceBoxes:
                y1 = max(0, faceBox[1] - padding)
                y2 = min(faceBox[3] + padding, frame.shape[0] - 1)
                x1 = max(0, faceBox[0] - padding)
                x2 = min(faceBox[2] + padding, frame.shape[1] - 1)

                if y2 <= y1 or x2 <= x1:
                    continue
                face = frame[y1:y2, x1:x2]
                if face.size == 0:
                    continue

                blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227),
                                             MODEL_MEAN_VALUES, swapRB=False)

                genderNet.setInput(blob)
                genderPreds = genderNet.forward()
                gender = genderList[genderPreds[0].argmax()]

                ageNet.setInput(blob)
                agePreds = ageNet.forward()
                predicted_age = int(sum([p * a for p, a in zip(agePreds[0], ageMidpoints)]))

                last_predictions.append((gender, predicted_age))

            last_check_time = current_time

        # Draw labels on video
        for i, faceBox in enumerate(faceBoxes):
            if i < len(last_predictions):
                gender, predicted_age = last_predictions[i]
                label = f"{gender}, {predicted_age} yrs"

                cv2.putText(resultImg, label, (faceBox[0] + 5, faceBox[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(resultImg, label, (faceBox[0] + 5, faceBox[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow("Detecting Age and Gender", resultImg)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()
