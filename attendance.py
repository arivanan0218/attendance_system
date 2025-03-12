import numpy as np
import imutils
import pickle
import time
import cv2
import csv

# File paths
embeddingFile = "output/embeddings.pickle"
embeddingModel = "openface_nn4.small2.v1.t7"
recognizerFile = "output/recognizer.pickle"
labelEncFile = "output/le.pickle"
conf = 0.5  # Minimum confidence for detection
attendance_recorded = set()  # Track students who have been marked

print("[INFO] Loading face detector...")
prototxt = "model/deploy.prototxt"
model = "model/res10_300x300_ssd_iter_140000.caffemodel"

try:
    detector = cv2.dnn.readNetFromCaffe(prototxt, model)
except Exception as e:
    print(f"❌ Error loading face detector: {e}")
    exit()

print("[INFO] Loading face recognizer...")
try:
    embedder = cv2.dnn.readNetFromTorch(embeddingModel)
    recognizer = pickle.loads(open(recognizerFile, "rb").read())
    le = pickle.loads(open(labelEncFile, "rb").read())
except Exception as e:
    print(f"❌ Error loading face recognizer: {e}")
    exit()

print("[INFO] Starting video stream...")
cam = cv2.VideoCapture(0)
if not cam.isOpened():
    print("❌ Error: Could not open webcam.")
    exit()

time.sleep(1.0)

while True:
    face_detected = False
    start_time = time.time()

    while True:
        ret, frame = cam.read()
        if not ret:
            print("❌ Error: Could not read frame from webcam.")
            break

        frame = imutils.resize(frame, width=600)
        (h, w) = frame.shape[:2]

        imageBlob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)), 1.0, (300, 300),
            (104.0, 177.0, 123.0), swapRB=False, crop=False
        )

        detector.setInput(imageBlob)
        detections = detector.forward()

        recognized = False  # Track if a valid student is recognized

        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > conf:
                face_detected = True
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                face = frame[startY:endY, startX:endX]
                (fH, fW) = face.shape[:2]

                if fW < 20 or fH < 20:
                    continue

                faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
                embedder.setInput(faceBlob)
                vec = embedder.forward()

                preds = recognizer.predict_proba(vec)[0]
                j = np.argmax(preds)
                proba = preds[j] * 100  # Convert to percentage
                name = le.classes_[j]

                # Fetch roll number from student.csv
                roll_number = None
                with open('student.csv', 'r') as csvFile:
                    reader = csv.reader(csvFile)
                    for row in reader:
                        if name in row:
                            roll_number = row[1]
                            break

                text = "{} : {} : {:.2f}%".format(name, roll_number, proba)
                y = startY - 10 if startY - 10 > 10 else startY + 10
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # If accuracy is 98% or higher and student is not already marked
                if proba >= 98 and name not in attendance_recorded:
                    print(f"✅ Verified Successfully: {name} ({roll_number}) - Marking attendance")
                    with open('attendance.csv', 'a', newline='') as csvFile:
                        writer = csv.writer(csvFile)
                        writer.writerow([name, roll_number, time.strftime("%Y-%m-%d %H:%M:%S")])
                    attendance_recorded.add(name)
                    recognized = True
                    break  # Stop processing for this student

        # If no valid student was recognized in 20 seconds, mark "Verification Unsuccessful"
        if not recognized and time.time() - start_time > 20:
            print("❌ Verification Unsuccessful - Fake student detected!")
            break

        # Show the frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # Press 'Esc' key to exit
            cam.release()
            cv2.destroyAllWindows()
            exit()

cam.release()
cv2.destroyAllWindows()
