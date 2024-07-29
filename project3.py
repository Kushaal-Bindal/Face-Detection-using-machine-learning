import face_recognition
import cv2
import numpy as np
import csv
import os
from datetime import datetime
from sklearn.linear_model import LogisticRegression

# Load known face encodings and names
alia_image = face_recognition.load_image_file("Photos/alia.jpg")
alia_encoding = face_recognition.face_encodings(alia_image)[0]

deepika_image = face_recognition.load_image_file("Photos/deepika.jpg")
deepika_encoding = face_recognition.face_encodings(deepika_image)[0]

ratan_image = face_recognition.load_image_file("Photos/ratan.jpg")
ratan_encoding = face_recognition.face_encodings(ratan_image)[0]

srk_image = face_recognition.load_image_file("Photos/srk.jpg")
srk_encoding = face_recognition.face_encodings(srk_image)[0]


known_face_encodings = [
    alia_encoding,
    deepika_encoding,
    ratan_encoding,
    srk_encoding
]

known_face_names = [
    "alia",
    "deepika",
    "ratan",
    "srk"
]

students = known_face_names.copy()

face_names = []

now = datetime.now()
current_date = now.strftime("%Y-%m-%d")

f = open(current_date + '.csv', 'w+', newline='')
lnwrite = csv.writer(f)

# Initialize logistic regression model for gender detection
logistic_model = LogisticRegression()

# Data for gender classification (X: known face encodings, y: gender labels)
X_train = known_face_encodings
y_train = ["female", "female", "male", "male"]  # Assuming genders based on known faces

# Train logistic regression model
logistic_model.fit(X_train, y_train)

video_capture = cv2.VideoCapture(0)

while True:
    _, frame = video_capture.read()
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert frame to RGB for face_recognition

    # Find all face locations in the frame
    face_locations = face_recognition.face_locations(rgb_frame)

    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for face_encoding, (top, right, bottom, left) in zip(face_encodings, face_locations):
        # Predict gender using logistic regression model
        gender_prediction = logistic_model.predict([face_encoding])[0]

        # Check if the face matches any known faces
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"
        if True in matches:
            best_match_index = matches.index(True)
            name = known_face_names[best_match_index]

            # Remove the detected student from the list
            if name in students:
                students.remove(name)
                current_time = now.strftime("%H-%M-%S")
                lnwrite.writerow([name, current_time, gender_prediction])  # Include gender prediction in CSV

        # Print only when a face is recognized
        print("Detected:", name, ", Gender:", gender_prediction)
        face_names.append(name)
    
        # Draw rectangle around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw label with name and gender
        label = f"{name}, {gender_prediction}"
        cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Display the frame
    cv2.imshow('Video', frame)

    # Exit loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close window
video_capture.release()
cv2.destroyAllWindows()
f.close()
