import cv2
import face_recognition as face_rec
import os
import numpy as np
from datetime import datetime
import csv

location = "Location A"
criminals_list = ["Jeff Bezos", "Sundar Pichai"]
# These Names aren't intentional, they are just for the purpose of the project
path = 'Images'
images = []
person_names = []
myList = os.listdir(path)
for image in myList:
    curImg = cv2.imread(f'{path}/{image}')
    images.append(curImg)
    person_names.append(os.path.splitext(image)[0])


def find_encodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_rec.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


def save_data(name_of_person, time, person_location, database):
    data = [[name_of_person, time, person_location]]
    file = open(database, 'a+', newline='')

    with file:
        write = csv.writer(file)
        write.writerows(data)


encode_known_persons = find_encodings(images)

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    image_small = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    image_small = cv2.cvtColor(image_small, cv2.COLOR_BGR2RGB)

    faces_in_current_image = face_rec.face_locations(image_small)
    encode_faces_in_current_image = face_rec.face_encodings(image_small, faces_in_current_image)

    for encode_face, face_location in zip(encode_faces_in_current_image, faces_in_current_image):
        matches = face_rec.compare_faces(encode_known_persons, encode_face)
        face_distance = face_rec.face_distance(encode_known_persons, encode_face)
        matchIndex = np.argmin(face_distance)

        if matches[matchIndex]:
            name = person_names[matchIndex]
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            print(name)
            if name in criminals_list and name not in open('criminal_database.csv').read():
                save_data(name, current_time, location, 'criminal_database.csv')
            elif name not in criminals_list and name not in open('database.csv').read():
                save_data(name, current_time, location, 'database.csv')

    cv2.imshow(location, img)
    cv2.waitKey(1)

