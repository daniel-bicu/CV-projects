# OpenCv - face detector app
# HAARCASCADE algorithm = trained on tons of faces (front faces)
# you can download dataset from opencv/data/haarcascades on Github
import cv2
from random import randrange


def detect_face(img):
    # Detect faces
    faces_coordinates = trained_face_data.detectMultiScale(grayscaled_img)
    # searching for different faces (multiscale)
    # it gives us the coordinates of faces - in fact of those rectangles

    # What's happening here is: detectMultiScale = no matter what is the scale of the face, it will detect (multifaces)

    # Detects objects of different sizes in the input image. The detected objects are returned as a list of rectangles.

    print(faces_coordinates)
    # (x, y, w, h) = faces_coordinates[0]  # a list of lists
    # eg. [367 140 337 337] - first 2 coordinates = up coordinates (left up) , next ones ( width, height )
    # x, y, w, h
    color = (0, 255, 0)
    border_pixel = 3

    for (x, y, w, h) in faces_coordinates:
        face_detected = cv2.rectangle(img, (x, y), (x + w, y + h), (
            randrange(256), randrange(256), randrange(256)), border_pixel)

    # Display the images with the faces

    cv2.imshow('Face detection', face_detected)
    cv2.waitKey()


def detect_faces_realtime():
    # Realtime video capture from webcam

    webcam = cv2.VideoCapture(0)  # or can give a string for a video you have

    # Go through frames from the video
    while True:
        result, frame = webcam.read()
        grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces_coordinates = trained_face_data.detectMultiScale(grayscaled_img)
        for (x, y, w, h) in faces_coordinates:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (
                randrange(256), randrange(256), randrange(256)), 3)

        cv2.imshow('Realtime Face detector app', frame)
        key = cv2.waitKey(1)  # wait key or go next after 1 mil. seconds

        if key > 0:
            if chr(key) == 'q':
                break

    webcam.release()


if __name__ == '__main__':
    # Load some pre-trained data on face frontals from OpenCv
    trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')  # make a classifier (detector)

    img = cv2.imread('generic_h_face.jpg')
    print(img.shape)  # ( 791, 1000, 3)
    grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # grayscale = No RGB so we have just a ONE CHANNEL,
    # cvt = convertColor

    # In OpenCv the order is reversed like RGB -> BGR so that's why e have BGR2GRAY = rgb to gray

    print(grayscaled_img.shape)  # (791, 1000, 1)

    """ show the image """
    # cv2.imshow('generic human', img)
    # cv2.waitKey()  # Pause the execution, and after the wait (any key) -> the execution will be continued

    # So now we have to train the algorithm to detect faces

    # detect_face(grayscaled_img)
    detect_faces_realtime()
