# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 21:24:20 2021

@author: Hrishi_rich
"""
# Import computer vision Library
import cv2


# Load some pre-trained data (haar cascade algorithm)
car_tracker = cv2.CascadeClassifier('haarcascades_cars.xml')
pedestrian_tracker = cv2.CascadeClassifier('haarcascade_fullbody.xml')


# Get video footage

video = cv2.VideoCapture('CAR & PEDESTAL (1).mp4')
#video = cv2.VideoCapture('CAR & PEDESTAL (2).mp4')
#video = cv2.VideoCapture('CAR & PEDESTAL (3).mp4')
#video = cv2.VideoCapture('CAR & PEDESTAL (4).mp4')
#video = cv2.VideoCapture('CAR & PEDESTAL (5).mp4')
#video = cv2.VideoCapture('CAR DASHCAM (accident prevention).mp4')
#video = cv2.VideoCapture('CAR DASHCAM (at night).mp4')
#video = cv2.VideoCapture('CAR DASHCAM (obstacle).mp4')
#video = cv2.VideoCapture('CAR DASHCAM (water obstacle).mp4')


# Iterate forever over frames
while True:
    
    # Read the current frames
    read_successful, frame = video.read()
    
    if read_successful:
        # Must convert to grayscale
        grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        break
    
    # Detect cars
    cars = car_tracker.detectMultiScale(grayscaled_frame)
    
    # Detect Pedestrians
    Pedestrians = pedestrian_tracker.detectMultiScale(grayscaled_frame)
    
    # Draw rectangles around cars
    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x+2, y+2), (x+w, y+h), (255, 0, 0), 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        
    # Draw rectangles around pedestrians
    for (x, y, w, h) in Pedestrians:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
        
    # Display the footage with the cars & pedestrians spotted
    cv2.imshow('Self Driving Cars', frame)
    
    # Listen for a key press for 1 millisecond, then move on
    key = cv2.waitKey(1)
    
    # Stop if Q key is pressed
    if key==81 or key==133:
        break
    
    
# Release the VideoCapture object
video.release()
        

print ('Code Finish')