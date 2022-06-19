import cv2
import numpy as np
import face_recognition

imgRohit=face_recognition.load_image_file('images/Rohit_Sharma_November_2016.jpg','RGB')
imgTest=face_recognition.load_image_file('images/rohittest.jpg','RGB')

faceLoc=face_recognition.face_locations(imgRohit)[0]
encodeRohit=face_recognition.face_encodings(imgRohit)[0]
cv2.rectangle(imgRohit,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,0),2)

faceLocTest=face_recognition.face_locations(imgTest)[0]
encodeTest=face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest,(faceLocTest[3],faceLocTest[0]),(faceLocTest[1],faceLocTest[2]),(255,0,0),2)

results=face_recognition.compare_faces([encodeRohit],encodeTest)
faceDis=face_recognition.face_distance([encodeRohit],encodeTest)
print(results,faceDis)
cv2.putText(imgTest,f'{results}{round(faceDis[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)

cv2.imshow("Rohit",imgRohit)
cv2.imshow("Rohit Test",imgTest)
cv2.waitKey(0)