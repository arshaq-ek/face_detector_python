import cv2
import face_recognition
import numpy as np
import os


path = 'C:\\Users\\arsha\\OneDrive\\Desktop\\face_detector\\images'
images = []
classnames =[]
mylist = os.listdir(path)
print(mylist)

for cl in mylist:
    crimg = cv2.imread(f'{path}/{cl}')
    images.append(crimg)
    classnames.append(os.path.splitext(cl)[0])

print(classnames)


def findencodings(images):
    encodelist = []
    for img in images:
        img= cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodelist.append(encode)
    return encodelist

encodelistknown = findencodings(images)
print("encoding complete...")


cap = cv2.VideoCapture(0)

while True:
    sucess , img = cap.read()
    imgs = cv2.resize(img,(0,0),None,0.25,0.25)
    imgs = cv2.cvtColor(imgs,cv2.COLOR_BGR2RGB)
    facesCurFrame = face_recognition.face_locations(imgs)
    encodeCurFrame= face_recognition.face_encodings(imgs)

    for encodeFace,faceLoc in zip(encodeCurFrame,facesCurFrame):
        matches = face_recognition.compare_faces(encodelistknown,encodeFace)
        facedis = face_recognition.face_distance(encodelistknown,encodeFace)
        print(facedis)
        matchindx = np.argmin(facedis)

        if matches[matchindx]:
            name = classnames[matchindx].upper()
            print(name)
            y1,x2,y2,x1 = faceLoc
            y1,x2,y2,x1 =  y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2 - 35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)



    cv2.imshow('Webcam',img)
    cv2.waitKey(1)