import cv2
import mediapipe as mp
import math

video = cv2.VideoCapture("polichinelos.mp4")
pose = mp.solutions.pose
Pose = pose.Pose(min_tracking_confidence=0.5,min_detection_confidence=0.5)
draw = mp.solutions.drawing_utils
cont = 0
check = True

while True:
    success,img = video.read()
    videoRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results = Pose.process(videoRGB)
    points = results.pose_landmarks
    draw.draw_landmarks(img,points,pose.POSE_CONNECTIONS)
    h,w, _ = img.shape
    
    if points:
        moDY = int(points.landmark[pose.PoseLandmark.RIGHT_INDEX].y*h)
        moDX = int(points.landmark[pose.PoseLandmark.RIGHT_INDEX].x*w)
        moEY = int(points.landmark[pose.PoseLandmark.LEFT_INDEX].y*h)
        moEX = int(points.landmark[pose.PoseLandmark.LEFT_INDEX].x*w)
    
        distMO = math.hypot(moDX-moEX, moDY-moEY)
        
        print(f"maos {distMO}")
        # maos <=150
        
        if check == True and distMO <=150:
            cont+=1
            check = False
            
        if distMO > 150:
            check = True
          
        texto = f'QTD {cont}'  
        cv2.rectangle(img,(20,240),(280,120),(255,0,0),-1)
        cv2.putText(img,texto,(40,200),cv2.FONT_HERSHEY_SIMPLEX,2,(255,255,255),5)
        

    cv2.imshow("Resultado",img)
    cv2.waitKey(10)
    