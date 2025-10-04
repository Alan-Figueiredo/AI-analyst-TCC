import cv2
import mediapipe as mp
import math

video = cv2.VideoCapture("videoteste.mp4")
pose = mp.solutions.pose
Pose = pose.Pose(min_tracking_confidence=0.5,min_detection_confidence=0.5)
draw = mp.solutions.drawing_utils

while True:
    success,img = video.read()
    videoRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results = Pose.process(videoRGB)
    points = results.pose_landmarks
    draw.draw_landmarks(img,points,pose.POSE_CONNECTIONS)
    h,w, _ = img.shape
    
    
    if points:
        moDY = int(points.landmark[pose.PoseLandmark.RIGHT_WRIST].y*h)
        moDX = int(points.landmark[pose.PoseLandmark.RIGHT_WRIST].x*w)
        moEY = int(points.landmark[pose.PoseLandmark.LEFT_WRIST].y*h)
        moEX = int(points.landmark[pose.PoseLandmark.LEFT_WRIST].x*w)
        
        dist = math.hypot(moDX - moEX, moDY - moEY)

        
        print(f"maos {dist}")
        

    cv2.imshow("Resultado",img)
    cv2.waitKey(1)
    