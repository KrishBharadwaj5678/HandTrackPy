import mediapipe as mp
import cv2
import time

cap = cv2.VideoCapture(0)

ctime=0
ptime=0

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(img,handLms,mpHands.HAND_CONNECTIONS)

    ctime=time.time()

    fps=1/(ctime - ptime)
    ptime=ctime

    cv2.putText(img,f"FPS: {str(int(fps))}",(13,70),cv2.FONT_HERSHEY_PLAIN,2,(255,0,0),3)

    cv2.imshow("Image",img)
    cv2.waitKey(1)