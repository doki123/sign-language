from re import L
# import time
import  string
import cv2
import numpy as np
import mediapipe as mp
import pandas as pd


cap = cv2.VideoCapture(0)
mpHands = mp.solutions.hands
# hands = mpHands.Hands()
# this is so that only one hand is detected at any time --> use above code if you want both hands
hands = mpHands.Hands(max_num_hands=1)
mpDraw = mp.solutions.drawing_utils

sign_class = int(input('What sign do you want to start from (index of alphabet letter)?'))
image_count = 0  # tracks the amount of signs that has been captured
df = {}
start_capture = 0  # when c has been pressed --> signals for camera to start recording
data_list = []
count = 0  # frame number --> used so that the columns in data all have different names


# Check if camera opened successfully
if (cap.isOpened() == False):
    print("Error opening video stream or file")

# Read the video
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(imgRGB)

        if results.multi_hand_landmarks:  # if hand is visible on screen and data is being acquired
            for handLms in results.multi_hand_landmarks:
                for id, lm in enumerate(handLms.landmark):
                    h, w, c = frame.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

                mpDraw.draw_landmarks(frame, handLms, mpHands.HAND_CONNECTIONS)
        if results.multi_hand_world_landmarks:  # if hand is visible on screen and data is being acquired
            for handLms in results.multi_hand_world_landmarks:

                if start_capture:

                    count += 1

                    data = {

                        'WRIST_X'+str(count): handLms.landmark[mpHands.HandLandmark.WRIST].x,
                        'WRIST_Y'+str(count): handLms.landmark[mpHands.HandLandmark.WRIST].y,
                        'WRIST_Z'+str(count): handLms.landmark[mpHands.HandLandmark.WRIST].z,


                        'THUMB_CMC_X'+str(count): handLms.landmark[mpHands.HandLandmark.THUMB_CMC].x,
                        'THUMB_CMC_Y'+str(count): handLms.landmark[mpHands.HandLandmark.THUMB_CMC].y,
                        'THUMB_CMC_Z'+str(count): handLms.landmark[mpHands.HandLandmark.THUMB_CMC].z,

                        'THUMB_MCP_X'+str(count): handLms.landmark[mpHands.HandLandmark.THUMB_MCP].x,
                        'THUMB_MCP_Y'+str(count): handLms.landmark[mpHands.HandLandmark.THUMB_MCP].y,
                        'THUMB_MCP_Z'+str(count): handLms.landmark[mpHands.HandLandmark.THUMB_MCP].z,

                        'THUMB_IP_X'+str(count): handLms.landmark[mpHands.HandLandmark.THUMB_IP].x,
                        'THUMB_IP_Y'+str(count): handLms.landmark[mpHands.HandLandmark.THUMB_IP].y,
                        'THUMB_IP_Z'+str(count): handLms.landmark[mpHands.HandLandmark.THUMB_IP].z,

                        'THUMB_TIP_X'+str(count): handLms.landmark[mpHands.HandLandmark.THUMB_TIP].x,
                        'THUMB_TIP_Y'+str(count): handLms.landmark[mpHands.HandLandmark.THUMB_TIP].y,
                        'THUMB_TIP_Z'+str(count): handLms.landmark[mpHands.HandLandmark.THUMB_TIP].z,

                        'INDEX_FINGER_MCP_X'+str(count): handLms.landmark[mpHands.HandLandmark.INDEX_FINGER_MCP].x,
                        'INDEX_FINGER_MCP_Y'+str(count): handLms.landmark[mpHands.HandLandmark.INDEX_FINGER_MCP].y,
                        'INDEX_FINGER_MCP_Z'+str(count): handLms.landmark[mpHands.HandLandmark.INDEX_FINGER_MCP].z,

                        'INDEX_FINGER_PIP_X'+str(count): handLms.landmark[mpHands.HandLandmark.INDEX_FINGER_PIP].x,
                        'INDEX_FINGER_PIP_Y'+str(count): handLms.landmark[mpHands.HandLandmark.INDEX_FINGER_PIP].y,
                        'INDEX_FINGER_PIP_Z'+str(count): handLms.landmark[mpHands.HandLandmark.INDEX_FINGER_PIP].z,

                        'INDEX_FINGER_DIP_X'+str(count): handLms.landmark[mpHands.HandLandmark.INDEX_FINGER_DIP].x,
                        'INDEX_FINGER_DIP_Y'+str(count): handLms.landmark[mpHands.HandLandmark.INDEX_FINGER_DIP].y,
                        'INDEX_FINGER_DIP_Z'+str(count): handLms.landmark[mpHands.HandLandmark.INDEX_FINGER_DIP].z,

                        'INDEX_FINGER_TIP_X'+str(count): handLms.landmark[mpHands.HandLandmark.INDEX_FINGER_TIP].x,
                        'INDEX_FINGER_TIP_Y'+str(count): handLms.landmark[mpHands.HandLandmark.INDEX_FINGER_TIP].y,
                        'INDEX_FINGER_TIP_Z'+str(count): handLms.landmark[mpHands.HandLandmark.INDEX_FINGER_TIP].z,

                        'MIDDLE_FINGER_MCP_X'+str(count): handLms.landmark[mpHands.HandLandmark.MIDDLE_FINGER_MCP].x,
                        'MIDDLE_FINGER_MCP_Y'+str(count): handLms.landmark[mpHands.HandLandmark.MIDDLE_FINGER_MCP].y,
                        'MIDDLE_FINGER_MCP_Z'+str(count): handLms.landmark[mpHands.HandLandmark.MIDDLE_FINGER_MCP].z,

                        'MIDDLE_FINGER_PIP_X'+str(count): handLms.landmark[mpHands.HandLandmark.MIDDLE_FINGER_PIP].x,
                        'MIDDLE_FINGER_PIP_Y'+str(count): handLms.landmark[mpHands.HandLandmark.MIDDLE_FINGER_PIP].y,
                        'MIDDLE_FINGER_PIP_Z'+str(count): handLms.landmark[mpHands.HandLandmark.MIDDLE_FINGER_PIP].z,

                        'MIDDLE_FINGER_DIP_X'+str(count): handLms.landmark[mpHands.HandLandmark.MIDDLE_FINGER_DIP].x,
                        'MIDDLE_FINGER_DIP_Y'+str(count): handLms.landmark[mpHands.HandLandmark.MIDDLE_FINGER_DIP].y,
                        'MIDDLE_FINGER_DIP_Z'+str(count): handLms.landmark[mpHands.HandLandmark.MIDDLE_FINGER_DIP].z,

                        'MIDDLE_FINGER_TIP_X'+str(count): handLms.landmark[mpHands.HandLandmark.MIDDLE_FINGER_TIP].x,
                        'MIDDLE_FINGER_TIP_Y'+str(count): handLms.landmark[mpHands.HandLandmark.MIDDLE_FINGER_TIP].y,
                        'MIDDLE_FINGER_TIP_Z'+str(count): handLms.landmark[mpHands.HandLandmark.MIDDLE_FINGER_TIP].z,

                        'RING_FINGER_MCP_X'+str(count): handLms.landmark[mpHands.HandLandmark.RING_FINGER_MCP].x,
                        'RING_FINGER_MCP_Y'+str(count): handLms.landmark[mpHands.HandLandmark.RING_FINGER_MCP].y,
                        'RING_FINGER_MCP_Z'+str(count): handLms.landmark[mpHands.HandLandmark.RING_FINGER_MCP].z,

                        'RING_FINGER_PIP_X'+str(count): handLms.landmark[mpHands.HandLandmark.RING_FINGER_PIP].x,
                        'RING_FINGER_PIP_Y'+str(count): handLms.landmark[mpHands.HandLandmark.RING_FINGER_PIP].y,
                        'RING_FINGER_PIP_Z'+str(count): handLms.landmark[mpHands.HandLandmark.RING_FINGER_PIP].z,

                        'RING_FINGER_DIP_X'+str(count): handLms.landmark[mpHands.HandLandmark.RING_FINGER_DIP].x,
                        'RING_FINGER_DIP_Y'+str(count): handLms.landmark[mpHands.HandLandmark.RING_FINGER_DIP].y,
                        'RING_FINGER_DIP_Z'+str(count): handLms.landmark[mpHands.HandLandmark.RING_FINGER_DIP].z,

                        'RING_FINGER_TIP_X'+str(count): handLms.landmark[mpHands.HandLandmark.RING_FINGER_TIP].x,
                        'RING_FINGER_TIP_Y'+str(count): handLms.landmark[mpHands.HandLandmark.RING_FINGER_TIP].y,
                        'RING_FINGER_TIP_Z'+str(count): handLms.landmark[mpHands.HandLandmark.RING_FINGER_TIP].z,

                        'PINKY_MCP_X'+str(count): handLms.landmark[mpHands.HandLandmark.PINKY_MCP].x,
                        'PINKY_MCP_Y'+str(count): handLms.landmark[mpHands.HandLandmark.PINKY_MCP].y,
                        'PINKY_MCP_Z'+str(count): handLms.landmark[mpHands.HandLandmark.PINKY_MCP].z,

                        'PINKY_PIP_X'+str(count): handLms.landmark[mpHands.HandLandmark.PINKY_PIP].x,
                        'PINKY_PIP_Y'+str(count): handLms.landmark[mpHands.HandLandmark.PINKY_PIP].y,
                        'PINKY_PIP_Z'+str(count): handLms.landmark[mpHands.HandLandmark.PINKY_PIP].z,

                        'PINKY_DIP_X'+str(count): handLms.landmark[mpHands.HandLandmark.PINKY_DIP].x,
                        'PINKY_DIP_Y'+str(count): handLms.landmark[mpHands.HandLandmark.PINKY_DIP].y,
                        'PINKY_DIP_Z'+str(count): handLms.landmark[mpHands.HandLandmark.PINKY_DIP].z,

                        'PINKY_TIP_X'+str(count): handLms.landmark[mpHands.HandLandmark.PINKY_TIP].x,
                        'PINKY_TIP_Y'+str(count): handLms.landmark[mpHands.HandLandmark.PINKY_TIP].y,
                        'PINKY_TIP_Z'+str(count): handLms.landmark[mpHands.HandLandmark.PINKY_TIP].z,

                    }

                    # if image_count % 10 == 0 and image_count != 0:  # makes sure every signs gets 10 data points
                    #     sign_class += 1
                    #     break

                    df.update(data)

        if count >= 40:
            
            df.update({'CLASS': sign_class})
            data_list.append(df)
            print(df['CLASS'])

            if image_count % 10 == 0 and image_count != 0:  # makes sure every signs gets 10 data points
                sign_class += 1

            image_count += 1
            print(df)
            df = {}  # note: this is an empty dictionary, not a dataframe
            start_capture = 0
            count = 0

        cv2.putText(frame, 'Frame: ' + str(int(count)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
        cv2.putText(frame, 'You are on image: ' + str(int(image_count)), (10, 130), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
        cv2.putText(frame, 'You are on sign: ' + string.ascii_lowercase[int(sign_class)], (10, 190), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
        cv2.imshow('Frame', frame)
        key = cv2.waitKey(1)

        if key & 0xFF == ord('q'):
            data = pd.DataFrame(data_list, index_col=0)
            data.to_csv('world_landmark_hands/world_handtracking_data2.csv')
            break

        if key == ord('c'):
            start_capture = 1
            count = 0
            df = {}

        if key == ord('a'):  # if you already have signs and need to add more, start directly at the sign you need to add and then add that in
            existing_data = pd.read_csv('world_landmark_hands/world_handtracking_data2.csv')
            data = pd.DataFrame(data_list)
            print(existing_data['CLASS'])
            new_data = existing_data.append(data)
            print(new_data['CLASS'])
            new_data.to_csv('world_landmark_hands/world_handtracking_data2.csv')
            break

    # Break the loop
    else:
        break

cap.release()
cv2.destroyAllWindows()
