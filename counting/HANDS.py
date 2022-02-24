import mediapipe as mp
import cv2
import numpy as np
import uuid
import os

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mp_drawing_styles = mp.solutions.drawing_styles

#stock-images
folder_path = "num_images"
list_of_images= os.listdir(folder_path)
overlayimg = []
for imgs in list_of_images:
    als_image = cv2.imread(f'{folder_path}/{imgs}')
    overlayimg.append(als_image)
print(len(overlayimg))
#webcam feed
cap = cv2.VideoCapture(0)

with mp_hands.Hands(
    static_image_mode=True,
    model_complexity=0,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.5) as hands:

    while cap.isOpened():
        ret, frame = cap.read()

        image = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)



        #render results
        pos_list = []
        if results.multi_hand_landmarks:
            for num, hand in enumerate(results.multi_hand_landmarks):
                mp_drawing.draw_landmarks(
                    image,
                    hand,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

        lmList = []
        if results.multi_hand_landmarks:
            myHand = results.multi_hand_landmarks[0]
            for id, lm in enumerate(myHand.landmark):
                # print(id, lm)
                h, w, c = image.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                # print(id, cx, cy)
                lmList.append([id, cx, cy])
        tipIds = [4, 8, 12, 16, 20]
        flist = []
        if len(lmList) != 0:

            # number fingers opened

            if lmList[tipIds[0]][1] > lmList[tipIds[0] - 3][1]:
                flist.append(1)
            else:
                flist.append(0)

            for id in range(1,5):
                if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]:
                    flist.append(1)
                else:
                    flist.append(0)

        totalFingers = flist.count(1)

        if totalFingers == 4:
            h, w, c = overlayimg[4].shape
            image[0:h, 0:w] = overlayimg[4]
            cv2.rectangle(image, (20, 225), (170, 425), (0, 255, 0), cv2.FILLED)
            cv2.putText(image, str(5), (45, 375), cv2.FONT_HERSHEY_PLAIN,
                        10, (255, 0, 0), 25)
        else:
            h, w, c = overlayimg[totalFingers-2].shape
            image[0:h, 0:w] = overlayimg[totalFingers-2]

            cv2.rectangle(image, (20, 225), (170, 425), (0, 255, 0), cv2.FILLED)
            cv2.putText(image, str(totalFingers), (45, 375), cv2.FONT_HERSHEY_PLAIN,
                        10, (255, 0, 0), 25)







        cv2.imshow("Hand",image)
        print(totalFingers)

        if cv2.waitKey(10) & 0xFF == ord('e'):
            break




cap.release()
cv2.destroyAllWindows()

#cases

'''
1.no finger-> A,S,T,M,N,O
2.two finger case->G,H,K,L,R,U,V,Y,P
3. one finger case->D,I,X,Z
4. more than 2 -> B,W,C,E,F
'''
