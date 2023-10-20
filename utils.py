import cv2
import numpy as np


def rectContours(contours):
    recCon = []
    for i in contours:
        area = cv2.contourArea(i)
        # print(area)
        if area > 50:
            peri = cv2.arcLength(i, True)
            aprox = cv2.approxPolyDP(i, 0.01 * peri, True)
            if (len(aprox) == 4):
                recCon.append(i)
                # print(area)
    recCon = sorted(recCon, key=cv2.contourArea, reverse=True)
    # print(recCon)
    # print(len(recCon))
    return recCon


def getCournerPoints(contour):
    peri = cv2.arcLength(contour, True)
    aprox = cv2.approxPolyDP(contour, 0.01 * peri, True)
    # print(aprox)
    return aprox


def reorder(myPoints):
    myPoints = myPoints.reshape((4, 2))
    myPointsNew = np.zeros((4, 1, 2), np.int32)
    add = myPoints.sum(1)
    # print(myPoints)
    print(add)
    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[3] = myPoints[np.argmax(add)]
    dif = np.diff(myPoints, axis=1)
    myPointsNew[1] = myPoints[np.argmin(dif)]
    myPointsNew[2] = myPoints[np.argmax(dif)]
    return myPointsNew
