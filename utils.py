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

    # print(add)
    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[3] = myPoints[np.argmax(add)]
    dif = np.diff(myPoints, axis=1)
    # print(dif)
    myPointsNew[1] = myPoints[np.argmin(dif)]
    myPointsNew[2] = myPoints[np.argmax(dif)]

    return myPointsNew


def splitBoxes(img, qustions=5, choises=5):
    rows = np.vsplit(img, qustions)
    boxes = []
    for r in rows:

        coloums = np.hsplit(r, choises)
        for box in coloums:
            boxes.append(box)

    return boxes


# def markAnswers(img, quitionsCo, choisesCo, myAns, Answers, widthImg, heightImg):
#     rowHeight = heightImg/quitionsCo
#     colWidth = widthImg/choisesCo

    # for row in range(quitionsCo):
    #     ans = Answers[row]
    #     myans = myAns[row]

    #     if ans == myans:
    #         cv2.circle(img, (int(ans*colWidth + (colWidth/2)),
    #                          int(row * rowHeight + (rowHeight/2))), 30, (0, 255, 0), -1)

    #     else:
    #         cv2.circle(img, (int(ans*colWidth + (colWidth/2)),
    #                          int(row * rowHeight + (rowHeight/2))), 15, (0, 255, 0), -1)

    #         cv2.circle(img, (int(myans*colWidth + (colWidth/2)),
    #                          int(row * rowHeight + (rowHeight/2))), 30, (0, 0, 255), -1)


def showAnswers(img, myIndex, grading, ans, quitions, choices):
    secW = int(img.shape[1]/quitions)
    secH = int(img.shape[0]/choices)

    for x in range(0, quitions):
        myAns = myIndex[x]
        cX = (myAns*secW)+secW//2
        cY = (x*secH) + secH//2

        if (grading[x] == 1):
            mycolor = (0, 255, 0)
        else:
            mycolor = (0, 0, 255)
            correctAns = ans[x]
            cv2.circle(img, ((correctAns*secW)+secW//2, (x*secH) +
                       secH//2), 20, (0, 255, 0), cv2.FILLED)

        cv2.circle(img, (cX, cY), 50, mycolor, cv2.FILLED)

    return img
