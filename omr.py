import cv2
import numpy as np
import utils
############
widthImg = 900
heightImg = 900
quitions = 5
choises = 5
ans = [1, 2, 0, 1, 4]
###########################

img = cv2.imread("./images/test1.jpg")
img = cv2.resize(img, (widthImg, heightImg))
imgCountours = img.copy()
imgBigestCountours = img.copy()
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

imgblur = cv2.GaussianBlur(imgGray, (3, 3), 1)
imgCanny = cv2.Canny(imgblur, 10, 50)
# imgCanny = cv2.dilate(imgCanny, (3, 3), iterations=1)
contours, _ = cv2.findContours(
    imgCanny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

arcCon = utils.rectContours(contours)
# cv2.drawContours(img, arcCon[4], -1, (255, 0, 255), 20)

biggestContour = utils.getCournerPoints(arcCon[3])
gradePoints = utils.getCournerPoints(arcCon[4])
# print(gradePoints)

if biggestContour.size != 0 and gradePoints.size != 0:
    cv2.drawContours(imgBigestCountours, biggestContour, -1, (0, 0, 255), 20)
    cv2.drawContours(imgBigestCountours, gradePoints, -1, (255, 0, 255), 20)

    biggestContour = utils.reorder(biggestContour)
    gradePoints = utils.reorder(gradePoints)

    pt1 = np.float32(biggestContour)
    pt2 = np.float32([[0, 0], [widthImg, 0], [
                     0, heightImg], [widthImg, heightImg]])
    matrix = cv2.getPerspectiveTransform(pt1, pt2)
    imgWrapColored = cv2.warpPerspective(
        img, matrix, (widthImg-60, heightImg-10))

    ptG1 = np.float32(gradePoints)
    ptG2 = np.float32([[0, 0], [325, 0], [
        0, 150], [325, 150]])
    matrixG = cv2.getPerspectiveTransform(ptG1, ptG2)
    imgGradeDisply = cv2.warpPerspective(img, matrixG, (325, 150))

    imgWrapGray = cv2.cvtColor(imgWrapColored, cv2.COLOR_BGR2GRAY)
    _, imgthersh = cv2.threshold(imgWrapGray, 200, 255, cv2.THRESH_BINARY_INV)

    boxes = utils.splitBoxes(imgthersh)
    # myPixelValue=np.zeros((quitions,choises))
    totalPixel = []
    for image in boxes:
        totalPixel.append(cv2.countNonZero(image))

    myPixelValue = np.reshape(totalPixel, (quitions, choises))
    # print(myPixelValue)
    myIndex = []  # my give answer
    for i in myPixelValue:
        # print(i)
        maxi = np.argmax(i)
        print(maxi)
        myIndex.append(maxi)
    print(myIndex)

cv2.imshow("img", img)
cv2.imshow("imgCanny", imgCanny)
cv2.imshow("imgCountours", imgCountours)
cv2.imshow("imgBigestCountours", imgBigestCountours)
cv2.imshow("imgWrapColored", imgWrapColored)
cv2.imshow("imgGradeDisply", imgGradeDisply)
cv2.imshow("imgthersh", imgthersh)

if cv2.waitKey(0) & 0xFF == ord("q"):
    cv2.destroyAllWindows()
