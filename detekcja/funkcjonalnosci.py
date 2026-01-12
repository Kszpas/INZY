import cv2 as cv

img = cv.imread('szachownica.jpg.png')
cv.imshow("Szachy", img)

#Konwersja do skali szarosci
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow("Gray", gray)

#blur
blur = cv.GaussianBlur(img, (3,3), cv.BORDER_DEFAULT)
cv.imshow("Blur", blur)







cv.waitKey(0)