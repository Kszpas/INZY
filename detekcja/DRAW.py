import cv2 as cv
import numpy as np


blank = np.zeros((500, 500, 3), dtype='uint8')
cv.imshow("Blank Canvas", blank)

#Pokoloruj obraz na zielono
# blank[200:300, 100:200] = 0, 255, 0
# cv.imshow("Green Canvas", blank)


#write text
cv.putText(blank, "Hello World", (50, 250), cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
cv.imshow("Text", blank)

cv.rectangle(blank, (0,0), (250, 500), (0, 255, 0), thickness=cv.FILLED)
cv.imshow("Rectangle", blank)


cv.rectangle(blank, (0,0), (250, 500), (0, 255, 0), thickness=cv.FILLED)
cv.imshow("Rectangle", blank)

cv.waitKey(0)

