import cv2 as cv


capture = cv.VideoCapture(1)

scale = 1.0

def zoom_frame(frame, scale):
    h, w = frame.shape[:2]
    center_x, center_y = w // 2, h // 2
    new_w, new_h = int(w / scale), int(h / scale)
    x1, y1 = center_x - new_w // 2, center_y - new_h // 2
    x2, y2 = center_x + new_w // 2, center_y + new_h // 2
    cropped = frame[y1:y2, x1:x2]
    return cv.resize(cropped, (w, h))

def mouse(event, x, y, flags, param):
    global scale
    if event == cv.EVENT_MOUSEWHEEL:
        if flags > 0 and scale < 5:  # zoom in
            scale += 0.1
        elif flags < 0 and scale > 1:  # zoom out
            scale -= 0.1
        print(f"Zoom: {scale:.1f}x")

cv.namedWindow("Zoom")
cv.setMouseCallback("Zoom", mouse)

while True:
    ret, frame = capture.read()
    if not ret:
        break

    zoomed = zoom_frame(frame, scale)
    cv.imshow("Zoom", zoomed)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break


# while True:
#     isTrue, frame = capture.read()
#     cv.imshow("Rozgrywka", frame)
#     if cv.waitKey(20) & 0xFF==ord('d'):
#         break
# capture.release()
# cv.destroyAllWindows()



#Live Video scaling
# def changeRes(width, height):
#     capture.set(3, width)
#     capture.set(4, height)
# changeRes(1280, 720)
    
#Skalowanie ramki już istniejącej
# def rescaleFrame(frame, scale=0.75):
#     width = int(frame.shape[1] * scale)
#     height = int(frame.shape[0] * scale)
#     dimensions = (width, height)
#     return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

#Wczytywanie obrazu
# img = cv.imread('szachownica.jpg.png')
# cv.imshow("Szachy", img)
# cv.waitKey(1000)

#Wczytywanie wideo, videocapture 0 dla wbudowanej kamery
