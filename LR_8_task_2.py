import cv2
import numpy as np

img = cv2.imread("shcherback.jpg")
kernel = np.ones((5, 5), np.uint8)
# Перетворення зображення у відтінки сірого
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Застосування гауссівського розмиття
imgBlur = cv2.GaussianBlur(imgGray, (7, 7), 0)
# Застосування алгоритму Canny для виявлення границь
imgCanny = cv2.Canny(img, 150, 200)
# Збільшення областей границь на зображенні за допомогою операції діляції 
imgDialation = cv2.dilate(imgCanny, kernel, iterations=1)
# Зменшення областей границь за допомогою операції ерозії
imgEroded = cv2.erode(imgDialation, kernel, iterations=1)

cv2.imshow("Gray Image", imgGray)
cv2.imshow("Blur Image", imgBlur)
cv2.imshow("Canny Image", imgCanny)
cv2.imshow("Dialation Image", imgDialation)
cv2.imshow("Eroded Image", imgEroded)
cv2.waitKey(0)
