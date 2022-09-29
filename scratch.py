import numpy as np
import cv2

img = np.ones((500, 500))
img = cv2.putText(img, "Hello World", (1,1), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), thickness=2)
print(img)
while True:
    cv2.imshow("image", img)
    cv2.waitKey(0)
