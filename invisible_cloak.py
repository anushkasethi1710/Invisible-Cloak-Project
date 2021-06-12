# Importing Libraries
import numpy as np
import cv2
import time

# Capturing webcam feed
cap = cv2.VideoCapture(0)

time.sleep(3)

background = 0

# Capturing the static background frame
for i in range(30):
    ret, background = cap.read()

# replace the red pixels ( or undesired area ) with
# background pixels to generate the invisibility feature.

while cap.isOpened():
    # take each frame
    ret, frame = cap.read()
    if not ret:
        break

    # converting bgr to hsv
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 1. Hue: This channel encodes color information. Hue can be
    # thought of an angle where 0 degree corresponds to the red color,
    # 120 degrees corresponds to the green color, and 240 degrees
    # corresponds to the blue color.

    # 2. Saturation: This channel encodes the intensity/purity of color.
    # For example, pink is less saturated than red.

    # 3. Value: This channel encodes the brightness of color.
    # Shading and gloss components of an image appear in this
    # channel reading the videocapture video

    # threshold the hsv value to get only red colors
    lower_red = np.array([0, 120, 70])
    upper_red = np.array([10, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red, upper_red)  # Separating the cloak part

    lower_red = np.array([170, 120, 70])
    upper_red = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv, lower_red, upper_red)

    mask1 = mask1 + mask2  # BITWISE OR

    mask1 = cv2.morphologyEx(mask1, cv2.MORPH_OPEN,
                           np.ones((3, 3), np.uint8), iterations=2)  # Noise Removal
    mask1 = cv2.morphologyEx(mask1, cv2.MORPH_DILATE,
                           np.ones((3, 3), np.uint8), iterations=1)  # Smooth image
    mask2 = cv2.bitwise_not(mask1)  # except the cloak

    res1 = cv2.bitwise_and(background, background, mask=mask1)  # Used fr segmentation of the color
    res2 = cv2.bitwise_and(frame, frame, mask=mask2)  # Used to substitute the cloak part
    final_output = cv2.addWeighted(res1, 1, res2, 1, 0)

    cv2.imshow("cloak", final_output)
    if cv2.waitKey(5) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
