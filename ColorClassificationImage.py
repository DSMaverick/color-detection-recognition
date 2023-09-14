import cv2
import numpy as np
import FeatureExtraction
from api import KNNCassifier
import os
import os.path
import sys

clicked_point = None
terminate_loop = False
prediction = None


def mouse_callback(event, x, y, flags, param):
    global clicked_point, terminate_loop
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_point = (x, y)
        terminate_loop = True


try:
    source_image = cv2.imread(sys.argv[1])
except:
    source_image = cv2.imread('data/tricolor.jpg')

cv2.namedWindow('Color Classifier')
cv2.setMouseCallback('Color Classifier', mouse_callback)

while True:
    display_image = source_image.copy()

    if prediction is not None:
        text = prediction
        text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_PLAIN, 3, 2)

        cv2.putText(
            display_image,
            text,
            (15, 45),
            cv2.FONT_HERSHEY_PLAIN,
            3,
            (0, 0, 0),
            2,
        )

    cv2.imshow('Color Classifier', display_image)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    if clicked_point is not None:
        prediction = None

        b, g, r = source_image[clicked_point[1], clicked_point[0]]
        color = [b, g, r]

        # Perform color prediction
        # Checking whether the training data is ready
        PATH = '.data/training.data'
        if os.path.isfile(PATH) and os.access(PATH, os.R_OK):
            print('Training data ready, Classifier loading.')
        else:
            open('data/training.data', 'w')
            FeatureExtraction.training()

        test_image = np.zeros((1, 1, 3), dtype=np.uint8)
        test_image[0, 0] = color

        # Get the prediction
        FeatureExtraction.color_histogram_of_test_image(
            test_image)
        prediction = KNNCassifier.main('data/training.data', 'data/test.data')
        print('Detected color:', prediction)

        clicked_point = None

cv2.imshow('Color Classifier', display_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
