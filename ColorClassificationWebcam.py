import cv2
import FeatureExtraction
from api import KNNCassifier
import os
import os.path

prediction = 'n.a.'

PATH = '.data/training.data'

if os.path.isfile(PATH) and os.access(PATH, os.R_OK):
    print('Training data ready, Classifier loading.')
else:
    print('Training data is being created.')
    open('data/training.data', 'w')
    FeatureExtraction.training()
    print('Training data ready, Classifier loading.')

# Attempt to open the camera by iterating through indexes from 0 to 9
for camera_index in range(10):
    cap = cv2.VideoCapture(camera_index)
    if cap.isOpened():
        print(f'Camera index {camera_index} opened successfully.')
        break
    else:
        print(f'Failed to open camera index {camera_index}.')

while True:
    ret, frame = cap.read()

    if not ret:
        break

    height, width, _ = frame.shape

    cx = int(width / 2)
    cy = int(height / 2)

    frame_size = 200
    frame_x = cx - frame_size // 2
    frame_y = cy - frame_size // 2

    cv2.rectangle(frame, (frame_x, frame_y), (frame_x +
                  frame_size, frame_y + frame_size), (0, 0, 0), 2)

    prediction_frame = frame[frame_y:frame_y +
                             frame_size, frame_x:frame_x + frame_size]

    cv2.putText(
        frame,
        prediction,
        (15, 45),
        cv2.FONT_HERSHEY_PLAIN,
        3,
        (0, 0, 0),
    )

    # Display the resulting frame if it has valid dimensions
    if frame.shape[0] > 0 and frame.shape[1] > 0:
        cv2.imshow(
            'Webcam Color Classifier - Use black square to input data', frame)

    FeatureExtraction.color_histogram_of_test_image(
        prediction_frame)

    prediction = KNNCassifier.main('data/training.data', 'data/test.data')
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
