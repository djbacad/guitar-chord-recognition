import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras_cv.layers import RandomShear
from tensorflow_addons.optimizers import AdaBelief
from tensorflow.keras.preprocessing import image

# Load the model from the .h5 file
model = tf.keras.models.load_model('models/vision/min_val_loss.h5', 
                                    custom_objects={'RandomShear':RandomShear,
                                                    'AdaBelief':AdaBelief})

# Define your dictionary of class labels and indices
class_labels = {0: 'CMajor', 1: 'DMajor', 2: 'GMajor'}

# Load the video capture device
cap = cv2.VideoCapture('test/test.mp4')

# Create a window to display the video
cv2.namedWindow('Video', cv2.WINDOW_NORMAL)

while True:
    # Read the frame from the video capture device
    ret, frame = cap.read()

    # If we couldn't read the frame, we've reached the end of the video
    if not ret:
        break

    # Resize the frame to (224, 224)
    frame = cv2.resize(frame, (224, 224))

    # Preprocess the frame
    frame_array = image.img_to_array(frame)
    frame_array = np.expand_dims(frame_array, axis=0)
    frame_array = tf.keras.applications.efficientnet_v2.preprocess_input(frame_array)

    # Make a prediction on the frame
    preds = model.predict(frame_array)

    # Get the predicted class label
    predicted_class_index = np.argmax(preds)
    predicted_class_label = class_labels[predicted_class_index]

    # define font properties
    font_size = 1.2
    font_color = (255, 0, 255)  # magenta color
    font_thickness = 2
    font = cv2.FONT_HERSHEY_SIMPLEX

    # draw text on frame
    cv2.putText(frame, str(predicted_class_label), (10, 30), font, font_size, font_color, font_thickness)

    # Show the frame
    cv2.imshow('Video', frame)

    # Check if the user pressed the 'q' key to quit
    if cv2.waitKey(1) == ord('q'):
        break

    # Read the next frame
    ret, frame = cap.read()

# Release the video capture device and close the window
cap.release()
cv2.destroyAllWindows()