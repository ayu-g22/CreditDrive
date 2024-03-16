import cv2
import os
import numpy as np

# Global variables
WINDOW_NAME = "Camera video"
CASCADE_FILE = "haar_xml_07_19.xml"

def detect_and_display(frame, nr, cascade_classifier=None):
    global traffic_template

    # Detect traffic lights through cascade classifier
    trLights = cascade_classifier.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=0, minSize=(24, 24))

    # Copy frame to traffic_template for further processing
    traffic_template = frame.copy()

    for i, (x, y, w, h) in enumerate(trLights):
        # Extract the region of interest
        trLightROI = traffic_template[y:y+h, x:x+w]

        # Save the region of interest
        cv2.imwrite(os.path.join(f"haar{nr}.jpg"), trLightROI)

        # Apply color map and detect circles
        detect_circles(trLightROI, nr)

def detect_circles(traffic_template, nr):
    # Apply color map to search for certain color
    resultImg = cv2.applyColorMap(traffic_template, cv2.COLORMAP_SUMMER)
    resultImg = cv2.inRange(resultImg, np.array([0, 90, 90]), np.array([204, 255, 255]))
    resultImg = cv2.GaussianBlur(resultImg, (9, 9), 0.5)

    # Detect circles using Hough transform
    circles = cv2.HoughCircles(resultImg, cv2.HOUGH_GRADIENT, 2, 90, param1=50, param2=20, minRadius=4, maxRadius=10)

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            # Draw the circle and save it
            cv2.circle(traffic_template, (x, y), r, (0, 0, 255), 3)
            cv2.imwrite((f"circle{nr}.jpg"),traffic_template)

def main():
    # Create folders for detected traffic lights
    os.makedirs('./', exist_ok=True)
    os.makedirs('./', exist_ok=True)

    # Load cascade classifier
    cascade_classifier = cv2.CascadeClassifier(CASCADE_FILE)

    # Open video capture
    capture = cv2.VideoCapture(0)

    if not capture.isOpened():
        print("--(!)Error opening video capture")
        return

    nr = 0
    while True:
        ret, frame = capture.read()
        if not ret:
            print("End of video.")
            break

        detect_and_display(frame, nr)
        nr += 1

    capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
