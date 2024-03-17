import cv2
import numpy as np

# Load pre-trained car detection classifier
car_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_car.xml')

# Function to calculate speed
def calculate_speed(previous_location, current_location, fps, pixels_per_meter):
    # Calculate distance travelled in pixels
    distance_pixels = np.linalg.norm(np.array(current_location) - np.array(previous_location))
    # Convert pixels to meters
    distance_meters = distance_pixels / pixels_per_meter
    # Calculate speed in meters per second
    speed_mps = distance_meters * fps
    # Convert speed to kilometers per hour
    speed_kph = speed_mps * 3.6
    return speed_kph

# Open video file
cap = cv2.VideoCapture('video1.mp4')

# Get FPS from the video file
fps = cap.get(cv2.CAP_PROP_FPS)
pixels_per_meter = 720  # Adjust this value according to your video scale

speed_limit_kph = 80  # Set the speed limit in kilo meters per hour
overspeeding_cars = []  # List to store data of overspeeding cars

# Initialize previous_locations
previous_location = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect cars in the frame
    cars = car_classifier.detectMultiScale(gray, 1.1, 3)

    # Draw rectangles around the detected cars, calculate speed, and mark overspeeding cars
    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        current_location = (x + w/2, y + h)
        if previous_location is not None:
            speed = calculate_speed(previous_location, current_location, fps, pixels_per_meter)
            cv2.putText(frame, f"Speed: {speed:.2f} km/h", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            if speed > speed_limit_kph:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)  # Draw red box around overspeeding car
                overspeeding_cars.append({'Speed': speed, 'Location': (x, y)})

        previous_location = current_location

    # Display the resulting frame
    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture
cap.release()
cv2.destroyAllWindows()

# Save data of overspeeding cars to a file
with open('overspeeding_cars.txt', 'w') as file:
    for car_data in overspeeding_cars:
        file.write(f"Speed: {car_data['Speed']:.2f} km/h, Location: {car_data['Location']}\n")
