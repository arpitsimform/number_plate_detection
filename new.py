import cv2
import os
import datetime
import requests
import csv
from ultralytics import YOLO

# Start webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# Model
model = YOLO("/home/arpit/Public/number_plate/model.pt")

# Object classes
classNames = ["License_Plate"]

# Create a CSV file for storing detected number plates
csv_file_path = 'detected_number_plates.csv'
with open(csv_file_path, 'w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['Timestamp', 'Number Plate'])

while True:
    # Capture a frame from the webcam
    success, img = cap.read()

    # Display the current frame
    cv2.imshow('Webcam', img)

    # Check for key press 'd' to detect number plates and save to CSV
    key = cv2.waitKey(1) & 0xFF
    if key == ord('d'):
        # Detect license plates in the frame
        results = model(img, stream=True)

        # Process detected license plates
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # Convert to int values

                # Draw bounding box on the image
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Extract the detected license plate region
                cropped_plate = img[y1:y2, x1:x2]

                # Send the cropped plate to PlateRecognizer API
                try:
                    _, image_jpg = cv2.imencode('.jpg', cropped_plate)
                    files = {'upload': ('image.jpg', image_jpg.tobytes(), 'image/jpeg')}
                    headers = {'Authorization': 'Token 2d2adf172fa4712f27a8ec6dc6db7f353369b8a6'}  # Replace with your Plate Recognizer API key
                    response = requests.post(
                        'https://api.platerecognizer.com/v1/plate-reader/',
                        headers=headers,
                        files=files
                    )
                    number_plate = response.json()['results'][0]['plate']
                    print("Detected number plate:", number_plate)

                    # Save the timestamp and number plate in the CSV file
                    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    with open(csv_file_path, 'a', newline='') as csv_file:
                        csv_writer = csv.writer(csv_file)
                        csv_writer.writerow([timestamp, number_plate])

                except requests.RequestException as req_err:
                    print(f"Error in API request: {req_err}")
                except Exception as e:
                    print(f"Error processing image or API response: {e}")

    # Check for key press 'q' to exit
    elif key == ord('q'):
        break

# Release the webcam
cap.release()
cv2.destroyAllWindows()
