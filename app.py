from flask import Flask, request, jsonify
import cv2
import datetime
import requests
import csv
from ultralytics import YOLO
import numpy as np

# app = Flask(__name__)

# # Model initialization
# model = YOLO("/home/arpit/Public/number_plate/model.pt")

# # CSV file setup
# csv_file_path = 'detected_number_plates.csv'
# with open(csv_file_path, 'w', newline='') as csv_file:
#     csv_writer = csv.writer(csv_file)
#     csv_writer.writerow(['Timestamp', 'Number Plate'])

# @app.route('/detect_license_plate', methods=['POST'])
# def detect_license_plate():
#     try:
#         # Read image from the request
#         image_data = request.files['image'].read()
#         nparr = np.frombuffer(image_data, np.uint8)
#         img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

#         # Detect license plates in the frame
#         results = model(img, stream=True)

#         # Process detected license plates
#         for r in results:
#             boxes = r.boxes
#             for box in boxes:
#                 x1, y1, x2, y2 = map(int, box.xyxy[0])  # Convert to integers

#                 # Extract the detected license plate region
#                 cropped_plate = img[y1:y2, x1:x2]

#                 # Send the cropped plate to PlateRecognizer API
#                 _, image_jpg = cv2.imencode('.jpg', cropped_plate)
#                 files = {'upload': ('image.jpg', image_jpg.tobytes(), 'image/jpeg')}
#                 headers = {'Authorization': 'Token 2d2adf172fa4712f27a8ec6dc6db7f353369b8a6'}  # Replace with your Plate Recognizer API key
#                 response = requests.post(
#                     'https://api.platerecognizer.com/v1/plate-reader/',
#                     headers=headers,
#                     files=files
#                 )
#                 number_plate = response.json()['results'][0]['plate']
#                 print("Detected number plate:", number_plate)

#                 # Save the timestamp and number plate in the CSV file
#                 timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#                 with open(csv_file_path, 'a', newline='') as csv_file:
#                     csv_writer = csv.writer(csv_file)
#                     csv_writer.writerow([timestamp, number_plate])

#         return jsonify({'License plate detected and recognized successfully': number_plate})

#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

# if __name__ == '__main__':
#     app.run(debug=True)

from flask import Flask, request, jsonify
import cv2
import datetime
import requests
import csv
from ultralytics import YOLO

app = Flask(__name__)

# Model initialization
model = YOLO("/home/arpit/Public/number_plate/model.pt")

# CSV file setup
csv_file_path = 'detected_number_plates.csv'
with open(csv_file_path, 'w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['Timestamp', 'Number Plate'])

@app.route('/detect_license_plates', methods=['POST'])
def detect_license_plates():
    try:
        # Read image from the request
        image_data = request.files['image'].read()
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Detect license plates in the frame
        results = model(img, stream=True)

        detected_plates = []

        # Process detected license plates
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Convert to integers

                # Extract the detected license plate region
                cropped_plate = img[y1:y2, x1:x2]

                # Send the cropped plate to PlateRecognizer API
                _, image_jpg = cv2.imencode('.jpg', cropped_plate)
                files = {'upload': ('image.jpg', image_jpg.tobytes(), 'image/jpeg')}
                headers = {'Authorization': 'Token 2d2adf172fa4712f27a8ec6dc6db7f353369b8a6'}  # Replace with your Plate Recognizer API key
                response = requests.post(
                    'https://api.platerecognizer.com/v1/plate-reader/',
                    headers=headers,
                    files=files
                )

                results = response.json().get('results', [])

                # Extract all detected plates from the response
                for result in results:
                    number_plate = result.get('plate', '')
                    detected_plates.append(number_plate)
                    print("Detected number plate:", number_plate)

        # Save the timestamp and all detected number plates in the CSV file
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(csv_file_path, 'a', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            for plate in detected_plates:
                csv_writer.writerow([timestamp, plate])

        return jsonify({'detected_plates': detected_plates})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
