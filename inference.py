import time
from keras.models import load_model
from mtcnn import MTCNN
from my_utils import alignment_procedure
import tensorflow as tf
import ArcFace
import cv2
import numpy as np
import pandas as pd
import pickle
import csv
import os
import datetime

# Define the base folder path where program-specific folders will be created
base_folder = 'Attendance_records'

# Define a list of program names
program_names = ["RDS", "REI", "RWS", "RSD"]

# Initialize sets to store recognized student IDs for each program
recognized_ids = {program: set() for program in program_names}

# Initialize a dictionary to store the CSV file paths for each program
csv_file_paths = {program: '' for program in program_names}

# Function to generate the CSV file name with today's date
def get_csv_file_name():
    today_date = datetime.date.today()
    # Format the date as 'dd/mm/yyyy' and replace slashes with underscores for the file name
    date_str = today_date.strftime("%d/%m/%Y").replace('/', '_')
    return f'attendanceList-{date_str}.csv'

def initialize_program_folders():
    # Create program-specific folders if they don't exist
    for program in recognized_ids.keys():
        program_folder = os.path.join(base_folder, program)
        os.makedirs(program_folder, exist_ok=True)

        # Get the CSV file name with today's date
        csv_file_name = get_csv_file_name()

        # Define the CSV file path for this program
        csv_file_path = os.path.join(program_folder, csv_file_name)
        csv_file_paths[program] = csv_file_path

        # Create the CSV file and write headers if it doesn't exist
        if not os.path.isfile(csv_file_path):
            with open(csv_file_path, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["Name", "Student ID", "Programme", "Tutorial Group", "Year and Semester","current_time"])
        else:
            # Read existing records from the CSV file and populate the recognized_ids set
            with open(csv_file_path, 'r') as file:
                reader = csv.reader(file)
                # next(reader)  # Skip header row
                for row in reader:
                    recognized_ids[program].add(row[1])  # student ID is in the second column (index 1)

def recognize_attendance():
    source = "0"  # Path to Video or webcam
    path_saved_model = 'models/model.h5'  # Path to saved .h5 model
    threshold = 0.80  # Min prediction confidence (0<conf<1)

    # Liveness Model
    liveness_model_path = 'models/liveness.model'
    label_encoder_path = 'models/le.pickle'

    if source.isnumeric():
        source = int(source)

    # Load saved FaceRecognition Model
    face_rec_model = load_model(path_saved_model, compile=True)

    # Load MTCNN
    detector = MTCNN()

    # Load ArcFace Model
    arcface_model = ArcFace.loadModel()
    target_size = arcface_model.layers[0].input_shape[0][1:3]

    # Liveness Model
    liveness_model = tf.keras.models.load_model(liveness_model_path)
    label_encoder = pickle.loads(open(label_encoder_path, "rb").read())

    cap = cv2.VideoCapture(source)

    start_time = None
    face_detected_time = None

    while True:
        success, img = cap.read()
        if not success:
            print('[INFO] Error with Camera')
            break

        detections = detector.detect_faces(img)
        if len(detections) > 0:
            for detect in detections:

                bbox = detect['box']
                xmin, ymin, xmax, ymax = int(bbox[0]), int(bbox[1]), \
                    int(bbox[2] + bbox[0]), int(bbox[3] + bbox[1])

                # Liveness
                img_roi = img[ymin:ymax, xmin:xmax]
                face_resize = cv2.resize(img_roi, (32, 32))
                face_norm = face_resize.astype("float") / 255.0
                face_array = tf.keras.preprocessing.image.img_to_array(face_norm)
                face_prepro = np.expand_dims(face_array, axis=0)
                preds_liveness = liveness_model.predict(face_prepro)[0]
                decision = np.argmax(preds_liveness)

                if decision == 0:
                    # Show Decision
                    cv2.rectangle(
                        img, (xmin, ymin), (xmax, ymax),
                        (0, 0, 255), 2
                    )
                    cv2.putText(
                        img, 'Liveness: Fake',
                        (xmin, ymin - 10), cv2.FONT_HERSHEY_PLAIN,
                        2, (0, 0, 255), 2
                    )

                    # Reset the timer if a face is detected as fake
                    face_detected_time = None

                else:
                    # Real
                    right_eye = detect['keypoints']['right_eye']
                    left_eye = detect['keypoints']['left_eye']

                    norm_img_roi = alignment_procedure(img, left_eye, right_eye, bbox)
                    img_resize = cv2.resize(norm_img_roi, target_size)
                    img_pixels = tf.keras.preprocessing.image.img_to_array(img_resize)
                    img_pixels = np.expand_dims(img_pixels, axis=0)
                    img_norm = img_pixels / 255  # normalize input in [0, 1]
                    img_embedding = arcface_model.predict(img_norm)[0]

                    data = pd.DataFrame([img_embedding], columns=np.arange(512))

                    predict = face_rec_model.predict(data)[0]
                    if max(predict) > threshold:
                        class_id = predict.argmax()
                        pose_class = label_encoder.classes_[class_id]
                        color = (0 ,255, 0)

                    else:
                        pose_class = 'Unknown Person'
                        color = (0, 0, 255)

                    # Show Result
                    cv2.rectangle(
                        img, (xmin, ymin), (xmax, ymax),
                        color, 2
                    )
                    cv2.putText(
                        img, f'{pose_class}',
                        (xmin, ymin - 10), cv2.FONT_HERSHEY_PLAIN,
                        2, (255, 0, 255), 2
                    )

                    # Check if it's time to display the attendance message
                    if start_time is None:
                        start_time = time.time()
                    elif time.time() - start_time >= 3:
                        # Check if a face has been continuously detected for 5 seconds
                        if face_detected_time is None:
                            face_detected_time = time.time()
                        elif time.time() - face_detected_time >= 3:

                            # Split the student information into individual details
                            name, student_id, programme, tutorial_group, year_and_sem = pose_class.split('_')

                            # Get the CSV file path for this program
                            csv_path = csv_file_paths[programme]

                            if student_id not in recognized_ids[programme]:
                                # Get the current timestamp
                                current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                                # Append the recognized person's details to the CSV file along with the timestamp
                                with open(csv_path, 'a', newline='') as file:
                                    writer = csv.writer(file)
                                    writer.writerow(
                                        [name, student_id, programme, tutorial_group, year_and_sem, current_time])

                                # Add the student ID to the list of recognized IDs for this program
                                recognized_ids[programme].add(student_id)

                                cv2.putText(
                                    img, "Attendance was taken!",
                                    (10, 40), cv2.FONT_HERSHEY_PLAIN,
                                    2, (0, 255, 0), 2
                                )
                            else:
                                cv2.putText(
                                    img, "Attendance is existing!",
                                    (10, 40), cv2.FONT_HERSHEY_PLAIN,
                                    2, (255, 0, 0), 2
                                )

        else:
            print('[INFO] Eyes Not Detected!!')

            # Reset the timer if no face is detected
            start_time = None
            face_detected_time = None

        cv2.imshow('Output Image', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

    print('[INFO] Inference on Videostream is Ended...')

# Initialize program-specific folders and recognized_ids sets
initialize_program_folders()

