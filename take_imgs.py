import os
import cv2

def takeImages(programme, year_and_sem, tutorial_group, name_of_student, student_id_person):
    source = "0"  # RTSP link or webcam-id
    path_to_save = "Data"  # Replace with the path to save dir
    min_confidence = 0.8
    number_of_images = 100

    # Create the folder structure
    student_folder = os.path.join(path_to_save, f"{programme}_{year_and_sem}_{tutorial_group}_{name_of_student}_{student_id_person}")
    os.makedirs(student_folder, exist_ok=True)
    path_to_save = student_folder

    opencv_dnn_model = cv2.dnn.readNetFromCaffe(
        prototxt="models/deploy.prototxt",
        caffeModel="models/res10_300x300_ssd_iter_140000_fp16.caffemodel"
    )

    if source.isnumeric():
        source = int(source)
    cap = cv2.VideoCapture(source)
    fps = cap.get(cv2.CAP_PROP_FPS)

    count = 0

    while True:
        success, img = cap.read()
        if not success:
            print('[INFO] Cam NOT working!!')
            break

        # Save Image
        if count % int(fps/5) == 0:
            img_name = len(os.listdir(path_to_save))
            cv2.imwrite(f'{path_to_save}/{img_name}.jpg', img)
            print(f'[INFO] Successfully Saved {img_name}.jpg')

            # Display image count on the video feed after capturing
            count_text = f'Image Count: {img_name}' + '/100'
            cv2.putText(
                img, count_text,
                org=(10, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1, color=(0, 255, 0), thickness=2
            )
        count += 1

        # Caffe Model - Face Detection
        h, w, _ = img.shape
        preprocessed_image = cv2.dnn.blobFromImage(
            img, scalefactor=1.0, size=(300, 300),
            mean=(104.0, 117.0, 123.0), swapRB=False, crop=False
        )
        opencv_dnn_model.setInput(preprocessed_image)
        results = opencv_dnn_model.forward()

        for face in results[0][0]:
            face_confidence = face[2]
            if face_confidence > min_confidence:
                bbox = face[3:]
                x1 = int(bbox[0] * w)
                y1 = int(bbox[1] * h)
                x2 = int(bbox[2] * w)
                y2 = int(bbox[3] * h)

                cv2.rectangle(
                    img, pt1=(x1, y1), pt2=(x2, y2),
                    color=(0, 255, 0), thickness=w//200
                )
        cv2.imshow('Webcam', img)
        cv2.waitKey(1)
        if img_name == number_of_images - 1:
            print(f"[INFO] Collected {number_of_images} Images")
            cv2.destroyAllWindows()
            break