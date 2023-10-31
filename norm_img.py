import os
import cv2
import glob
import numpy as np
import keras
from keras import layers, Sequential
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import os
import pandas as pd
import tensorflow as tf
import pickle
from my_utils import alignment_procedure
from mtcnn import MTCNN
import ArcFace


def normal_img():
    path_to_dir = "data/"
    path_to_save = "norm_data"

    Flage = True
    detector = MTCNN()

    # Checking that, Already All Data Normalized or NOT
    # It will only Normalize, NOT normalized Data
    # If NOT It will create new save Dir
    class_list_update = []
    class_list_dir = []
    class_list_save = []
    if os.path.exists(path_to_save):
        class_list_save = os.listdir(path_to_save)
        class_list_dir = os.listdir(path_to_dir)
        class_list_update = list(set(class_list_dir) ^ set(class_list_save))
    else:
        os.makedirs(path_to_save)

    if len(class_list_update) == 0:
        if len(class_list_dir) == 0 and len(class_list_save) == 0:
            class_list = os.listdir(path_to_dir)
        else:
            if (set(class_list_dir) == set(class_list_save)):
                Flage = False
            else:
                Flage = True
    else:
        class_list = class_list_update

    if Flage:
        class_list = sorted(class_list)
        for name in class_list:
            img_list = glob.glob(os.path.join(path_to_dir, name) + '/*')

            # Create Save Folder
            save_folder = os.path.join(path_to_save, name)
            os.makedirs(save_folder, exist_ok=True)

            for img_path in img_list:
                img = cv2.imread(img_path)

                detections = detector.detect_faces(img)

                if len(detections) > 0:
                    right_eye = detections[0]['keypoints']['right_eye']
                    left_eye = detections[0]['keypoints']['left_eye']
                    bbox = detections[0]['box']
                    norm_img_roi = alignment_procedure(img, left_eye, right_eye, bbox)

                    # Save Norm ROI
                    cv2.imwrite(f'{save_folder}/{os.path.split(img_path)[1]}', norm_img_roi)
                    print(f'[INFO] Successfully Normalised {img_path}')

                else:
                    print(f'[INFO] Not detected Eyes in {img_path}')

            print(f'[INFO] Successfully Normalised All Images from {len(os.listdir(path_to_save))} Classes\n')
        print(f"[INFO] All Normalised Images Saved in '{path_to_save}'")

    else:
        print('[INFO] Already Normalized All Data..')

    # train image    
    path_to_dir = 'norm_data'
    checkpoint_path = 'models/model.h5'
    label_encoder_path = 'models/le.pickle'

    # Load ArcFace Model
    model = ArcFace.loadModel()
    model.load_weights("arcface_weights.h5")
    print("ArcFace expects ", model.layers[0].input_shape[0][1:], " inputs")
    print("and it represents faces as ",
          model.layers[-1].output_shape[1:], " dimensional vectors")
    target_size = model.layers[0].input_shape[0][1:3]
    print('target_size: ', target_size)

    # Variable for store img Embedding
    x = []
    y = []

    names = os.listdir(path_to_dir)
    names = sorted(names)
    class_number = len(names)

    for name in names:
        img_list = glob.glob(os.path.join(path_to_dir, name) + '/*')
        img_list = sorted(img_list)

        for img_path in img_list:
            img = cv2.imread(img_path)
            img_resize = cv2.resize(img, target_size)
            img_pixels = tf.keras.preprocessing.image.img_to_array(img_resize)
            img_pixels = np.expand_dims(img_pixels, axis=0)
            img_norm = img_pixels / 255  # normalize input in [0, 1]
            img_embedding = model.predict(img_norm)[0]

            x.append(img_embedding)
            y.append(name)
            print(f'[INFO] Embedding {img_path}')
        print(f'[INFO] Completed {name} Part')
    print('[INFO] Image Data Embedding Completed...')

    # DataFrame
    df = pd.DataFrame(x, columns=np.arange(512))
    x = df.copy()
    x = x.astype('float64')

    le = LabelEncoder()
    labels = le.fit_transform(y)
    labels = tf.keras.utils.to_categorical(labels, class_number)

    # Train Deep Neural Network
    x_train, x_test, y_train, y_test = train_test_split(x, labels,
                                                        test_size=0.2,
                                                        random_state=0)

    model = Sequential([
        layers.Dense(1024, activation='relu', input_shape=[512]),
        layers.Dense(512, activation='relu'),
        layers.Dense(class_number, activation="softmax")
    ])

    # Model Summary
    print('Model Summary: ', model.summary())

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Add a checkpoint callback to store the checkpoint that has the highest
    # validation accuracy.
    checkpoint = keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                 monitor='val_accuracy',
                                                 verbose=1,
                                                 save_best_only=True,
                                                 mode='max')
    earlystopping = keras.callbacks.EarlyStopping(monitor='val_accuracy',
                                                  patience=20)

    print('[INFO] Model Training Started ...')
    # Start training
    epochs = 100
    batch_size = 16
    history = model.fit(x_train, y_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data=(x_test, y_test),
                        callbacks=[checkpoint, earlystopping])

    print('[INFO] Model Training Completed')
    print(f'[INFO] Model Successfully Saved in /{checkpoint_path}')

    # save label encoder
    f = open(label_encoder_path, "wb")
    f.write(pickle.dumps(le))
    f.close()
    print('[INFO] Successfully Saved models/le.pickle')
