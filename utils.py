import csv
import cv2
import sklearn
import numpy as np
import matplotlib.image as mpimg

def get_samples(path):
    """get samples from a csv file"""
    csv_path = path + '/driving_log.csv'

    samples = []

    with open(csv_path) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            samples.append(line)

    return samples

def get_sample_image(root_path, samples):
    return cv2.imread(root_path + '/IMG/' + samples[0][0].split('/')[-1])

def get_image_steering(root_path):
    """
    get images and corresponding measurements from given path
    """

    lines = get_samples(root_path)

    images = []
    measurements = []
    for line in lines:
        source_path = line[0]
        steering_value = line[3]
        filename = source_path.split('/')[-1]
        current_path = root_path + '/IMG/' + filename
        image = cv2.imread(current_path)
        images.append(image)
        measurements.append(steering_value)

    return images, measurements

def generator(root_path, samples, batch_size=32, aug_image_fn=lambda x: x, aug_measurement_fn=lambda x: x):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:

                filename_by_col = lambda col: root_path + '/IMG/' + batch_sample[col].split('/')[-1]
                
                center_image = cv2.imread(filename_by_col(0))
                center_angle = float(batch_sample[3])
                
                # steering factor applied to side cameras 
                steering_correction = 0.25

                left_image = cv2.imread(filename_by_col(1))
                left_angle = float(batch_sample[3]) + steering_correction
                
                right_image = cv2.imread(filename_by_col(2))
                right_angle = float(batch_sample[3]) - steering_correction

                for image, angle in zip([center_image, left_image, right_image], [center_angle, left_angle, right_angle]):
                    images.append(aug_image_fn(image))
                    angles.append(aug_measurement_fn(angle))

            X_train = np.array(images)
            y_train = np.array(angles)

            yield sklearn.utils.shuffle(X_train, y_train)
            