import csv
import cv2
import sklearn
import numpy as np

def get_samples(path):
    """get samples from a csv file"""
    csv_path = path + '/driving_log.csv'

    samples = []

    with open(csv_path) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            samples.append(line)

    return samples

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

def generator(root_path, samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = root_path + '/IMG/'+batch_sample[0].split('/')[-1]
                center_image = cv2.imread(name)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)
            