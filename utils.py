import csv
import cv2
import sklearn
import numpy as np
import matplotlib.image as mpimg

def flip_randomly(image, steer):
    """
     will randomly flip an image and it's steering 
    """
    ind_flip = np.random.randint(2)
    if ind_flip==0:
        flipped_image = cv2.flip(image,1)
        flipped_steer = -steer
        return flipped_image, flipped_steer
    else:
        return image, steer

def flip(image, steer):
    flipped_image = cv2.flip(image,1)
    flipped_steer = -steer
    return flipped_image, flipped_steer

def get_samples(path):
    """get samples from a csv file"""
    csv_path = path + '/driving_log.csv'

    samples = []

    with open(csv_path) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            samples.append(line)

    return samples

def split_camera_angles(samples, angle_adjustment):
    """
    get pairs of image names and measurements for center, left, and right cameras
    """
    centre = []
    left = []
    right = [] 

    for line in samples:
        steering = float(line[3])
        centre.append((line[0], steering))
        left.append((line[1], steering + angle_adjustment))
        right.append((line[2], steering - angle_adjustment))

    return centre, left, right



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

def generator(root_path, samples, batch_size=24, aug=False, num_aug=5):
    """
    generator that will flip the image and apply a certain number of augmentations based on 'num_aug'
    """
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:

                filename_by_col = lambda col: root_path + '/IMG/' + batch_sample[col].split('/')[-1]
                
                image = cv2.imread(filename_by_col(0))
                angle = float(batch_sample[1])
                
                images.append(image)
                angles.append(angle)

                flipped_image, flipped_angle = flip(image, angle)
                images.append(flipped_image)
                angles.append(flipped_angle)

                if aug:
                    for aug_i  in range(num_aug):
                        x, y = None, None
                        keep_pr = 0
                        x, y = preprocess_aug(image, angle)
                        images.append(x)
                        angles.append(y)
            
            X_train = np.array(images)
            y_train = np.array(angles)

            yield sklearn.utils.shuffle(X_train, y_train)

def preprocess_aug(image, steering):
    aug_image = augment_brightness_camera_images(image)
    aug_image, aug_steering = flip_randomly(aug_image, steering)
    aug_image, aug_steering = trans_image(aug_image, aug_steering, 150)
    return aug_image, aug_steering

def augment_brightness_camera_images(image):
    """
    courtesy of: 
    https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9#.bi5td84fk
    adjust brightness 
    """
    image1 = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    image1 = np.array(image1, dtype = np.float64)
    random_bright = .5 + np.random.uniform()
    image1[:,:,2] = image1[:,:,2] * random_bright
    image1[:,:,2][image1[:,:,2]>255]  = 255
    image1 = np.array(image1, dtype = np.uint8)
    image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
    return image1


def trans_image(image,steer,trans_range):
    """
    courtesy of: 
    https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9#.bi5td84fk
    translate image horizontally and vertically while applying a steering adjustment based on
    the horizontal translation
    """
    rows,cols,channels = image.shape

    # Translation
    tr_x = trans_range * np.random.uniform() - trans_range / 2
    steer_ang = steer + tr_x  / trans_range * 2 * .8
    tr_y = 40 * np.random.uniform() - 40 / 2
    
    #tr_y = 0
    Trans_M = np.float32([[1,0,tr_x],[0,1,tr_y]])
    image_tr = cv2.warpAffine(image,Trans_M,(cols,rows))
    
    return image_tr,steer_ang
