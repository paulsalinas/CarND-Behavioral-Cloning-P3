
# Behavioral Cloning Project

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[sample]: ./images/sample.png "sample"
[flipped]: ./images/flipped.png "flipped"
[brightness]: ./images/brightness.png "brightness"
[translated]: ./images/translated.png "translation"
[pre1]: ./images/preprocessed1.png "pre1"
[pre2]: ./images/preprocessed2.png "pre2"
[pre0]: ./images/preprocessed0.png "pre0"
[pre3]: ./images/preprocessed3.png "pre3"


## Rubric Points

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* utils.py containing the generator used for keras generator and some utility functions that helped with processing the csv file and image augmentation
* run1.mp4 video of the successful run 
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md (the file you just opened!) summarizing the results 

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py and utils.py file contains the code for training and saving the convolution neural network. The model.py shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works. The utils.py contains the generator and a bunch of utility functions that help with csv processing and image processing.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model architecture was the Nvidia pipeline with some dropout layers in between. Keras code is very clean and clearly indicates the layers in the architecture:

~~~

model = Sequential()

# preprocessing layer
model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
model.add(Lambda(normalize))

model.add(Convolution2D(24, 5, 5, subsample=(2, 2),  activation='relu'))
model.add(Convolution2D(36, 5, 5, subsample=(2, 2),  activation='relu'))
model.add(Convolution2D(48, 5, 5, subsample=(2, 2),  activation='relu'))
model.add(Dropout(0.5))

model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Dropout(0.5))

model.add(Flatten())

model.add(Dense(100))
model.add(Dropout(0.5))

model.add(Dense(50))

model.add(Dropout(0.5))
model.add(Dense(10))

model.add(Dropout(0.5))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
~~~

I used dropout layers inbetween to prevent overfitting and I used a RELU activation for the convolution layers.

This pipeline was very successful the Nvidia trial which makes it a perfect candidate for this problem. For this problem, the network might be a bit excessive and it might be a good exercise to see how 'successful' this model would be with a few layers removed.

#### 2. Attempts to reduce overfitting in the model

As seen in the code snippet in the section above, I've placed dropout layers to prevent overfitting. I also used 20% of the sample data as validation data for the model (model.py line 20). 
#### 3. Model parameter tuning

As shown again in the code snippet above, the architecture used the adam optimizer and thus the learning rate was not tuned manually.

#### 4. Appropriate training data

For this, I used the Udacity sample data and made sure to use all available camera images in addition to a set of augmentations that will be described below.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

I tried to continually incrementally improve the architecture and data given the training loss vs. validation loss and also the driving performance.

I first started with a simple Lenet architecture with just the centre camera and the car drove very poorly.

I then added the Nvidia Pipeline as suggested in course and I found the driving improved drastically as in the driving was alot smoother but still drove off the road when it a turn. This indicated that I may not have enough 'turning' or 'swaying' data in my training set.

From here, I decided it was time to use the left/right cameras to help with the turning behavior. For the left camera data, I added a steering of +0.25 and for the right camera I used -0.25. This further helped the model with keeping straight and the early bend of track 1 but when it came to tight turns towards the end of the track the model failed. 

I also noticed my model can use some dropout layers to prevent the model overfitting the data. This helped with my testing loss being better than my validation loss.

I knew data augmentation was the next step, especially with dealing with the tighter turns.

The augmentation of data was the key to successfully finishing the course, and I will describe that in further detail below. The augmented data allowed my model to finish track 1.

#### 2. Final Model Architecture

The final model architecture (model.py lines 48-75) was the Nvidia pipeline and was described in the code snippet above 

#### 3. Creation of the Training Set & Training Process

As I mentioned above, I used all camera angles with a factor applied to right and left camera angles.

For each of those data, I flipped each image and inverted the steering.

For each of those images I described above, I applied a random brightness, translation, and random flipping augmentation to generate randomly augmented images.

Here's an example of a sample image:

![sample][sample]

I then flipped the image and inverted the steering.

![flipped][flipped]

Then I applied a set of augmentations to generate even more data:

First I applied a random brightness augmentation:

![brightness][brightness]

Second a random vertical and horizontal translation. Vertical translation to mimic the vehicle veering off and a vertical translation to mimic road steepness. For the horizontal translation, I adusted the steering based on the translation. At first, this did not help the model get around tight turns but I increased the adjustment until I obtained the desired result

![translated][translated]

Thirdly, I randomly flipped the images. Doing this 4 times gives us additional training results:


![pre0][pre0]
![pre1][pre1]
![pre2][pre2]
![pre3][pre3]

By doing this, I ended up with 115,704 samples.

Further, I normalized the data (lambda x: x / 255 - 0.) and cropped the top portion and bottom portion of the images to reduce some noise as a result of the car hood and the horizon.

I found that the ideal number of epochs in this case was 5 and I found the results worsened when I tried a larger number like 10. 
