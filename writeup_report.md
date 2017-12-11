# **Behavioral Cloning** 

## Writeup

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/Model.png "Model Visualization"
[image2]: ./examples/recover1.jpg "Recovery Image"
[image3]: ./examples/recover2.jpg "Recovery Image"
[image4]: ./examples/recover3.jpg "Recovery Image"
[image5]: ./examples/centerline.jpg "Centerline Driving"
[image6]: ./examples/flipped.jpg "Flipped Image"
[image7]: ./examples/left_drive.jpg "Left Drive Image"
[image8]: ./examples/right_drive.jpg "Right Drive Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results
* additionally, I have included evalModel.py for measuring the accuracy against a test set prior to deployment

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

The model.py file accepts command line arguments denoting the relative sub folder from where to obtain the data samples for training. It presumes the data is stored in ```../data/<sub folder>/``` with the file structure output by the simulator underneath. In this way, you can retarget a different data folder if the accuracy or performance is not as expected.

Further, evalModel.py similarly accepts command line arguments denoting the model to test and the relative sub folder of the data to use for testing. This will give you a gauge for how accurate the model performs in data it has not seen, prior to deployment in the simulator.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model heavily borrows from the [End-to-End model developed by Nvidia](https://arxiv.org/pdf/1704.07911). 
My model consists of a convolution neural network with 5x5 and 3x3 filter sizes and depths between 32 and 108 (model.py lines 81-85) 

The data is normalized in the model using a Keras lambda layer (code line 80). I did not use any activation functions as this is a regression type problem and both positive and negative normalized values are valid.

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 87, 89 & 91). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-14). Using evalModel.py as a quick check, I was able to gauge the accuracy of the model on unseen data before executing the model through the simulator. If the error was too high (> 0.4), it was a strong indication that my hyperparamter adjustment was incorrect and therefore saved me the time of testing the model on the track. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 95).

I had experimented with different dropout rates, however these seemed to have little improvement on the overall performance, and were therefore left at 0.5. Empirically, simply adding the dropout layers seemed to have the desired effect.

I added tunable constants for bias and offset of the measured steering angles (lines 20 and 21). The initial data set I used was generated with a force-feedback steering wheel, which had an inherent bias while turning which was difficult to tune out. The final data used did not require this parameter as it was trained with a mouse. The offset was used for data augmentation using left and right camera images. It was determined empirically that the offset that should be applied to the normalized steering angle is 0.5 (positive bias for left camera images, negative bias for right camera images). In classical PID terminology, this value achieved a slight overshoot type performance, but the model performed well overall.

Because the input images are larger than those used in the Nvidia End-to-End paper (66x200 from the Nvidia example vs 85x320 for my model), I initially added 50% more filters at each level, but ended up increasing the last two as the model seemed to be underfitting (lines 81-85). 

In monitoring the validation loss during training my version of the Nvidia model, it was empirically determined that the model's validation accuracy worsened past 3 epochs.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving and recovering from the left and right sides of the road.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to start with the Nvidia End-to-End driving model, as it was a proven architecture, and adapt it as limitations were found.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I initially used only centre camera images, which resulted in very low training and validation loss, but poor performance on the track. I therefore resorted to augmenting the data with left and right camera images.

I found that my model had a low mean squared error on the training set but a high mean squared error on the validation set when using augmented data. This implied that the model was overfitting. 

To combat the overfitting, I modified the model to include dropout layers before each of the fully connected layers.

Then I found that although the model was no longer overfitting the data, with the augmented set, the loss was unacceptably high (>0.4) for both training and validation sets. This indicated that the model was underfitting. This lead me to increase the number of filters available to the model by 50% at each level, and again at the last two layers. The loss reduced to <0.17.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track just after the bridge and again at the right-hand turn. To improve the driving behavior in these cases, I re-did the recovery driving samples at those locations. At the bend at the end of the bridge, I used less steep driving angles to encourage more straightline driving. At the right-hand turn, I included more recovery driving examples to encourage the model to turn right at that turn.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 78-93) consisted of a convolution neural network with the following layers and layer sizes in the following visualization obtained using Keras visualization tools.

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

Initially, I had used a force-feedback steering wheel, however the bias introduced by the electromechanical system yielded unpredictable results. This led me to switch strategies and use a mouse to gather the data. With the bias eliminated, the model performed markedly better.

Based on the lessons learned from the Nvidia End-to-End driving paper, where their dataset emphasized turning data and less straight line driving data, I initially gathered uniquely recovery data, starting from the left and right sides of the track on turns. A few examples are as follows:

![alt text][image2]
![alt text][image3]
![alt text][image4]

I then recorded two laps on track using center lane driving, one clockwise around the track, the other counterclockwise. Here is an example image of center lane driving:

![alt text][image5]

After the collection process, I had 23,100 data points. I then manually preprocessed this data by removing those entries in the driving_log.csv file that were invalid samples (e.g. steering away from centerline). 

The model.py file preprocessed the data by inverting the images and measurement angle to augment the data set (line 31). An example of a flipped image is seen below. 

![alt text][image6]

It then augmented the data by using left and right images (lines 48 - 66). It automatically sets the steering angle to the respective left and right maxes to avoid saturation of the steering angle, which would yield unpredictable results. Examples of left and right images are as follows:

![alt text][image7]
![alt text][image8]

I finally randomly shuffled the data set and put 20% of the data into a validation set. After removing some of the straight line driving data, the final training set size was therefore 13300 data points. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3 as evidenced by monitoring the validation loss, and confirmed by evaluating the model's performance against unseen data in a test set (derived from driving runs not used in the training set). I used an adam optimizer so that manually training the learning rate wasn't necessary.
