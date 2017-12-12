# Vehicle Detection Project

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.jpg
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video_final.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

* You're reading it!
* [vehicle_detection.ipynb](https://github.com/WangYuanMike/CarND-Vehicle-Detection/blob/master/vehicle_detection.ipynb) is the Jupyter notebook for this vehicle detection and tracking project. All example images can be checked in the output of this jupyter notebook.

### Feature extraction and classifier training

#### 1. Load data
I used the vehicle and non-vehicle dataset provided in lecture 40 Tips and Tricks for the Project as the training and test data. All of the them are 64x64 pixels, and the number of vehicle images and non-vehicle images are roughly the same. I also use pickle to dump them as two .p files, so that the data loading time could be effectively reduced when I need to rerun the whole jupyter notebook. Plus, since all of the images are in format png, I rescaled them by 255 to make sure the training features would be consistent with the inference features.
* non_vehicle_images (8968, 64, 64, 3)
* vehicle_images (8792, 64, 64, 3)

#### 2. Extract HOG, color spatial, and color histogram features
I used the recommended parameter values in the lecture to retrieve the HOG and color histogram features. For the color spatial features, I changed the resize size from (32, 32) to (16, 16) after several round of testing, because it could reduce the number of color spatial features to 768, so that HOG features (1764) would be the majority for training and predicting. I have tried out several color spaces, e.g. YCrCb, but all of them did not show a convincing better result than RGB. Therefore, I decide to choose RGB as the color space to extract all features. Finally I combined all the features and used StandardScalor() to scale them.

| parameter | value |
|:---------:|:-----:|
| HOG orient    | 9     |
| HOG pixels_per_cell | 8|
| HOG cell_per_block | 2 |
| color spatial size| (16, 16) |
| color histogram bins | 32 |
| HOG features | (7, 7, 2, 2, 9) = 1764 |
| color spatial features | 768 |
| color histogram features | 96 |
| total number of features | 2628 |
| color space for all features | RGB |

#### 3. Train classifier
First of all, data set (including both vechicle and non-vehicle images) has been splitted into training set and test set.
* Train set size: 14208
* Test set size: 3552

Then I trained a LinearSVC() as suggested from the lecture. The test accuracy is higher than 98%, and it took about 7 seconds to train the model. But later I found it has problem in detecting black cars in the test images, especially when the sliding window size is big, e.g. 96, or 128, which finally causes unstable tracking rectangle in the final video generation. Therefore, I decided to change the model to SVC() with default kernel and gamma values and a smaller C value (0.5) to prevent overfitting. The test accuracy of SVC() model is higher than 99%, and it becomes much better in detecting black cars. In the meantime, the detection of white cars by the new model works pretty fine as well. The cons of the SVC() model is that it took more than 120 seconds to train, and the predict time is also much longer than the linear one.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

First I implemented a function named find_car_single_image() to get box list of a image. Then I tested 6 parameter combinations:
* pix_per_cell=8(window=64), cells_per_step=1
* pix_per_cell=8(window=64), cells_per_step=2
* pix_per_cell=12(window=96), cells_per_step=1
* pix_per_cell=12(window=96), cells_per_step=2
* pix_per_cell=16(window=128), cells_per_step=1
* pix_per_cell=16(window=128), cells_per_step=2
And from the output images, the best one is **pix_per_cell=12(window=96), cells_per_step=1**. I did not use mutiple scale to search, because the SVC() model is pretty slow on my laptop. Plus, the window size I chose is fine to track small cars(cars which are far away), and for big cars(cars which are near) it will draw multiple boxes which finally form a good heatmap of the car.

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

The examples of test images can be checked in the jupyter notebook, and the optimization process can be checked above.

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video_final.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

The corresponding coding cells are called "Sliding window to search vehicles", "Use heatmap to draw bound box to track vehicle", and "Video Generation". **process() in class VideoProcessor** is the main method for video generation. In this method, I first get box list of the current frame by sliding window search, then I add half of box list history into the box list of the current frame (kind of using exponentially average to make the bounding box more stable), then I used heatmap (with a threshold of 2) and label function to detect and draw bounding box on cars. And finally I used the box list, which contains the exponential average history of box list, to update the history variable of the instance.

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

* The main issue is that SVC() model is too slow to make prediction, therefore I have to give up multi-scale window search on my laptop. Maybe I can still use LinearSVC() model and some data augmentation technic to mitigate the black car detection problem.
* The other issue is that the bounding box is still not stable enough, also occasionally false positive appears. I am interested in more advanced approach to stablize bounding box with history.

