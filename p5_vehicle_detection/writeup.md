**Vehicle Detection Project**

####1. Provide a Writeup / README that includes all the rubric points
    and how you addressed each one.  You can submit your writeup as
    markdown or pdf.
    [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md)
    is a template writeup for this project you can use as a guide and
    a starting point.

You're reading it!

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG
    features from the training images.

The code for this step is contained cell 29 of the IPython notebook. I
started by reading in all the `vehicle` and `non-vehicle` images.
Here is an example of one of each of the `vehicle` and `non-vehicle`
classes:

![][output_images/hog_image.jpg]

I then explored different color spaces and different `skimage.hog()`
parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).
I grabbed random images from each of the two classes and displayed
them to get a feel for what the `skimage.hog()` output looks like.

####2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and the defaults from the
lecture videos returned the best results when used across all channels
with a color transform into YCrCb.

####3. Describe how (and identify where in your code) you trained a
    classifier using your selected HOG features (and color features if
    you used them).

I trained a linear SVM using spatial, binning, and HOG features. This
is shown in cell 30 and results in these features:

![][output_images/car_features.jpg]

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented
    a sliding window search.  How did you decide what scales to search
    and how much to overlap windows?

I re-used the sliding window function from the lecture videos with a
scale tuned by hand. Cars have larger latitudinal features that are
captured well by a window that is 4x the kernel size in the HOG
transform. Likewise, a longitudinal size of 2.5x the kernal did
extremely well in catching features.

####2. Show some examples of test images to demonstrate how your
    pipeline is working.  What did you do to optimize the performance
    of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features
plus spatially binned color and histograms of color in the feature
vector, which provided a nice result.  Here are some example images:

![][output_images/sliding_windows.jpg]

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline
    should perform reasonably well on the entire project video
    (somewhat wobbly or unstable bounding boxes are ok as long as you
    are identifying the vehicles most of the time with minimal false
    positives.)

After extensive testing, I settled on the parameters described
above. The processing takes forever, but it does extremely well, here
is the best result I could muster: [link to my video
result](./final_video.mp4)


####2. Describe how (and identify where in your code) you implemented
    some kind of filter for false positives and some method for
    combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the
video.  From the positive detections I created a heatmap and then
thresholded that map to identify vehicle positions.  I then used
`scipy.ndimage.measurements.label()` to identify individual blobs in
the heatmap.  I then assumed each blob corresponded to a vehicle.  I
constructed bounding boxes to cover the area of each blob detected.

Here's an example result showing the heatmap from a series of frames
of video, the result of `scipy.ndimage.measurements.label()` and the
bounding boxes then overlaid on the last frame of video:

Here is a frame with thresholded bounding boxes and the corresponding
heatmap:

![alt text][output_images/heatmap.jpg]

###Discussion

####1. Briefly discuss any problems / issues you faced in your
    implementation of this project.  Where will your pipeline likely
    fail?  What could you do to make it more robust?

1. Jeez, really need to get the single HOG image created and then
slide windows over that. The overhead of running HOG on every window
is too high, but it's almost drop-dead time for this project, so I'll
leave that for my "spare time."

2. I want to use a convnet to do the vehicle detections. This should
be faster and simpler to implement. With far fewer parameters to tune,
and using TF backend, a Keras model for the hundreds of lines of code
here should be much simpler and faster.

3. Once I found a good set of parameters, the pipeline performed
amazingly. I was kind of stunned. As soon as I started tweaking the
pipeline, something went wrong and the bounding boxes became unstable
again. I removed a centroiding method from the pipeline and have not
implemented it again yet.

