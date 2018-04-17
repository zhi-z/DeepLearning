**Machine Learning Engineer Nanodegree** 

# Capstone Project 

![img](file:///C:/Users/JH/AppData/Local/Temp/msohtmlclip1/01/clip_image001.gif)

MichaelVirgo 

May 4, 2017 

# Lane Detection with DeepLearning 

![img](file:///C:/Users/JH/AppData/Local/Temp/msohtmlclip1/01/clip_image001.gif)

## Project Overview 

One of the most common tasks while driving,although likely overlooked due to its constant use when a human drives a car,is keeping the car in its lane. As long as a person is not distracted,inebriated, or otherwise incapacitated, most people can do this after basictraining. However, what comes very simply to a person – keeping the car betweenits lane’s lines – is a much harder problem for a computer to solve.  

Why is this complicated for a computer? Tobegin with, a computer does not inherently understand what the yellow and whitestreaks on a road are, the shifts in pixel values between those and the pixelsrepresenting the road in a video feed. One way to help a computer learn to atleast detect these lines or the lanes itself is through various computer visiontechniques, including camera calibration (removing the distortion inherent tothe camera used), color and gradient thresholds (areas of the image wherecertain colors or changes in color are concentrated), perspectivetransformation (similar to obtaining a bird’s-eye view of the road), and more.As part of the first term of the separate Self-Driving Car Nanodegree program,I was tasked with using some of these different computer vision techniques thatrequire a decent amount of manual input and selection to arrive at the endresult (see my Advanced Lane Lines project[ ](https://github.com/mvirgo/Advanced-Lane-Lines)[here](https://github.com/mvirgo/Advanced-Lane-Lines)[)](https://github.com/mvirgo/Advanced-Lane-Lines). 

Withthe knowledge gained from the Machine Learning Nanodegree the Deep Learningcourse on Udacity's website, I wondered if there might be a better approach tothis problem - one directly involving deep learning. Deep learning involvesutilizing multiple-layered neural networks, which use mathematical propertiesto minimize losses from predictions vs. actuals to converge toward a finalmodel, effectively learning as they train on data. 

### Why thismatters 

Youmay say that a fully autonomous vehicle might not necessarily need to directlyidentify the lane lines - it might otherwise just learn that there areboundaries it is not meant to cross, but not see them as much different fromother types of boundaries. If we're skipping straight to a fully autonomousvehicle, this may be true. However, for many consumers, they will likely see more step-by-step changes, and showing them (potentially on anin-vehicle screen) that the car can always sense the lanes will go a long wayin getting them comfortable with a computer doing the driving for them. Evenshort of this, enhanced lane detection could alert an inattentive driver whenthey drift from their lane. 

## Problem Statement 

Human beings, when fully attentive, do quitewell at identifying lane line markings under most driving conditions. Computersare not inherently good at doing the same. However, humans have a disadvantageof not always being attentive (whether it be because of changing the radio,talking to another passenger, being tired, under the influence, etc.), while acomputer is not subject to this downfall. As such, if we can train a computerto get as good as a human at detecting lane lines, since it is already significantlybetter at paying attention full-time, the computer can take over this job fromthe human driver. Using deep learning, I will train a model that can that ismore robust, and faster, than the original computer vision-based model. Themodel will be based off a neural network architecture called a “convolutional”neural network, or “CNN” for short, which are known to perform well on imagedata. This is a great architecture candidate since I will feed the model framesfrom driving videos in order to train it. CNNs work well with images as theylook first for patterns at the pixel level (groups of pixels around eachother), progressing to larger and larger patterns in more expanded areas of theimage. 

## Evaluation Metrics 

As will be explained further in the Analysissection, my initial approach in the project was to teach a CNN to calculate thepolynomial coefficients of the lane lines, and then draw the lane area basedoff of those lines. This approach, similar to a regression-type problem, madesense to use mean-squared error to minimize loss, meaning the differencebetween the actual coefficients and the model’s prediction (MSE uses the meanof all the squared differences to calculate loss). The final approach I usedalso utilized MSE, as I used a fully convolutional neural network (i.e. onethat lacks any fully connected layers) to generate the green lane to be drawnonto the original image. Using MSE here meant minimizing the loss between thepredicted pixel values of the output lane image and what the lane image labelwas. I will also evaluate it directly against my original pure computervision-based model in both accuracy and speed, as well as on even morechallenging videos than my CV-based model was capable of doing. 

## Analysis 

### Datasets and Inputs 

The datasets I used for the project areimage frames from driving video I took from my smartphone. The videos werefilmed in 720p in horizontal/landscape mode, with 720 pixels on the y-axis and1280 pixels on the x-axis, at 30 fps. In order to cut down on training time,the training images were scaled down to 80 by 160 pixels (a slightly differentaspect ratio than the beginning, primarily as it made it easier for appropriatecalculations when going deeper in the final CNN architecture). In order to calculatethe original labels, which were six coefficients (three for each lane line,with each of the three being a coefficient for a polynomial-fit lane line), Ialso had to do a few basic computer vision techniques first. I had to performimage calibration with OpenCV to correct for my camera’s inherent distortion,and then use perspective transformation to put the road lines on a flatplane.  

Initially, I wanted to make the model morerobust my original model by drawing over the lane lines in the image, which canbe blurry or fade away into the rest of the image the further to the back ofthe image it is. I drew over 1,420 perspective transformed road images in red,and ran a binary color threshold for red whereby the output image would showwhite wherever there had been sufficient red values and no activation (black)where the red values were too low. With this, I re-ran my original model,modified to output only the six coefficients instead of the lane drawing, sothat I could train the network based on those coefficients as labels. 

However, I soon found that the amount ofdata I had was not sufficient, especially because most of the images were offairly straight lanes. So, I went back and ran the original CV model over allmy videos from roads that were very curvy. I also added in a limited amount ofimages from the regular project video (I wanted to save the challenge video fora true test after finalizing my model) from Udacity’s SDC Nanodegree AdvancedLane Lines project I previously completed, so that the model could learn someof the distortion from a different camera. However, this introduced acomplication to my dataset – Udacity’s video needed a different perspective transformation,which has a massive effect on the re-drawn lane. I will come back to this issuelater. 

I ended upobtaining a mix of both straight lines and various curved lines, as well asvarious conditions (night vs. day, shadows, rain vs. sunshine) in order to helpwith the model's overall generalization. These will help to cover more of thereal conditions that drivers see every day. I will discuss more of the imagestatistics later regarding total training images used, but have provided twocharts below regarding the percentage breakouts of road conditions for thoseobtained from my own driving video collected. 

| ![img](file:///C:/Users/JH/AppData/Local/Temp/msohtmlclip1/01/clip_image002.gif) Straight   ![img](file:///C:/Users/JH/AppData/Local/Temp/msohtmlclip1/01/clip_image003.gif) | Curvature   ![img](file:///C:/Users/JH/AppData/Local/Temp/msohtmlclip1/01/clip_image004.gif) | Very Curvy | Weather & Time         ![img](file:///C:/Users/JH/AppData/Local/Temp/msohtmlclip1/01/clip_image005.gif) Clear Night ![img](file:///C:/Users/JH/AppData/Local/Temp/msohtmlclip1/01/clip_image006.gif)Cloudy  Afternoon |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ---------- | ------------------------------------------------------------ |
|                                                              |                                                              |            |                                                              |

 

I noted before that one issue with theoriginal videos collected was that too much of the data came from straightlanes, which is not apparent from the above charts – although “Very Curvy” madeup 43% of the original dataset, which I initially believed would be sufficient,I soon found out the breakout was terribly centered around straight, as can beseen in the below chart from one of the coefficient labels’ distributions. Thisis a definite problem to be solved. 

![img](file:///C:/Users/JH/AppData/Local/Temp/msohtmlclip1/01/clip_image009.gif) CoefficientLabel 

### Algorithms and Techniques 

First, I must extract the image frames fromvideos I obtain. Next, I must process the images for use in making labels, forwhich I will use the same techniques I previously used in my computervision-based model. Next, I will calibrate for my camera’s distortion by taking pictures of chessboard images with the samecamera the video was obtained with, and undistort the image using “cv2.findChessboardCorners” and “cv2.calibrateCamera”. With the image now undistorted, I will find good source points(corners of the lane lines at the bottom and near the horizon line) anddetermine good destination points (where the image gets transformed out to) toperspective transform the image (mostly by making educated guesses at whatwould work best. When I have these points, I can use “cv2.getPerspectiveTransform” to get a transformation matrix and “cv2.warpPerspective” to warp the image to a bird’s eye-like view of the road. 

From here, in order to enhance the model’s robustness in areas with less than clear lane lines, I drew redlines over the lane lines in each image used. On these images, I will use abinary thresholding on areas of the image with high red values so that thereturned image only has values where the red lane line was drawn. Then,histograms will be computed using where the highest amount of pixels fallvertically (since the image has been perspective transformed, straight lanelines will appear essentially perfectly vertical) that split out from themiddle of the image so that the program will look for a high point on the leftand a separate high point on the right. Sliding windows that search for morebinary activation going up the image will then be used to attempt to follow theline. 

Based off the detected pixels from thesliding windows, “numpy.polyfit” will be used to return polynomial functions that are most closelyfit to the lane line as possible (using a polynomial allows for it to trackcurved lines as well as straight). This function actually returns the threecoefficients of the “ax^2+bx+c” equation, where “a”, “b” and “c” are the coefficients. I will append the total of six coefficients(three for each of the two lane lines) to a list to use as labels for training.

![img](file:///C:/Users/JH/AppData/Local/Temp/msohtmlclip1/01/clip_image010.jpg)**Initial Model**                   However, prior to training,I will want to check whether the labels are even accurate at all. Using thelabels from above, I can feed an image through the original undistortion andperspective transformation, create an image “blank”with “numpy.zeros_like”, make lane points from the polynomial coefficients by calculatingthe full polynomial fit equation from above for each line, and then use “cv2.fillPoly”with those to create a lane drawing. Using “cv2.warpPerspective” with the inverse of my perspective transformation matrix calculatedbefore, I can revert this lane drawing back to the space of the original image,and then use “cv2.addWeighted” to merge the lane drawing with the original image. This way, I canmake sure I feed accurate labels to the model. 

Lastly, myproject will use Keras with TensorFlow backend in order to create aconvolutional neural network. Using “keras.models.Sequential”, 

I can create theneural network with convolutional layers 

(keras.layers.Convolution2D)and fully-connected layers 

(keras.layers.Dense). I will first try amodel architecture similar to the one at left, which I used successfully in aprevious project for Behavioral Cloning. 

### Benchmark  

I plan tocompare the results of the CNN versus the output of the computer vision-basedmodel 

I used in my SDCNanodegree project (linked to above). Note that because there is not a “ground-truth” for my data, I cannot directly compare to that model from aloss/accuracy perspective, but as the end result is very visual, I will see whichmodel produces the better result. Part of this comes down to robustness – my pure CV model failed to produce lane lines past the first fewseconds of a Challenge video in my previous project. If this model can mostlysucceed on the Challenge video (i.e. no more than a few seconds without thelane shown) without having been specifically trained on images from that video,it will have exceeded this benchmark. A second benchmark will be the speed ofthe model – the CV-based model canonly generate roughly 4.5 frames per second, which compared to 30 fps videoincoming is much slower than real-time. The model will exceed this benchmark ifthe writing of the video exceeds 4.5 fps. 

## Methodology 

### Data Preprocessing  

Some of the general techniques I used to preprocessmy data to create labels are discussed above in the “Algorithms and Techniques”section. However, there was a lot more to making sure my model got sufficientquality data for training. First, after loading all the images from each frameof video I took, an immediate problem popped up. Where I had purposefullygathered night video and rainy video, both of these severely cut down on thequality of images. In the night video, where my smartphone camera already wasof slightly lesser quality, I was also driving on the highway, meaning a muchbumpier video, leading to blurry images. I sorted through each and every singlegathered image to check for quality, and ended up removing roughly one-third ofmy initial image data from further usage. 

From here, I also wondered whether the modelmight overfit itself by getting a sneak peek at its own validation data ifimages were too similar to each other – at 30 frames per second, there is not a whole lot of change fromone frame to the next. I decided to only use one out of every ten images fortraining. With these, I drew over the images in red as mentioned above, andthen ran my programs for making labels and checking the labeled images. Here, Ifound that process for making the labels was flawed – the original code for my sliding windows failed completely oncurves. This was because the initial code, when it hit the side of an image,would keep searching straight up the image – causing the polynomial line to think it should go up the image too.By fixing my code to end when it hit the side of the image, I vastly improvedits usefulness on curves. After re-checking my labels and tossing out some badones, I moved on to check the distribution of my labels. It was bad – there were still hardly any big curves! Even the videos from curvyroads still had mostly straight lines. So, I re-ran my process over one inevery five images, only from the four videos with mostly curved lines.  

I also decided at this point to add a littlebit of the Udacity basic project video from my CVbased project (not from theChallenge video though because I wanted that as the true test of the model’sability to generalize) in order to train the model for different distortion(which I had already obtained previously). However, this caused a new issue – these images needed a different perspective as well. This was notso hard for creating the labels, but I knew it could cause an issue on the tailend, as checking the labels would require the specific inverse perspectivetransformation to re-draw the lines. 

As shown previously “Datasets and Inputs”section, after adding in this additional data, my distribution was still fairlyunequal. However, I found by using the histograms of the distributions of eachlabel, I could find where the exact values were where only a limited amount ofimages fell. By iterating through each of the labels, and finding whichtraining images were on the fringes of the data, I could then come back andgenerate “new” data; this data was just rotations of these images outside themain distribution, but my CNN would likely become much more likely to notoverfit to straight lines. 

**Improving the Distribution of Lane Labels** 

![img](file:///C:/Users/JH/AppData/Local/Temp/msohtmlclip1/01/clip_image012.gif)

​                                           CoefficientLabel                                                                           CoefficientLabel 

 

The changes in the distribution of imagelabels for the second coefficients are shown above. I originally normalized thelabels with sklearn’s “StandardScaler” (causing the differences in values above),which improved training results but also did need to be reversed after trainingto return the correct label. 

At this point, the approach depending on themodel diverge. In my initial models, I took in either a perspective transformedimage or regular road image, downscaled it from 720x1280x3 to 45x80x3 (scalingdown 16X), gray-scaled the image, added back a third dimension (cv2.cvtColorremoves the dimension when gray-scaling but Keras wants it to be able to runproperly), and then normalized the image (new_image = (new_image / 255) * .8 -1) to be closer to a mean of zero and standard deviation of one, which is keyin machine learning (the algorithms tend to converge better). 

On the flipside, once I changed to a fully convolutional model, I instead was onlydown-sizing the road images to 80x160x3 (a slightly different aspect ratio thanthe original but roughly 8X scaled down), without any further gray-scaling ornormalization (I instead on my Batch Normalization layer in Keras to helpthere). Additionally, since a fully convolutional model essentially returnsanother image as output, instead of saving down my lane labels as numbers only,I also saved down the generated lane drawings prior to merging them with theroad image. These new lane image labels were to be the true labels of my data.I still used my image rotations to generate additional data, but also rotatedthe lane image labels along with the training image (based on the distributionsof the original label coefficients still). I also added in a horizontal flip ofeach image and corresponding lane image label to double my dataset fortraining. For these new lane image labels, I dropped off the ‘R’ and ‘B’ colorchannels as the lane was being drawn in green, hoping to make training moreefficient with less of an output to generate. 

### Image Statistics 

Here are some statistics from my datapre-processing: 

x    21,054 total images gathered from 12 videos (a mix ofdifferent times of day, weather, traffic, and road curvatures –see previous pie chart breakout) 

x      The roads also contain difficult areas such asconstruction and intersections x     14,235 of the total thatwere usable of those gathered (due to blurriness, hidden lines, etc.) x      1,420 total images originally extracted from those toaccount for time series (1 in 10) x  227 of the 1,420 unusable due tothe limits of the CV-based model used to label (down from 446 due to variousimprovements made to the original model) for a total of 1,193 images 

x    Another 568 images (of 1,636 pulled in) gathered frommore curvy lines to assist in gaining a wider distribution of labels (1 inevery 5 from the more curved-lane videos; from 8,187 frames) 

​        x    In total, 1,761 original images 

x    I pulled in the easier project video from Udacity'sAdvanced Lane Lines project (to help the model learn an additional camera'sdistortion) - of 1,252 frames, I used 1 in 5 for 250 total, 217 of which wereusable for training 

x      A total of 1,978 actual images used between mycollections and the one Udacity video x   After checking histograms for eachcoefficient of each label for distribution, I created an additional 4,404images using small rotations of the images outside the very center of theoriginal distribution of images. This was done in three rounds of slowly movingoutward from the center of the data (so those further out from the center ofthe distribution were rotated multiple times). 6,382 images existed at thispoint. 

x    Finally, I added horizontal flips of each and everyroad image and its corresponding label, which doubled the total images. All inall, there were a total of 12,764 images for training. 

### Implementation 

My first CNNbuilt used perspective transformed images as input, and can be seen in the 

[“perspect_NN.py” file](https://github.com/mvirgo/MLND-Capstone/blob/master/perspect_NN.py)[.](https://github.com/mvirgo/MLND-Capstone/blob/master/perspect_NN.py) It used batch normalization, fourconvolutional layers with a shrinking number of filters, followed by a poolinglayer, flatten layer, and four fully-connected layers, with the finalfully-connected layer having six outputs – the six coefficient labels of the lane lines.Each layer used RELU activation, or rectified linear units, as this activationhas been found to be faster and more effective than other activations. I triedsome of the other activations as well just in case, but found RELU to be themost effective, as expected. Also, in order to help prevent overfitting andincrease robustness, I added in strategic dropout to layers with the mostconnections, and also used Keras’s ImageDataGenerator to add in imageaugmentation like more rotations, vertical flips, and horizontal shifts. Ioriginally used mean-squared-error for loss, but found that mean-absolute erroractually produced a model that had could handle more variety in curves. Notethat I also made the training data and labels into arrays before feeding themodel as it works with Keras. Also, I shuffled the data to make sure that thedifferent videos were better represented and the model would not just overfiton certain videos. Last up was splitting into training and validation sets so Icould check how the model was performing. 

**PerspectiveImage****’****s Histogram and Sliding Windows** 

![img](file:///C:/Users/JH/AppData/Local/Temp/msohtmlclip1/01/clip_image013.jpg) 

After training thisfirst model and creating a function to actually see the re-drawn lanes, I foundthis first model to be moderately effective, given that you were using the sameperspective transformation from the original model (see video [here](https://youtu.be/ZZAgcSqAU0I)[)](https://youtu.be/ZZAgcSqAU0I). However, my end goal was to ignore the need to perspective transforman image for the CNN altogether, so after finding that the first model wasmoderately effective at producing a re-drawn lane, I shifted course. In [“](https://github.com/mvirgo/MLND-Capstone/blob/master/road_NN.py)[road_NN.py](https://github.com/mvirgo/MLND-Capstone/blob/master/road_NN.py)[”](https://github.com/mvirgo/MLND-Capstone/blob/master/road_NN.py)[,](https://github.com/mvirgo/MLND-Capstone/blob/master/road_NN.py) this second model is included. Other thanfeeding in a regular road image, the only change I made to this model wasadding a Crop layer, whereby the top third of the image was removed (I playedaround with one half or one third without much difference). I found quicklythat the CNN could, in fact, learn the lane coefficients without perspectivetransformation, and the resulting model was actually a little bit moreeffective even (see video [here](https://www.youtube.com/watch?v=Vq0vlKdyXnI)[)](https://www.youtube.com/watch?v=Vq0vlKdyXnI). 

#### Good Prediction vs. Poor Prediction 

![img](file:///C:/Users/JH/AppData/Local/Temp/msohtmlclip1/01/clip_image014.jpg) 

Therewas still one big problem – if my model was predicting the lane labelcoefficients, this meant that the lines still need to be drawn in aperspective-transformed space, and reverted to the road space, even if theoriginal image was not perspective-transformed. This caused massive issues ingeneralizing to new data – I would need the transformation matrix of anynew data (even slight changes in camera mounting would cause a need for a newmatrix). 

### Refinement 

Myfirst thought was whether or not I could actually look directly at theactivation of the convolutional layers to see what the layer was looking at. Iassumed that if the CNN was able to determine the appropriate linecoefficients, it was probably activating over the actual lines of lane, or atleast some similar area in the image that would teach it the values to predict.

After someresearch, I found the [keras](https://github.com/raghakot/keras-vis)[-](https://github.com/raghakot/keras-vis)[vis](https://github.com/raghakot/keras-vis)[[1\]](#_ftn1)library to be great for looking at the activation of each layer. This library can actually look at the classactivation maps (in my case the “classes” are actually each of the coefficient labelssince this is not a classification problem) in each layer. I thought I hadfound my solution, until I looked at the activation maps themselves. 

#### Class Activation Maps 

![img](file:///C:/Users/JH/AppData/Local/Temp/msohtmlclip1/01/clip_image016.jpg) 

Whilethe above activation maps of the first few layers look okay, these wereactually some of the clearest I could find. Interestingly enough, the CNNactually often learned by looking at *onlyone lane line* – it was calculating the position of the otherline based off of the one it looked at. But that was only the case for curves – forstraight lines, it was not activating on the lane lines at all! It was actuallyactivating directly on the road in front of the car itself, and deactivatingover the lane lines. As a result, I realized the model was activating indifferent ways for different situations, which would make using the activationmaps directly almost impossible. Also, notice in the above second image thatthe non-cropped part of the sky is also being activated (the dark portion) – due to thevarious rotations and flips, the model was also activating in areas that wastelling it top from bottom. Other activation maps also activated over the carat the bottom of the image for the same purpose. 

Ialso briefly tinkered with trying to improve the activation maps above by usingtransfer learning. Given that in my Behavioral Cloning project, the car neededto stay on the road, I figured it had potentially learned a similar, butperhaps more effective, activation. Also, I had tens of thousands of images totrain on for that project, so the model was already more robust. After using “model.pop” on thatmodel to remove the final fully-connected layer (which had only one output forthat project), I added a new fully-connected layer with six outputs. Then, Itrained the already-established model further on my real road images (the oldmodel was trained on simulated images), and actually found that it did a betterjob on looking at both lines, but still failed to have a consistent activationI could potentially use to re-draw lines more accurately. 

Atthis point, I began to consider what I had read on image segmentation,especially [SegNet](http://mi.eng.cam.ac.uk/projects/segnet/#research)[[2\]](#_ftn2),which was specifically designedto separate different components of a road out in an output image by using afully convolutional neural network. This approach was different from mine inthat a *fully* convolutional neuralnetwork does not have any fully-connected layers (with many more connectionsbetween them), but only uses convolutional layers followed by deconvolutionallayers to essentially make a whole new image. I realized I could skip the 

Lane Detection with Deep Learning 

undoing of the perspective transformationfor the lane label entirely but actually training directly to the lane drawingas the output. By doing so, it meant that even if the camera was mounteddifferently, the lanes were spaced differently, etc., the model would still beable to return an accurate drawing of the predicted lane. **Results** 

#### The Final Model 

Although I hadmade a CNN previously that ended in fully-connected layers, I had never beforemade a fully convolutional neural network, and there were some challenges ingetting the underlying math to work for my layers. Unlike in the forward passin normal Convolution layers, Keras’s Deconvolution layers flip around the backpropagation of the neuralnetwork to face the opposite way, and therefore need to be more carefullycurated to arrive at the correct size (including the need to specify the outputsize). I chose to make my new model a mirror of itself, with Convolutionallayers and Pooling in slowly decreasing in size layers, with the midpointswitching to Upsampling (reverse-pooling) and Deconvolution layers of the samedimensions. The final deconvolution layer ends with one filter, which isbecause I only wanted a returned image in the ‘G’ color channel, as Iwas drawing my predicted lanes in green (it later is stacked up with zeroed-out‘R’ and ‘B’ channels to merge with the original road image). Choosing to input80x160x3 images (smaller images were substantially less accurate in theiroutput, likely due to the model being unable to identify the lane off in thedistance very well) without grayscaling (which tended to hide yellow lines onlight pavement), I also normalized the incoming labels by just dividing by 255(such that the labels were from 0 to 1 for ‘G’ pixel values).  

The final model iswithin the [“](https://github.com/mvirgo/MLND-Capstone/blob/master/fully_conv_NN.py)[fully_conv_NN.py](https://github.com/mvirgo/MLND-Capstone/blob/master/fully_conv_NN.py)[”file](https://github.com/mvirgo/MLND-Capstone/blob/master/fully_conv_NN.py)[.](https://github.com/mvirgo/MLND-Capstone/blob/master/fully_conv_NN.py) I stuck with RELU activation and some of theother convolution parameters (strides of (1,1) and ‘valid’ paddinghad performed the best) from my prior models, but also added more extensivedropout. I had wanted to use dropout on every Convolutional and Deconvolutionallayer, but found it used up more memory than I had. I also tried to use BatchNormalization prior to each layer but found it also used up too much memory,and instead I settled for just using it at the beginning. A more interestingdiscovery, given that using MSE for loss had previously failed, was that itperformed much better than any other loss function with this new model. Alsointriguing was that adding *any* typeof image augmentation with ImageDataGenerator, whether it be rotations, flips,channel shifts, shifts along either the horizontal or vertical axes, etc., didnot result in a more robust model, and often had worse results on any test images I looked at. Typically, Iexpect the image augmentation to improve the final model, but in this case,skipping any augmentation (although I kept the generator in anyway without it,as it is good practice) lead to a better 

model.Channel shifts helped with shadows, but worsened overall performance. This isfed into the “draw_det[ected_lanes.py](https://github.com/mvirgo/MLND-Capstone/blob/master/draw_detected_lanes.py)[”file](https://github.com/mvirgo/MLND-Capstone/blob/master/draw_detected_lanes.py)[, in which th](https://github.com/mvirgo/MLND-Capstone/blob/master/draw_detected_lanes.py)e model predicts the lane, it is averaged overfive frames (to account for any odd predictions), and then merges with theoriginal road image from a video frame.

![img](file:///C:/Users/JH/AppData/Local/Temp/msohtmlclip1/01/clip_image017.jpg) 

#### Evaluation andValidation 

After 20 epochs, my model finished with MSEfor training of 0.0046 and validation of 0.0048, which was significantly lowerthan any previous model’s Ihad tried (although a bit of apples and oranges against the models using sixpolynomial coefficients as labels). I first tried the trained model against oneof my own videos, one of the hilly and curved roads for which the model hadpotentially seen up to 20% of the images for, although likely much less – from the image statistics earlier, I had to throw out a largeportion of the images from these videos, so even though I ran it on one in fiveimages, the model probably only saw 5-10% of them. Fascinatingly, the modelperformed great across the entire image, only losing the right side of the laneat one point when the line became completely obscured by leaves. The modelactually performed near perfectly even on a lot of the areas I knew I hadpreviously had to throw out, because my CVbased model could not appropriatelymake labels for them. The output video can be seen [here](https://youtu.be/bTMwF1UoZ68)[.](https://youtu.be/bTMwF1UoZ68) 

### Justification 

However,the fact remained that the model had in fact seen some of those images. Whatabout trying it on the [Challenge video](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/challenge_video.mp4)[[3\]](#_ftn3)created by Udacity for the Advanced Lane Lines project? It had never been trained on a single frame of thatvideo. Outside of a small hiccup going under the overpass in the video, themodel performed great, with a little bit of noise on the right side where theseparated lane lines were. It had passed my first benchmark –outperforming my CVbased model, which had failed on this video. This video canbe seen [here](https://youtu.be/_qwET69bYa8)[.](https://youtu.be/_qwET69bYa8) 

Mysecond benchmark was with regards to speed, and especially when including GPUacceleration, the deep learning model crushed the earlier model – itgenerated lane line videos at between 25-29 fps, far greater than the 4.5 fpsfor the CV model. Even without GPU acceleration, it still averaged 5.5 fps,still beating out the CV model. Clearly, GPU acceleration is key in unlockingthe potential of this model, running almost real-time with 30 fps video. Withregards to both robustness and speed, the deep learning-based model is adefinite improvement on the usual CV-based techniques. 

## Conclusion 

### More Visualizations 

Below I have included some additional visualizations, comparing thevarious stages of my own model as well as in comparison to my original modelusing typical computer vision techniques. **Improving Models** 

![img](file:///C:/Users/JH/AppData/Local/Temp/msohtmlclip1/01/clip_image022.gif)

 

The CV-basedmodel believed both lines to be on the right side of the lane, hence only afaint line and not a full lane drawn. Some of this comes down to weaknesses inthe algorithm there, which lacked checks to see whether the lanes were separatefrom each other. 

 

**Computer Vision TechniquesModel vs. Deep Learning Model Udacity Challenge Video Output** 

![img](file:///C:/Users/JH/AppData/Local/Temp/msohtmlclip1/01/clip_image025.gif)

### Reflection 

My project began with collecting drivingvideo, which I then extracted the individual frames from. After curating thedata to get rid of various blurry or other potentially confusing images, Icalculated the calibration needed to undistort my images, and perspectivetransformed them to be able to calculate the lines. After additional imageprocessing to improve the dataset at hand, I then created six coefficientlabels, three each for both lane lines. Next, I created a program to make thoselabels into re-drawn lanes, and then had to improve my original label checkingalgorithm to work better for curves. Following this, any still poorly labeledimages were removed from the dataset. 

After checking histograms of the coefficientlabels, I realized I needed additional curved line images, and gatheredadditional data for curved lines, as well as from a different camera, in orderto help even out the distribution. After finding they still needed a betterdistribution, I 

found ranges of the labels to iteratethrough and create additional training images through rotation of theoriginals. 

The next step was to actually build andtrain a model. I built a somewhat successful model usingperspective-transformed images, built a slightly improved model by feeding inregular road images, but still was not at a sufficient level of quality. Aftertrying to use activation maps of the convolutional layers, I moved on to afully convolutional model. After changing the training labels to be the ‘G’ color channelcontaining the detected lane drawing, a robust model was created that was fasterand more accurate than my previous model based on typical computer visiontechniques. 

Two very interesting, but very challengingissues arose during this project. I had never before used my own dataset intraining a model, and curating a good dataset was a massive time commitment,and especially due to the limits of the early models I used, often difficult totell how sufficient of a dataset I had. The second challenge was in settling ona model – I originally worried Iwould have to also somehow train the neural network to detect perspectivetransformation points or similar. Instead, I learned for the first time how touse a fully convolutional neural network, and it solved the problem.  

### Improvement 

One potential improvement to the model couldbe the use of a recurrent neural network (RNN). The current version of my modeluses an averaging across five frames to smooth out any issues on a single framedetection, outside of the actual neural network itself. On the other hand, aRNN would be able to directly look at previous frames in order to learn thatwhat was detected in a previous frame matters to the current frame. By doingso, it would potentially lose any of the more erratic predictions entirely. Ihave not yet used a RNN architecture, but I plan to do so eventually for futureprojects. 

  

------

[[1\]](#_ftnref1) https://github.com/raghakot/keras-vis 

[[2\]](#_ftnref2) http://mi.eng.cam.ac.uk/projects/segnet/#research 

[[3\]](#_ftnref3) https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/challenge_video.mp4