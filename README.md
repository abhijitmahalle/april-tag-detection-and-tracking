 [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# April Tag Detection and Image Superimposition
This repository contains code to detect a custom April tag which is a fiducial marker. The program detects the April Tag using Fast Fourier Transform, detects its corner using Shi Thomasi corner detection. It utilizes the concepts of homography to superimpose an image over the detected April Tag. A virtual cube is also drawn over the tag using the concpets of projection and calibration matrices. Note that no inbuilt OpenCV functions were used except for FFT and Shi Thomasi.

## Project Description
The project consists of two parts:
1. In part 1, the goal was to superimpose a custom image (in this case, Testudo, which is University of Maryland's mascot) over an April Tag. 
2. In part 2, the goal was to draw a 3-D virtual cube over the detected April Tag.

### Approch to part 1
1. The corners of the April Tag were detected using [Shi-Thomasi](https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_shi_tomasi/py_shi_tomasi.html) corner detector. 
2. The orientation of the April Tag in the video-frame was determined.
3. Homography matrix was computed between the detected corners and the Testudo image and inverse warping was performed to set the value of the pixels in the frame to that of the testudo image. This eliminates the "holes" that may arise if the forward warping was performed.

### Approch to part 2
1. Projective transformation in 3-D was used to project the points of a 3-D cube onto a 2-D video frame. The homography matrix was computed for the 3-D points unlike the 2-D ones in the part 1.

## Dependencies
  - Python 2.0 or above
  - OpenCV
## Instructions to run the code
Run the following command in the terminal for the corresponding problem.

To detect edges of the April Tag using FFT
```
python ar_tag_detection.py
```
To find the orientation of the April Tag by decoding it
```
python decode_ar_tag.py
```
To superimpose an image over the April Tag
```
python superimposition.py
```
To draw 3-D virtual cube over the April Tag
```
python virtual_cube.py
```


## Results
Fast Fourier Transform followed by Low pass filter to detect edges of April Tag
<p align="center">
  <img src=https://github.com/AbhijitMahalle/AR_tag_detection/blob/master/results/fft.png>
<p align="center">
  
### Part 1
<p align="center">
  <img src=https://github.com/abhijitmahalle/AR_tag_detection/blob/master/gif/testudo_superimposed.gif> 
<p align="center">
  
### Part 2
<p align="center">
  <img src=https://github.com/abhijitmahalle/AR_tag_detection/blob/master/gif/virtual_cube.gif>
<p align="center">
