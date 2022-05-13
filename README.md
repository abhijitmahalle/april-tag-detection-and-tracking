# April Tag Detection and Image Superimposition
This repository contains the code to detect a custom AR tag which is a fiducial marker. The program detects the April Tag using Fast Fourier Transform, detects its corner using Shi Thomasi corner detection. It utilizes the concepts of homography to superimpose an image over the detected April Tag. A virtual cube is also drawn over the tag using the concpets of projection and calibration matrices. Note that no inbuilt OpenCV functions were used except for FFT and Shi Thomasi

### Instructions to run the code:
Run the following command in the terminal for the corresponding problem:
```
python ar_tag_detection.py
python decode_ar_tag.py
python superimposition.py
python virtual_cube.py
```
<img src=https://github.com/abhijitmahalle/AR_tag_detection/blob/master/gif/testudo_superimposed.gif  width=49% height=50%> <img src=https://github.com/abhijitmahalle/AR_tag_detection/blob/master/gif/virtual_cube.gif  width=49% height=50%>
