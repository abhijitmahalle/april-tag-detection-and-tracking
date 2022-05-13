import cv2
import scipy.fftpack as sp
import matplotlib.pyplot as plt
import copy
import numpy as np

capture = cv2.VideoCapture('data/1tagvideo.mp4')
capture.set(cv2.CAP_PROP_POS_FRAMES, 13)
isTrue, img = capture.read()
gblur = cv2.GaussianBlur(img,(5, 5),0)
gray = cv2.cvtColor(gblur, cv2.COLOR_BGR2GRAY)
plt.imshow(img)

def circularMask(img_shape, radius, high_pass = True):
    '''
    Input:
    img_shape    : tuple (height, width)
    radius : scalar
    high_pass : boolean, default value True
    
    Output:
    np.array of shape  that says True within a circle with diameter =  around center 
    '''
    assert len(img_shape) == 2
    rows, cols = img_shape
    center_y, center_x = int(rows/2), int(cols/2)
    y, x = np.ogrid[:rows, :cols]
    mask_area = (x - center_x)**2 + (y - center_y)**2 <= radius**2
    
    if high_pass:
        mask = np.ones((rows, cols)) 
        mask[mask_area] = 0
    else:
        mask = np.zeros((rows, cols)) 
        mask[mask_area] = 1

    return mask

def findEdges(gray_img, mask):
    img_copy = copy.deepcopy(gray_img)
    
    f = sp.fft2(img_copy)  
    f_shift = sp.fftshift(f)
    magnitude_spectrum_f_shift = np.log(np.abs(f_shift))

    f_shift_masked = f_shift * mask
    magnitude_spectrum_f_shift_masked = np.log(np.abs(f_shift_masked))

    processed_img = sp.ifftshift(f_shift_masked)
    processed_img = sp.ifft2(processed_img)
    processed_img = np.abs(processed_img)
    processed_img = np.where(processed_img < 45, 0, 255)

    fx, plts = plt.subplots(2,2,figsize = (15,10))
    plts[0][0].imshow(img_copy, cmap = 'gray')
    plts[0][0].set_title('Grayscale Image')
    plts[0][1].imshow(magnitude_spectrum_f_shift, cmap = 'gray')
    plts[0][1].set_title('FFT of Grayscale Image')
    plts[1][0].imshow(magnitude_spectrum_f_shift_masked, cmap = 'gray')
    plts[1][0].set_title('Mask + FFT of Grayscale Image')
    plts[1][1].imshow(processed_img, cmap = 'gray')
    plts[1][1].set_title('Detected April Tag')
    return processed_img

shape = gray.shape[:2]
mask = circularMask(shape, 50)
a = findEdges(gray, mask)

