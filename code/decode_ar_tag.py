import cv2 
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread('data/april_tag.png')
plt.imshow(img, cmap = 'gray')
resized = cv2.resize(img, (200,200), interpolation = cv2.INTER_AREA)
gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
thresh, binary_img = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

def decodeAprilTag(img):
    # List of 8 25x200 image slices
    sliced = np.array_split(img,8,axis=0)

    # List of 8 lists of 8 25x25 image blocks
    blocks = [np.array_split(img_slice,8,axis=1) for img_slice in sliced]

    while np.median(blocks[5][5])!=255:
        img = np.rot90(img)
        sliced = np.array_split(img,8,axis=0)
        blocks = [np.array_split(img_slice,8,axis=1) for img_slice in sliced]
        
    index = [(3,3),(3,4),(4,4),(4,3)]    
    value = []

    for i in index:
        if np.median(blocks[i[0]][i[1]]) == 255:
            value.append(1)
        else:
            value.append(0)
          
    strings = [str(integer) for integer in value]
    a_string = "".join(strings)
   
    # stacking them back together
    img_stacked = np.block(blocks)
    
    return int(a_string, 2)

tag_id = decodeAprilTag(binary_img)
print(tag_id)



