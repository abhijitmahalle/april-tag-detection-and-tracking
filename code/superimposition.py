import cv2 
import numpy as np
import matplotlib.pyplot as plt
import copy
import math

def preprocessing(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gblur = cv2.GaussianBlur(gray,(19, 19),0)
    thresh = cv2.threshold(gblur, 200, 255, cv2.THRESH_BINARY)[1]
    threshblur = cv2.GaussianBlur(thresh,(5, 5),0)
    kernel = np.ones((3,3), np.uint8)
    img_dilation = cv2.dilate(threshblur, kernel, iterations=3)
    img_erosion = cv2.erode(img_dilation, kernel, iterations=3)
    return img_erosion

def detectCorners(img):
    corners = cv2.goodFeaturesToTrack(img,15,0.05,50)
    corners = np.int0(corners)
    corners  = corners.reshape(corners.shape[0],2)
    return corners

def distance(pt1, pt2):
    return math.sqrt(((pt1[0]-pt2[0])**2)+((pt1[1]-pt2[1])**2))

def checkIfPointNearLine(pt1, pt2, corners):
    points = copy.deepcopy(corners)
    
    for i in points:
        d = np.abs(np.cross(pt2-pt1, i-pt1)/np.linalg.norm(pt2-pt1))
        if d<10:
            corners = np.delete(corners, [np.where(corners == i)[0][0]], 0)     
    return corners

def tagCorners(corners):
    x_min, y_min = np.argmin(corners, axis=0)
    x_max, y_max = np.argmax(corners, axis=0)
    
    pt1 = corners[x_min]
    pt2 = corners[y_max]
    pt3 = corners[x_max]
    pt4 = corners[y_min]

    corners = np.delete(corners, [x_min, y_max, x_max, y_min], 0)
    
    corners = checkIfPointNearLine(pt1, pt2, corners)
    corners = checkIfPointNearLine(pt2, pt3, corners)
    corners = checkIfPointNearLine(pt3, pt4, corners)
    corners = checkIfPointNearLine(pt4, pt1, corners)
    
    x_min, y_min = np.argmin(corners, axis=0)
    x_max, y_max = np.argmax(corners, axis=0)
    
    leftmost_corner = corners[x_min]
    bottommost_corner = corners[y_max]
    rightmost_corner = corners[x_max]
    topmost_corner = corners[y_min]
         
    tag_corners = np.float32([leftmost_corner, bottommost_corner, rightmost_corner, topmost_corner])
    return tag_corners

def spImgCorners(sp_img):
    width = sp_img.shape[1]
    height = sp_img.shape[0]
    sp_img_corners = np.float32([[0, 0],[0, height-1],[width-1, height-1],[width-1, 0]])
    return sp_img_corners

def homography(input_pts, output_pts):
    x1 = input_pts[0][0]
    x2 = input_pts[1][0]
    x3 = input_pts[2][0]
    x4 = input_pts[3][0]
    y1 = input_pts[0][1]
    y2 = input_pts[1][1]
    y3 = input_pts[2][1]
    y4 = input_pts[3][1]
    xp1 = output_pts[0][0]
    xp2 = output_pts[1][0]
    xp3 = output_pts[2][0]
    xp4 = output_pts[3][0]
    yp1 = output_pts[0][1]
    yp2 = output_pts[1][1]
    yp3 = output_pts[2][1]
    yp4 = output_pts[3][1]
    
    A = np.matrix([[-x1, -y1, -1, 0, 0, 0, x1*xp1, y1*xp1, xp1], 
                   [0, 0, 0, -x1, -y1, -1, x1*yp1, y1*yp1, yp1], 
                   [-x2, -y2, -1, 0, 0, 0, x2*xp2, y2*xp2, xp2], 
                   [0, 0, 0, -x2, -y2, -1, x2*yp2, y2*yp2, yp2],
                   [-x3, -y3, -1, 0, 0, 0, x3*xp3, y3*xp3, xp3], 
                   [0, 0, 0, -x3, -y3, -1, x3*yp3, y3*yp3, yp3],
                   [-x4, -y4, -1, 0, 0, 0, x4*xp4, y4*xp4, xp4], 
                   [0, 0, 0, -x4, -y4, -1, x4*yp4, y4*yp4, yp4]])
    
    U, Sigma, V = np.linalg.svd(A)
    
    H = np.reshape(V[-1, :], (3, 3))
    Lambda = H[-1,-1]
    H = H/Lambda
    return H

def aprilTagOrientation(img, tag_corners):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    binary = np.where(gray < 210, 0, 255)
   
    input_pts= np.float32([[0, 0],[0, 199],[199, 199],[199, 0]])
    tag = np.zeros((200, 200))
    H = homography(input_pts, tag_corners)
    
    for i in range(200):
        for j in range(200):
            pos = np.matmul(H, np.array([[j], [i], [1]]))
            pos = pos/pos[-1]
            tag[i][j] = binary[int(pos[1])][int(pos[0])]  
    
    sliced = np.array_split(tag,8,axis=0)
    blocks = [np.array_split(img_slice,8,axis=1) for img_slice in sliced]
    
    for i in range(len(blocks)):
        for j in range(len(blocks[i])):
            median = np.median(blocks[i][j], axis = 1)
            median = np.median(median)
            blocks[i][j][:] = median
      
    counter = 0
    
    while np.mean(blocks[5][5])!=255:
        tag = np.rot90(tag,3)
        counter+=1
        if counter>3:
            break
        sliced = np.array_split(tag,8,axis=0)
        blocks = [np.array_split(img_slice,8,axis=1) for img_slice in sliced]
    return counter

def superimpose(src_img, sp_img, tag_corners):
    input_pts = spImgCorners(sp_img)
    H = homography(input_pts, tag_corners)
    processed_img = copy.deepcopy(src_img)
    width = sp_img.shape[1]
    height = sp_img.shape[0]
    
    for i in range(height):
        for j in range(width):
            pos = np.matmul(H, np.array([[j], [i], [1]]))
            pos = pos/pos[-1]
            processed_img[int(pos[1])][int(pos[0])] =  sp_img[i][j] 
            
    return processed_img

def performSuperimposition(video, sp_img):
    print("Generating video ouput named 'testudo_superimposed'...\n")
    out = cv2.VideoWriter('testudo_superimposed.avi',cv2.VideoWriter_fourcc(*'XVID'), 27, (1920,1080))
    while True:
        try:
            isTrue, img = video.read()

            if isTrue == False:
                break
            sp_img_copy = copy.deepcopy(sp_img)
            processed_img = preprocessing(img)
            corners = detectCorners(processed_img)
            tag_corners = tagCorners(corners)
            tag_corners = np.int0(tag_corners)
            counter = aprilTagOrientation(img, tag_corners)
            
            while counter!=0:
                sp_img_copy = cv2.rotate(sp_img_copy, cv2.ROTATE_90_COUNTERCLOCKWISE)
                counter-=1

            superimposed_img = superimpose(img, sp_img_copy, tag_corners)
    
            out.write(superimposed_img)
        except:
            isTrue, img = video.read()
            sp_img_copy = copy.deepcopy(sp_img)
            continue
    out.release()
    print("Video output generated.\n")

capture = cv2.VideoCapture('data/1tagvideo.mp4')
sp_img = cv2.imread("data/testudo.png")
performSuperimposition(capture, sp_img)

