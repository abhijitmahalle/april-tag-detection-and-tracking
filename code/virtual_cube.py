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
    rightmost_corner = corners[x_max]
    topmost_corner = corners[y_min]
    bottommost_corner = corners[y_max]
            
    tag_corners = np.float32([leftmost_corner, bottommost_corner, rightmost_corner, topmost_corner])
    
    return tag_corners

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

def projectionMatrix(K, H):
    K_inv = np.linalg.inv(K) 
    B_tilde = np.matmul(K_inv, H)
    if np.linalg.det(B_tilde)<0:
        B_tilde =  B_tilde*(-1)
    Lambda = 2/(np.linalg.norm(np.matmul(K_inv, H[:,0])) + np.linalg.norm(np.matmul(K_inv, H[:,1])))
    r1 = Lambda * B_tilde[:,0]
    r2 = Lambda * B_tilde[:,1]
    r3 = np.cross(r1, r2, axis = 0)
    t = Lambda * B_tilde[:,2]
    B =  np.column_stack((r1, r2, r3, t))
    P = np.matmul(K, B) 
    return P

def findCubeCorners(P):
    pt1 = np.array([[0],[0],[-1],[1]])
    pt2 = np.array([[0],[1],[-1],[1]])
    pt3 = np.array([[1],[1],[-1],[1]])
    pt4 = np.array([[1],[0],[-1],[1]])
    
    cubept1 = np.matmul(P, pt1)
    cubept1 = cubept1/cubept1[-1]
    cubept2 = np.matmul(P, pt2)
    cubept2 = cubept2/cubept2[-1]
    cubept3 = np.matmul(P, pt3)
    cubept3 = cubept3/cubept3[-1]
    cubept4 = np.matmul(P, pt4)
    cubept4 = cubept4/cubept4[-1]
    
    cubept1 = cubept1[:-1]
    cubept2 = cubept2[:-1]
    cubept3 = cubept3[:-1]
    cubept4 = cubept4[:-1]
    
    cubept1 = np.asarray(cubept1)
    cubept2 = np.asarray(cubept2)
    cubept3 = np.asarray(cubept3)
    cubept4 = np.asarray(cubept4)
    
    cube_corners = np.float32([cubept1, cubept2, cubept3, cubept4])
    return cube_corners

def drawCube(img, tag_corners, cube_corners):
    processed_img = copy.deepcopy(img)
    cube_corners = cube_corners.astype(int)
    for i in range(4):
        p1 = (cube_corners[i][0][0], cube_corners[i][1][0])
        p2 = (tag_corners[i][0], tag_corners[i][1])
        processed_img = cv2.line(processed_img, p1, p2, (0, 255, 0), thickness=3, lineType=8)
        
    for i in range(3):
        p1 = (cube_corners[i][0][0], cube_corners[i][1][0])
        p2 = (cube_corners[i+1][0][0], cube_corners[i+1][1][0])
        processed_img = cv2.line(processed_img, p1, p2, (0, 255, 0), thickness=3, lineType=8)
    
    p1 = (cube_corners[0][0][0], cube_corners[0][1][0])
    p2 = (cube_corners[3][0][0], cube_corners[3][1][0])
    processed_img = cv2.line(processed_img, p1, p2, (0, 255, 0), thickness=3, lineType=8)
    return processed_img

def virtualCube(video, K):
    print("Generating video ouput named 'virtual_cube'...\n")
    out = cv2.VideoWriter('virtual_cube.avi',cv2.VideoWriter_fourcc(*'XVID'), 27, (1920,1080))
    april_tag_wc = np.float32([[0, 0],[0, 1],[1, 1],[1, 0]])
    while True:
        try:
            isTrue, img = video.read()

            if isTrue == False:
                break
           
            processed_img = preprocessing(img)
            corners = detectCorners(processed_img)
            tag_corners = tagCorners(corners)
            tag_corners = np.int0(tag_corners)
            H = homography(april_tag_wc, tag_corners)    
            P = projectionMatrix(K, H)
            cube_corners = findCubeCorners(P)
            cube_img = drawCube(img, tag_corners, cube_corners)
            out.write(cube_img)
        except:
            isTrue, img = video.read()
            continue
    out.release()
    print("Video output generated.\n")

K = np.array([[1346.100595, 0, 932.1633975], [0, 1355.933136, 654.8986796], [0, 0, 1]])
capture = cv2.VideoCapture('data/1tagvideo.mp4')
virtualCube(capture, K)


