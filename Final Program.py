#Team project: Yuxi Luo, William Kostuch, and Hazel Simpson
#August 2015, Carleton College
import cv2
import numpy as np
from matplotlib import pyplot as plt
    
#Function for selecting new object with mouse.    
def mouseDraw(event,x,y,flags,param):
    '''Draws rectangle with mouse'''
    global ix,iy, draw, img2, saveImg, displayImg
    if event == cv2.EVENT_LBUTTONDOWN:
        draw = True
        iy = y
        ix = x
    elif event == cv2.EVENT_LBUTTONUP:
        draw = False
        cv2.rectangle(trackImg,(ix,iy),(x,y),(255,0,0))
        x = 1280 - x
        ix = 1280 - ix        
        saveImg = img[iy:y, x:ix]
        displayImg = cv2.flip(saveImg, 1)
        cv2.imshow('Saved',displayImg)
        img2 = cv2.imwrite('Tracking Image.jpg',saveImg)
    return img2
cv2.setMouseCallback('Tracking', mouseDraw)    

#Access webcam
vidCap = cv2.VideoCapture(0)

#Make kernel for use in image noise reduction
kernel = np.ones((5,5),np.uint8)

#Initialize tracking window size and position, mouse variables
l,r,h,w = 0,2560,0,1600
window = (h,l,w,r) 
draw = False 
ix,iy = -1,-1    
    
#Target image initialization  
img2 = cv2.imread("ball.jpeg")

#Loops until program is manually stopped, displaying and manipulating every frame
while True:
    #Each frame from the webcam
    ret, img = vidCap.read()
    frame = img
    
    #Remove noise from images
    img = cv2.dilate(img,kernel,iterations=1)
    img = cv2.erode(img, kernel, iterations=1)
    img2 = cv2.dilate(img2,kernel,iterations=1)
    img2 = cv2.erode(img2,kernel,iterations=1)
    
    #Blurs the frame for later use
    frameBlur = cv2.blur(img, (30,30))
    
    #Convert each frame to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    #Limit the hue in the video to help with consistent tracking
    MAX = np.array([121,255,255],np.uint8)
    MIN = np.array([95,127,140],np.uint8)    
    
    #Convertes target image to HSV, creates histogram for it
    hsvroi = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
    histroi = cv2.calcHist([hsvroi],[0],None,[180],[0,180]) 
    cv2.normalize(histroi ,histroi ,0,255,cv2.NORM_MINMAX)    
    
    #Creates backprojection and shows it
    bp = cv2.calcBackProject([hsv], [0], histroi, [0, 180], 1)
    cv2.imshow("Back Projection", bp)
        
    #Makes criteria for Camshift
    criteria = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )
        
    #Applies camshift to picture, prints information
    ret, window = cv2.CamShift(bp, window, criteria)
    #print ret
    
    #Draws rectangle for tracking, prints points of each corner
    pts = cv2.boxPoints(ret)
    pts = np.int0(pts)
    trackImg = cv2.polylines(frameBlur,[pts],True, (179,0,255),10)
    #print pts
    
    #Sets new variable for further manipulation
    trackImg = cv2.fillPoly(frameBlur, [pts], (0,0,0))
    
    #Creates masks for start of blurring
    mask = np.zeros((720,1280,3), np.uint8)
    mask = cv2.fillPoly(mask,[pts],(255,255,255))
    #cv2.imshow('mask',mask)

    #Shows tracking box with black background
    bwa1 = cv2.bitwise_and(frame, mask)
    #cv2.imshow('bwa1',bwa1)

    #Adds un-blurred trackbox with blurred frame
    bwa2= cv2.add(trackImg, bwa1)
    trackImg = bwa2
    
    
    #Final tracking image, complete with blur and flipped frame
    trackImg = cv2.flip(trackImg, 1)    
    
    #Stops program if user hits "q" key    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
    
    #Mouse selection for tracking window
    #Freezes video for selection when user presses "s"
    if cv2.waitKey(1) & 0xFF == ord('s'):
        cv2.setMouseCallback('Tracking',mouseDraw)
        #Allows real-time tracking of mouse on image
        while True:
            #Places text instructions on screen
            message = "Select object to track, then press escape"
            cv2.putText(trackImg, message, (3, 25), font, 2, (0, 0, 255), 2)
            cv2.imshow('Tracking',trackImg)
            message = ""
            #Updates target image for new object
            img2 = cv2.imread("Tracking Image.jpg")
            k = cv2.waitKey(10) & 0xFF
            #If user presses escape, video resumes
            if k == 27:
                cv2.destroyWindow('Saved')
                break        
    
    #More text instructions
    font = cv2.FONT_HERSHEY_PLAIN
    message = "Press 's' to pause video"
    cv2.putText(trackImg, message, (3, 25), font, 2, (0, 0, 255), 2)               
    message = ""
    
    #Displays final result
    cv2.imshow('Tracking',trackImg)    
    
vidCap.release()
cv2.destroyAllWindows()
