import cv2 as cv 
img =cv.imread('photo/cat.png')
cv.imshow('cat',img)
#used to rescale an image function used to resize a video 
def rescaleFrame(frame,scale=0.5):
    #used for images and videos and live videos
    width=int(frame.shape[1]*scale)
    height=int(frame.shape[0]*scale)

    dimentions=(width,height)

    return cv.resize(frame,dimentions,interpolation=cv.INTER_AREA)

resized_img=rescaleFrame(img,scale=5)
cv.imshow('image',resized_img)
cv.waitKey(0)


capture =cv.VideoCapture('video/bat1.mp4') 
# to resize  a video
while True:
   isTrue,frame =capture.read()

   resized_vid=rescaleFrame(frame,scale=0.2)
   cv.imshow('Video',frame) 
   cv.imshow('Video resized',resized_vid) 
   if cv.waitKey(20) & 0xFF==ord('s'):
      break 



capture.release()
cv.destroyAllWindows() 

def changeRes (width,height):
    #used for live videos only that has been captured from camera 
    capture.set(3,width)
    capture.set(4,height)