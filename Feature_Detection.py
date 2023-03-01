from pickle import NONE
from sys import flags
import cv2 as cv
import numpy as np
import os
import glob

 # sift=cv.xfeatures2d.SIFT_create()
            # surf=cv.xfeatures2d.SURF_create()
Old_Path = glob.glob("C:/Users/AHMED FAROUK/Documents/python computer vision/old frame objects/*")
New_Path = glob.glob("C:/Users/AHMED FAROUK/Documents/python computer vision/new frame objects/*")
Old_Objetcs = []

for img in Old_Path:
    n = cv.imread(img)
    scale_percent = 1 # percent of original size
    width = int(n.shape[1] * scale_percent )
    height = int(n.shape[0] * scale_percent )
    dim = (width, height)
    resized = cv.resize(n, dim, interpolation = cv.INTER_AREA)
    Old_Objetcs.append(resized)

New_Objetcs = []
for img in New_Path:
    n = cv.imread(img)
    scale_percent = 1 # percent of original size
    width = int(n.shape[1] * scale_percent )
    height = int(n.shape[0] * scale_percent )
    dim = (width, height)
    resized = cv.resize(n, dim, interpolation = cv.INTER_AREA)
    New_Objetcs.append(resized) 
    
def myFunc(e):
    return e['good_match']      

# cv.imshow('asdasd',Old_Objetcs[0])
# cv.waitKey(0)

def Feature_Det():
   
    All=[]
    Best_matches=[]

    for i in range(len(New_Objetcs)):
        New_Object=New_Objetcs[i]
        # print(len(Old_Objetcs))
        for j in range(len(Old_Objetcs)):
           
            Old_Object=Old_Objetcs[j]
            orb = cv.ORB_create()
            # sift=cv.xfeatures2d.SIFT_create()
            # surf=cv.xfeatures2d.SURF_create()
            kpOld, desOld = orb.detectAndCompute(Old_Object, None)
            kpNew, desNew = orb.detectAndCompute(New_Object, None)

            bf=cv.BFMatcher()
            try:
                matches=bf.knnMatch(desOld,desNew,k=2)
            except:
                pass
            good_matches=[]
            for m,n in matches:
                if m.distance<0.72*n.distance:
                    good_matches.append([m])
            Current_match ={'New_index':i ,'Old_index':j,'good_match':len(good_matches) ,'kpOld':kpOld ,'kpNew':kpNew}
            All.append(Current_match)

           
        # print(len(All))        
        All.sort( reverse=True,key=myFunc) # this means the first or greater value of good_match is the same object
        Best_matches.append(All[0])
        All.clear()
            # imgKP=cv.drawKeypoints(img1,kp1,None)
            # print(len(good_matches))
            # cv.imshow('asdasd',img_Final)
            # cv.waitKey(0)
    # print(len(Best_matches))
    for k in range(len(Best_matches)):
        old=Best_matches[k]['Old_index']       
        new=Best_matches[k]['New_index']
        best=Best_matches[k]['good_match'] 

        kpOld=Best_matches[k]['kpOld'] 
        kpNew=Best_matches[k]['kpNew'] 
        img_Final=cv.drawMatchesKnn(Old_Objetcs[old],kpOld,New_Objetcs[new],kpNew,good_matches,None,flags=2)
        # cv.imshow('old',Old_Objetcs[old])      
        # cv.imshow('new',New_Objetcs[new])    
        cv.imshow('final',img_Final)    
        print(best)  
        cv.waitKey(0)
       

Feature_Det()       