from cgi import test
from ctypes.wintypes import HHOOK
from multiprocessing.dummy import Array
from pickletools import uint8
import cv2 as cv 
import numpy as np
from kalmanfilter import KalmanFilter





def Test(): 
    r=50
    blank2=np.zeros((100,100,3),dtype=np.uint8)
    PhospheneImage=cv.circle(blank2.copy(),(50,50),r,(255,255,255),-1) #-1 means fill the circle


    blank=np.zeros((500,500,3),dtype='uint8')
    x=50
    y=100

    blank[y-r:y+r,x-r:x+r]=PhospheneImage
    cv.imshow('asd',blank)
    cv.waitKey(0)

# Test()
Old_Objetcs= [1,2,3]
New_Objetcs= [6,1]
# print(len(New_Objetcs))
# print(len(Old_Objetcs))
Diff=abs(len(New_Objetcs)-len(Old_Objetcs))
New_Greater=False





# if len(New_Objetcs)>len(Old_Objetcs):
#      IOU_Array =[[0]*(len(Old_Objetcs)+Diff)]*(len(New_Objetcs))
#      New_Greater=True
    
# else :
#     IOU_Array =[[0]*(len(Old_Objetcs))]*(len(New_Objetcs)+Diff)
#     New_Greater=False

MaxAmong= max(len(Old_Objetcs),len(New_Objetcs))

IOU_Array =[[0 for i in range(MaxAmong)] for j in range(MaxAmong)]

test_me =[0]


def FillUnionOverInterSection(New_Objetcs,Old_Objetcs) :
  
    
    for i in range(len(New_Objetcs)):

        for j in range(len(Old_Objetcs)):
            # if (len(Old_Objetcs) > j):
                # print(Old_Objetcs[j]+New_Objetcs[i])
                IOU_Array[i][j]=Old_Objetcs[j]+New_Objetcs[i]
                # print(IOU_Array)
            # else :
            #     IOU_Array[i][j]=0
            #  Old_Dimensions=New_Objetcs[j]
            #  value=UnionOverInterSection(Old_Dimensions,New_Dimensions)
            
             
            # print(IOU_Array[i][j])
    # print(len(New_Objetcs))
    # print(len(Old_Objetcs))        
    if len(New_Objetcs) <len(Old_Objetcs):
        # print('asdasd')
        Current_insert=[0]*len(Old_Objetcs)
        IOU_Array[len(New_Objetcs)]=Current_insert

    print(IOU_Array)

# FillUnionOverInterSection(New_Objetcs,Old_Objetcs)
# def UnionOverInterSection(Old_Dimensions,New_Dimensions):
#     x_new,x2_new,y_new,y2_new=New_Dimensions['x'],New_Dimensions['x2'],New_Dimensions['y'],New_Dimensions['y2']
#     x_old,x2_old,y_old,y2_old=Old_Dimensions['x'],Old_Dimensions['x2'],Old_Dimensions['y'],Old_Dimensions['y2']
#     First_Corner=(max(x_new,x_old),max(y_new,y_old))
#     Second_Corner=(min(x2_new,x2_old),min(y2_new,y2_old))
#     width=Second_Corner[0]-First_Corner[0]
#     height=Second_Corner[1]-First_Corner[1]
#     # print('current width'+str(width)+' current height:'+str(height))
#     Inter_Area=width *height

#     First_box_Area=(x2_new-x_new)* (y2_new-y_new)
#     Second_box_Area=(x2_new-x_new)* (y2_new-y_new)
#     Union_Area=(First_box_Area+Second_box_Area) - Inter_Area
#     return(Inter_Area/Union_Area)             

kf= KalmanFilter ()

pre=kf.predict(100,20)
pre=kf.predict(100,20)
pre=kf.predict(100,20)
pre=kf.predict(100,20)
pre=kf.predict(100,20)
pre=kf.predict(100,20)

# once u create another must be wra b3dd 3l6ool
kf2= KalmanFilter ()
pre2=kf2.predict(10,2)
pre2=kf2.predict(10,2)
pre2=kf2.predict(10,2)
pre2=kf2.predict(10,2)

list_a =  3
list_b = [ 6]
list_c = [ 9]
# list_a=[*list_a,*list_b]

# tuple_a=('b','b','a')

# All= tuple_a+(list_a,)+(list_c,)



# print(All)


array =[1,2,3,4,5,6]
for i in range(len(array)-4,len(array)):
    print(array[i])

print(sum(array))


# for i in range()



print(pre)
print(pre2)


def EquationOfLine(Last_Dimensions,Previous_Dimensions,frame_height):
    x2,y2=Last_Dimensions[0],Last_Dimensions[1]
    x1,y1=Previous_Dimensions[0],Previous_Dimensions[1]
    slope =(y2-y1) / (x2-x1)
    y_inter=y1-(slope*x1)
    newX_inter=(frame_height-y_inter)/slope
    print('aa')
    # cv.line(frame, Previous_Dimensions, (newX_inter,frame_height), (0,255,0), 2) # line  starts from the 0.25 width of screen


EquationOfLine((100,200),(500,600),1200)

testme ={'name':'ahmed','age':15}

print(testme['name'])

