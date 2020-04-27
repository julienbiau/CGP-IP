import random
import numpy as np
import cv2
import sys
import math

class Functions:

    # Function Int
    # Connection 0 Int
    # Connection 1 Int
    # Parameter 0 Real no limitation
    # Parameter 1 Int [−16, +16]
    # Parameter 2 Int [−16, +16]
    # Gabor Filter Frequ. Int [0, 16]
    # Gabor Filter Orient. Int [−8, +8]

    num_functions = 50
    ksize = (3,3)

    @classmethod
    def getRandomFunction(cls):
        return random.randrange(1,1+cls.num_functions)

    @classmethod
    def execute(cls, func, connection0, connection1, parameter0, parameter1, parameter2, gabor_filter_frequency, gabor_filter_orientation):
        # CONST
        if func==4:
            return np.full(connection0.shape,fill_value=parameter0,dtype=float)
        # NOP
        elif func==5:
            return connection0
        # ADD connection0 connection1
        elif func==6:
            return np.add(connection0,connection1)
        # SUB connection0 connection1
        elif func==7:
            return np.subtract(connection0,connection1)
        # MUL connection0 connection1
        elif func==8:
            return np.multiply(connection0,connection1)
        # LOG connection0
        elif func==9:
            return np.log(connection0)
        # EXP connection0
        elif func==10:
            return np.exp(connection0)
        # SQRT connection0
        elif func==11:
            return np.sqrt(np.abs(connection0))
        # ADDC connection0 parameter0
        elif func==12:
            return connection0 + parameter0
        # SUBC connection0 parameter0
        elif func==13:
            return connection0 - parameter0
        # MULLC connection0 parameter0
        elif func==14:
            return connection0 * parameter0
        # DILATE connection0
        elif func==15:
            return cv2.dilate(np.int16(connection0),cls.ksize)
        # ERODE connection0
        elif func==16:
            return cv2.erode(np.int16(connection0),cls.ksize)
        # LAPLACE connection0
        elif func==17:
            return connection0 #cv2.Laplacian(connection0,cv2.CV_8U) # TO CHECK
        # CANNY connection0
        elif func==18:
            return cv2.Canny(np.uint8(connection0),0,0) # TO CHECK
        # GAUSS
        elif func==19:
            return cv2.GaussianBlur(np.int16(connection0),cls.ksize,0) # TO CHECK
        # GAUSS2 parameter1 parameter2
        elif func==20:
            return cv2.GaussianBlur(np.int16(connection0),cls.ksize,0) # TO CHECK
        # MIN connection0 connection1
        elif func==21:
            return np.minimum(connection0,connection1)
        # MAX connection0 connection1
        elif func==22:
            return np.maximum(connection0,connection1)
        # AVG connection0 connection1
        elif func==23:
            return (connection0+connection1)/2.0
        # ABSDIFFERENCE connection0 connection1
        elif func==24:
            return np.abs(connection0-connection1)
        # MINC connection0 parameter0
        elif func==25:
            return np.minimum(connection0,np.full(connection0.shape,fill_value=parameter0,dtype=float))
        # MAXC connection0 parameter0
        elif func==26:
            return np.maximum(connection0,np.full(connection0.shape,fill_value=parameter0,dtype=float))
        # NORMALIZE
        elif func==27:
            # Normalised [0,255] as integer
            return 255*(connection0 - np.min(connection0))/np.ptp(connection0).astype(float)
        # SOBEL
        elif func==28:
            return connection0 # cv2.Sobel(np.int16(connection0),cv2.CV_8U,1,1,cls.ksize) # TO CHECK
        # SOBELX
        elif func==29:
            return connection0 # cv2.Sobel(np.int16(connection0),cv2.CV_8U,abs(parameter1),0,cls.ksize) # TO CHECK
        # SOBELY
        elif func==30:
            return connection0 # cv2.Sobel(np.int16(connection0),cv2.CV_8U,0,abs(parameter1),cls.ksize) # TO CHECK
        # THRESHOLD connection0 parameter0
        elif func==31:
            retval, dst = cv2.threshold(np.int16(connection0),parameter0,0,cv2.THRESH_TRUNC)
            return dst
        # SMOOTHMEDIAN
        elif func==32:
            return connection0
        # SMOOTHBILATERAL
        elif func==33:
            return connection0
        # SMOOTHBLUR
        elif func==34:
            return connection0
        # UNSHARPEN
        elif func==35:
            return connection0
        # SHIFT
        elif func==36:
            num_rows, num_cols = connection0.shape[:2]
            translation_matrix = np.float32([ [1,0,parameter1], [0,1,parameter2] ])
            return cv2.warpAffine(np.int16(connection0), translation_matrix, (num_cols, num_rows))
        # SHIFTUP
        elif func==37:
            num_rows, num_cols = connection0.shape[:2]
            translation_matrix = np.float32([ [1,0,0], [0,1,-1] ])
            return cv2.warpAffine(np.int16(connection0), translation_matrix, (num_cols, num_rows))
        # SHIFTDOWN
        elif func==38:
            num_rows, num_cols = connection0.shape[:2]
            translation_matrix = np.float32([ [1,0,0], [0,1,1] ])
            return cv2.warpAffine(np.int16(connection0), translation_matrix, (num_cols, num_rows))
        # SHIFTLEFT
        elif func==39:
            num_rows, num_cols = connection0.shape[:2]
            translation_matrix = np.float32([ [1,0,-1], [0,1,0] ])
            return cv2.warpAffine(np.int16(connection0), translation_matrix, (num_cols, num_rows))
        # SHIFTRIGHT
        elif func==40:
            num_rows, num_cols = connection0.shape[:2]
            translation_matrix = np.float32([ [1,0,1], [0,1,0] ])
            return cv2.warpAffine(np.int16(connection0), translation_matrix, (num_cols, num_rows))
        # RESCALE parameter0
        elif func==41:
            height, width = connection0.shape[:2] # TO CHECK +1 to only reduce
            return cv2.resize(cv2.resize(np.int16(connection0), (int(width/(abs(parameter0)+1)), int(height/(abs(parameter0)+1))), interpolation = cv2.INTER_AREA) , (width, height), interpolation = cv2.INTER_AREA) 
        # GABOR
        elif func==42:
            return connection0
        # RESIZETHENGABOR
        elif func==43:
            return connection0
        # MINVALUE
        elif func==44:
            return np.full(connection0.shape,fill_value=connection0.min(),dtype=float)
        # MAXVALUE
        elif func==45:
            return np.full(connection0.shape,fill_value=connection0.max(),dtype=float)
        # AVGVALUE
        elif func==46:
            return np.full(connection0.shape,fill_value=connection0.mean(),dtype=float)
        # LOCALMIN parameter1 parameter2
        elif func==47:
            retval = np.ndarray(connection0.shape,dtype=float)
            height, width = connection0.shape[:2]
            x = abs(parameter1)
            y = abs(parameter2)
            xa = 0
            xb = 0
            yc = 0
            yd = 0
            if x==0:
                x = 1
            if y==0:
                y = 1
            for i in range(0,height):
                yc = i - y
                yd = i + y
                if yc<0:
                    yc = 0
                if yd>=height:
                    yd = height-1
                for j in range(0,width):
                    xa = j - x
                    xb = j + x
                    if xa<0:
                        xa = 0
                    if xb>=width:
                        xb = width-1
                    retval[i,j] = connection0[yc:yd,xa:xb].min()
            return retval
        # LOCALMAX parameter1 parameter2
        elif func==48:
            return connection0
        # LOCALAVG parameter1 parameter2
        elif func==49:
            return connection0
        # LOCALNORMALIZE parameter1 parameter2
        elif func==50:
            return connection0
