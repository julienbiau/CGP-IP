import random
import numpy as np
import cv2
import sys
import math
from multiprocessing import Pool

data = None
data_x = None
data_y = None

def init_pool_mmmn(data_arg,data_y_arg,data_x_arg):
    global data
    global data_x
    global data_y
    data = data_arg
    data_x = data_x_arg
    data_y = data_y_arg

def localmin(a):
    y,x = a
    return data[y-data_y:y+data_y,x-data_x:x+data_x].min()

def localmax(a):
    y,x = a
    return data[y-data_y:y+data_y,x-data_x:x+data_x].max()

def localmean(a):
    y,x = a
    return data[y-data_y:y+data_y,x-data_x:x+data_x].mean()

def localnormalize(a):
    y,x = a
    tmp = (data[y,x] - np.min(data[y-data_y:y+data_y,x-data_x:x+data_x]))
    tmp = tmp/max(1,np.max(data[y-data_y:y+data_y,x-data_x:x+data_x]))
    return (255*(tmp)).astype("uint8")

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
    nb_processes = 16

    two_arguments = [False,False,False,False,False,False,True,True,True,False,False,False,False,False,False,False,False,False,False,False,False,True,True,True,True,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False]

    @classmethod
    def getRandomFunction(cls):
        return random.randrange(1,1+cls.num_functions)

    @classmethod
    def needSecondArgument(cls,function):
        return cls.two_arguments[function]

    @classmethod
    def execute(cls, func, connection0, connection1, parameter0, parameter1, parameter2, gabor_filter_frequency, gabor_filter_orientation):
        # CONST
        if func==4:
            return np.full(connection0.shape,fill_value=parameter0,dtype="uint8")
        # NOP
        elif func==5:
            return connection0
        # ADD connection0 connection1
        elif func==6:
            if True:
                tmp = np.add(connection0,connection1,dtype="int16")
                tmp[tmp>255] = 255
                return np.asarray(tmp,dtype="uint8")
            else:
                return np.add(connection0,connection1) # with overflow modulo
        # SUB connection0 connection1
        elif func==7:
            if True:
                tmp = np.subtract(connection0,connection1,dtype="int16")
                tmp[tmp<0] = 0
                return np.asarray(tmp,dtype="uint8")
            else:
                return np.subtract(connection0,connection1) # with overflow modulo
        # MUL connection0 connection1
        elif func==8:
            if True:
                tmp = np.multiply(connection0,connection1,dtype="int16")
                tmp[tmp>255] = 255
                return np.asarray(tmp,dtype="uint8")
            else:
                return np.multiply(connection0,connection1) # with overflow modulo
        # LOG connection0
        elif func==9:
            if True:
                tmp = np.array(connection0, dtype="uint8")
                tmp[tmp==0] = 1
                return np.asarray(np.log(tmp),dtype="uint8")
            else:
                return np.asarray(np.log(connection0), dtype="uint8") # with 0 warnings
        # EXP connection0
        elif func==10:
            if True:
                tmp = np.exp(connection0, dtype="float64")
                tmp[tmp>255] = 255
                return np.asarray(tmp,dtype="uint8")
            else:
                return np.asarray(np.exp(connection0), dtype="uint8") # inf => 0
        # SQRT connection0
        elif func==11:
            return np.asarray(np.sqrt(connection0), dtype="uint8")
        # ADDC connection0 parameter0
        elif func==12:
            if True:
                tmp = np.int16(connection0) + parameter0
                tmp[tmp>255] = 255
                return np.asarray(tmp,dtype="uint8")
            else:
                return np.asarray(connection0 + parameter0, dtype="uint8") # with overflow modulo
        # SUBC connection0 parameter0
        elif func==13:
            if True:
                tmp = np.int16(connection0) - parameter0
                tmp[tmp<0] = 0
                return np.asarray(tmp,dtype="uint8")
            else:
                return np.asarray(connection0 - parameter0, dtype="uint8") # with overflow modulo
        # MULLC connection0 parameter0
        elif func==14:
            if True:
                tmp = np.int16(connection0) * parameter0
                tmp[tmp>255] = 255
                return np.asarray(tmp,dtype="uint8")
            else:
                return np.asarray(connection0 * parameter0, dtype="uint8") # with overflow modulo
        # DILATE connection0
        elif func==15:
            return cv2.dilate(connection0,cls.ksize)
        # ERODE connection0
        elif func==16:
            return cv2.erode(connection0,cls.ksize)
        # LAPLACE connection0
        elif func==17:
            return cv2.Laplacian(connection0,cv2.CV_8U,cls.ksize)
        # CANNY connection0
        elif func==18:
            return cv2.Canny(connection0,100,200) # FIXED MIN MAX VALUES ?
        # GAUSS
        elif func==19:
            return cv2.GaussianBlur(connection0,cls.ksize,0)
        # GAUSS2 parameter1 parameter2
        elif func==20:
            x = abs(parameter1)
            y = abs(parameter2)
            if x%2==0:
                x = x + 1
            if y%2==0:
                y = y + 1
            return cv2.GaussianBlur(connection0,(abs(x),abs(y)),0)
        # MIN connection0 connection1
        elif func==21:
            return np.minimum(connection0,connection1)
        # MAX connection0 connection1
        elif func==22:
            return np.maximum(connection0,connection1)
        # AVG connection0 connection1
        elif func==23:
            return np.asarray(connection0/2+connection1/2, dtype="uint8")
        # ABSDIFFERENCE connection0 connection1
        elif func==24:
            return np.abs(connection0-connection1)
        # MINC connection0 parameter0
        elif func==25:
            return np.minimum(connection0,np.full(connection0.shape,fill_value=abs(parameter0),dtype="uint8"))
        # MAXC connection0 parameter0
        elif func==26:
            return np.maximum(connection0,np.full(connection0.shape,fill_value=abs(parameter0),dtype="uint8"))
        # NORMALIZE
        elif func==27:
            # Normalised [0,255] as integer
            return (255*(connection0 - np.min(connection0))/max(1,np.max(connection0))).astype("uint8")
        # SOBEL
        elif func==28:
            return cv2.Sobel(connection0,cv2.CV_8U,1,1,cls.ksize)
        # SOBELX
        elif func==29:
            return cv2.Sobel(connection0,cv2.CV_8U,1,0,cls.ksize)
        # SOBELY
        elif func==30:
            return cv2.Sobel(connection0,cv2.CV_8U,0,1,cls.ksize)
        # THRESHOLD connection0 parameter0
        elif func==31:
            retval, dst = cv2.threshold(connection0,abs(parameter0),0,cv2.THRESH_TRUNC)
            return dst
        # SMOOTHMEDIAN
        elif func==32:
            return cv2.medianBlur(connection0,cls.ksize[0])
        # SMOOTHBILATERAL
        elif func==33:
            return cv2.bilateralFilter(connection0,parameter1,parameter2,parameter2)
        # SMOOTHBLUR
        elif func==34:
            return cv2.blur(connection0,cls.ksize)
        # UNSHARPEN
        elif func==35:
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            return cv2.filter2D(connection0, -1, kernel)
        # SHIFT
        elif func==36:
            num_rows, num_cols = connection0.shape[:2]
            translation_matrix = np.float32([ [1,0,parameter1], [0,1,parameter2] ])
            return cv2.warpAffine(connection0, translation_matrix, (num_cols, num_rows))
        # SHIFTUP
        elif func==37:
            num_rows, num_cols = connection0.shape[:2]
            translation_matrix = np.float32([ [1,0,0], [0,1,-1] ])
            return cv2.warpAffine(connection0, translation_matrix, (num_cols, num_rows))
        # SHIFTDOWN
        elif func==38:
            num_rows, num_cols = connection0.shape[:2]
            translation_matrix = np.float32([ [1,0,0], [0,1,1] ])
            return cv2.warpAffine(connection0, translation_matrix, (num_cols, num_rows))
        # SHIFTLEFT
        elif func==39:
            num_rows, num_cols = connection0.shape[:2]
            translation_matrix = np.float32([ [1,0,-1], [0,1,0] ])
            return cv2.warpAffine(connection0, translation_matrix, (num_cols, num_rows))
        # SHIFTRIGHT
        elif func==40:
            num_rows, num_cols = connection0.shape[:2]
            translation_matrix = np.float32([ [1,0,1], [0,1,0] ])
            return cv2.warpAffine(connection0, translation_matrix, (num_cols, num_rows))
        # RESCALE parameter0
        elif func==41:
            height, width = connection0.shape[:2] # TO CHECK +1 to only reduce
            return cv2.resize(cv2.resize(connection0, (int(width/(abs(parameter0)+1)), int(height/(abs(parameter0)+1))), interpolation = cv2.INTER_AREA) , (width, height), interpolation = cv2.INTER_AREA) 
        # GABOR
        elif func==42:
            g_kernel = cv2.getGaborKernel(cls.ksize, gabor_filter_frequency, gabor_filter_orientation, 10.0, 0.5) # CHECK 10 and 0.5 VALUES
            return cv2.filter2D(connection0, -1, g_kernel)
        # RESIZETHENGABOR
        elif func==43:
            height, width = connection0.shape[:2] # TO CHECK +1 to only reduce
            tmp = cv2.resize(cv2.resize(connection0, (int(width/(abs(parameter0)+1)), int(height/(abs(parameter0)+1))), interpolation = cv2.INTER_AREA) , (width, height), interpolation = cv2.INTER_AREA) 
            g_kernel = cv2.getGaborKernel(cls.ksize, gabor_filter_frequency, gabor_filter_orientation, 10.0, 0.5) # CHECK 10 and 0.5 VALUES
            return cv2.filter2D(tmp, -1, g_kernel)
        # MINVALUE
        elif func==44:
            return np.full(connection0.shape,fill_value=connection0.min(),dtype="uint8")
        # MAXVALUE
        elif func==45:
            return np.full(connection0.shape,fill_value=connection0.max(),dtype="uint8")
        # AVGVALUE
        elif func==46:
            return np.full(connection0.shape,fill_value=connection0.mean(),dtype="uint8")
        # LOCALMIN parameter1 parameter2
        elif func==47:
            if False:
                retval = np.ndarray(connection0.shape,dtype="uint8")
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
                    yd = yd + 1
                    for j in range(0,width):
                        xa = j - x
                        xb = j + x
                        if xa<0:
                            xa = 0
                        if xb>=width:
                            xb = width-1
                        xb = xb + 1
                        retval[i,j] = connection0[yc:yd,xa:xb].min()
                return retval
            elif False:
                height, width = connection0.shape[:2]
                x = abs(parameter1)
                y = abs(parameter2)
                if x==0:
                    x = 1
                if y==0:
                    y = 1
                tmp = np.full((height+2*y,width+2*x),fill_value=255,dtype="uint8")
                tmp[y:height+y,x:width+x] = connection0
                retval = np.ndarray((height,width),dtype="uint8")
                for i in range(y,height+y):
                    for j in range(x,width+x):
                        retval[i-y,j-x] = tmp[i-y:i+y+1,j-x:j+x+1].min()

                return retval
            elif True:
                height, width = connection0.shape[:2]
                x = abs(parameter1)
                y = abs(parameter2)
                if x==0:
                    x = 1
                if y==0:
                    y = 1

                tmp = np.full((height+2*y,width+2*x),fill_value=255,dtype="uint8")
                tmp[y:height+y,x:width+x] = connection0

                pool = Pool(processes=cls.nb_processes,initializer=init_pool_mmmn, initargs=(tmp,y,x))

                indexes = []

                for i in range(y,height+y):
                    for j in range(x,width+x):
                        indexes.append((i,j))

                res = np.array(pool.map(localmin,indexes))

                pool.close()

                return res.reshape((height,width))
            else:
                height, width = connection0.shape[:2]
                x = abs(parameter1)
                y = abs(parameter2)
                if x==0:
                    x = 1
                if y==0:
                    y = 1
                retval = np.ndarray((height+2*y,width+2*x),dtype="uint8")
                indices = [[]] * 256
                for i in range(0,height):
                    for j in range(0,width):
                        #print(connection0[i,j])
                        indices[connection0[i,j]].append((i,j))

                for i in range(255,0,-1):
                    for j in range(0,len(indices[i])):
                        #print(str(i)+" "+str(j))
                        yi,xi = indices[i][j]
                        retval[yi-y:yi+y,xi-x:xi+x] = i

                return retval[y:-y,x:-1]
        # LOCALMAX parameter1 parameter2
        elif func==48:
            if False:
                retval = np.ndarray(connection0.shape,dtype="uint8")
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
                    yd = yd + 1
                    for j in range(0,width):
                        xa = j - x
                        xb = j + x
                        if xa<0:
                            xa = 0
                        if xb>=width:
                            xb = width-1
                        xb = xb + 1
                        retval[i,j] = connection0[yc:yd,xa:xb].max()
                return retval
            elif True:
                height, width = connection0.shape[:2]
                x = abs(parameter1)
                y = abs(parameter2)
                if x==0:
                    x = 1
                if y==0:
                    y = 1

                tmp = np.full((height+2*y,width+2*x),fill_value=0,dtype="uint8")
                tmp[y:height+y,x:width+x] = connection0

                pool = Pool(processes=cls.nb_processes,initializer=init_pool_mmmn, initargs=(tmp,y,x))

                indexes = []

                for i in range(y,height+y):
                    for j in range(x,width+x):
                        indexes.append((i,j))

                res = np.array(pool.map(localmax,indexes))

                pool.close()

                return res.reshape((height,width))
        # LOCALAVG parameter1 parameter2
        elif func==49:
            if False:
                retval = np.ndarray(connection0.shape,dtype="uint8")
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
                    yd = yd + 1
                    for j in range(0,width):
                        xa = j - x
                        xb = j + x
                        if xa<0:
                            xa = 0
                        if xb>=width:
                            xb = width-1
                        xb = xb + 1
                        retval[i,j] = connection0[yc:yd,xa:xb].mean()
                return retval
            elif True:
                height, width = connection0.shape[:2]
                x = abs(parameter1)
                y = abs(parameter2)
                if x==0:
                    x = 1
                if y==0:
                    y = 1

                tmp = np.copy(connection0)

                pool = Pool(processes=cls.nb_processes,initializer=init_pool_mmmn, initargs=(connection0,y,x))

                indexes = []

                for i in range(y,height-y):
                    for j in range(x,width-x):
                        indexes.append((i,j))

                tmp[y:height-y,x:width-x] = np.array(pool.map(localmean,indexes)).reshape((height-2*y,width-2*x))

                pool.close()

                return tmp
        # LOCALNORMALIZE parameter1 parameter2
        elif func==50:
            if False:
                try:
                    retval = np.ndarray(connection0.shape,dtype="uint8")
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
                        yd = yd + 1
                        for j in range(0,width):
                            xa = j - x
                            xb = j + x
                            if xa<0:
                                xa = 0
                            if xb>=width:
                                xb = width-1
                            xb = xb + 1
                            if connection0[i,j] < np.min(connection0[yc:yd,xa:xb]):
                                print(connection0[yc:yd,xa:xb])
                            tmp = (connection0[i,j] - np.min(connection0[yc:yd,xa:xb]))
                            tmp = tmp/max(1,np.max(connection0[yc:yd,xa:xb]))
                            retval[i,j] = (255*(tmp)).astype("uint8")
                except:
                    print(i)
                    print(j)
                    print(yc)
                    print(yd)
                    print(xa)
                    print(xb)
                    print(connection0[i,j])
                    print(np.min(connection0[yc:yd,xa:xb]))
                return retval
            elif True:
                height, width = connection0.shape[:2]
                x = abs(parameter1)
                y = abs(parameter2)
                if x==0:
                    x = 1
                if y==0:
                    y = 1

                tmp = np.copy(connection0)

                pool = Pool(processes=cls.nb_processes,initializer=init_pool_mmmn, initargs=(connection0,y,x))

                indexes = []

                for i in range(y,height-y):
                    for j in range(x,width-x):
                        indexes.append((i,j))

                tmp[y:height-y,x:width-x] = np.array(pool.map(localnormalize,indexes)).reshape((height-2*y,width-2*x))

                pool.close()

                return tmp
