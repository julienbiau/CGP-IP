from cgpip import CGPIP
from cgpip import Chromosome
from cgpip import Functions
from cgpip import STD_UINT8
import sys
import os
import cv2
import numpy as np
from tqdm import tqdm
import warnings
import time
from numpy.lib.stride_tricks import as_strided
import random

warnings.filterwarnings("error")

def pool2d(A, kernel_size, stride, padding, pool_mode='max'):
    '''
    2D Pooling

    Parameters:
        A: input 2D array
        kernel_size: int, the size of the window
        stride: int, the stride of the window
        padding: int, implicit zero paddings on both sides of the input
        pool_mode: string, 'max' or 'avg'
    '''
    # Padding
    A = np.pad(A, padding, mode='constant', constant_values=255)

    print(A.shape)
    # Window view of A
    output_shape = ((A.shape[0] - kernel_size[0])//stride,
                    (A.shape[1] - kernel_size[1])//stride)

    print(output_shape)
    #kernel_size = (kernel_size, kernel_size)
    A_w = as_strided(A, shape = output_shape + kernel_size, 
                        strides = (stride*A.strides[0],
                                   stride*A.strides[1]) + A.strides)
    #print(A_w.shape)
    #print(A_w)
    A_w = A_w.reshape(-1, *kernel_size)
    print(A_w.shape)
    # Return the result of pooling
    if pool_mode == 'min':
        return A_w.min(axis=(1,2)).reshape(output_shape)
    elif pool_mode == 'max':
        return A_w.max(axis=(1,2)).reshape(output_shape)
    elif pool_mode == 'avg':
        return A_w.mean(axis=(1,2)).reshape(output_shape)
    elif pool_mode == 'norm':
        #(255*(connection0[i,j] - np.min(connection0[yc:yd,xa:xb]))/max(1,np.max(connection0[yc:yd,xa:xb]))).astype("uint8")
        maximums = A_w.max(axis=(1,2))
        #return TODO

def load_data_3_uint8(input_dir,output_dir,max_el):
    inputs = []
    outputs = []

    filenames = sorted(os.listdir(input_dir))
    for i in tqdm(range(max_el)):
        input = cv2.imread(input_dir+"/"+filenames[i])
        inputs.append([np.asarray(input[:,:,0],dtype="uint8"),np.asarray(input[:,:,1],dtype="uint8"),np.asarray(input[:,:,2],dtype="uint8")])

    filenames = sorted(os.listdir(output_dir))
    for i in tqdm(range(max_el)):
        output = cv2.imread(output_dir+"/"+filenames[i])
        outputs.append([np.asarray(output[:,:,0],dtype="uint8"),np.asarray(output[:,:,1],dtype="uint8"),np.asarray(output[:,:,2],dtype="uint8")])

    return inputs, outputs

def load_data_6_1_uint8(input_dir,output_dir,max_el):
    inputs = []
    outputs = []

    filenames = sorted(os.listdir(input_dir))
    for i in tqdm(range(max_el)):
        input = cv2.imread(input_dir+"/"+filenames[i])
        input2 = cv2.cvtColor(input,cv2.COLOR_RGB2HSV)
        inputs.append([np.asarray(input[:,:,0],dtype="uint8"),np.asarray(input[:,:,1],dtype="uint8"),np.asarray(input[:,:,2],dtype="uint8"),np.asarray(input2[:,:,0],dtype="uint8"),np.asarray(input2[:,:,1],dtype="uint8"),np.asarray(input2[:,:,2],dtype="uint8")])

    filenames = sorted(os.listdir(output_dir))
    for i in tqdm(range(max_el)):
        output = cv2.imread(output_dir+"/"+filenames[i])
        outputs.append([np.asarray(output[:,:,1],dtype="uint8")])

    return inputs, outputs

if __name__ == '__main__':

    max_iterations = 300
    size_mutation = 2
    num_islands = 8
    num_indiv = 5
    graph_length = 50
    mutation_rate = 0.05
    sync_interval_island = 150
    batch_size = 10

    Functions.add(STD_UINT8)

    random.seed(7)

    if len(sys.argv)==1:
        cgp = CGPIP(Functions,graph_length,mutation_rate,size_mutation,num_islands,num_indiv,sync_interval_island,max_iterations,True,False,Chromosome.FITNESS_MCC,Chromosome.GOLDMAN_MUTATE,batch_size)

        if os.path.exists('./chromo.txt'):
            cgp.load_chromosome('./chromo.txt')

        inputs, outputs = load_data_6_1_uint8('../CGP-IP-DATA/lunar/images/render','../CGP-IP-DATA/lunar/images/clean',200)
        
        cgp.load_data(inputs, outputs, 6, 1)

        cgp.run()
    elif sys.argv[1]=='display':
        if os.path.exists(sys.argv[2]):
            chromosome = Chromosome(0,0,0,Chromosome.FITNESS_MCC)
            chromosome.fromFile(sys.argv[2])

            chromosome.printGraph()

            cgp = CGPIP(graph_length,mutation_rate,size_mutation,num_islands,num_indiv,sync_interval_island,max_iterations,True,False,Chromosome.FITNESS_MCC)

            inputs, outputs = load_data_6_1_uint8('../CGP-IP-DATA/lunar/images/render','../CGP-IP-DATA/lunar/images/clean',10)
            cgp.load_data(inputs, outputs, 6, 1)

            for j in range(0,len(inputs)):
                print(chromosome.calculateFitness([inputs[j]],[outputs[j]]))

                width, height = inputs[j][0].shape

                i = np.zeros([width,height,3],dtype=np.uint8)
                i[:,:,0] = np.int8(inputs[j][0])
                i[:,:,1] = np.int8(inputs[j][1])
                i[:,:,2] = np.int8(inputs[j][2])

                o = np.zeros([width,height,3],dtype=np.uint8)
                for k in range(0,len(outputs[j])):
                    o[:,:,k] = np.int8(outputs[j][k])

                output_values = chromosome.getOutputValues()

                r = np.zeros([width,height,3],dtype=np.uint8)
                for k in range(0,len(output_values)):
                    r[:,:,k] = np.int8(output_values[k])
                
                print(i)
                print(o)
                print(r)

                cv2.imshow("input", i)

                cv2.imshow("output", o)

                cv2.imshow("result", r)

                cv2.waitKey(0)
    elif sys.argv[1]=='test':
        chromosome = Chromosome(1,1,50,Chromosome.FITNESS_MCC)
        chromosome.random()

        input = np.ndarray((512,512),dtype="uint8")
        inputs = [input]

        for i in range(0,50):
            chromosome.setFunctionForNode(i,i+1)
            chromosome.setConnectionsForNode(i,1,1)

        chromosome.print()
        
        chromosome.setOutputNodes([1])

        chromosome.executeChromosome(inputs,True)

        start_time = time.time()

        pool2d(input, kernel_size=(abs(8)*2, abs(8)*2), stride=1, padding=abs(8), pool_mode='min')

        print(time.time()-start_time)

