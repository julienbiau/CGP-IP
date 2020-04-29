from cgpip import CGPIP
from cgpip import Chromosome
import sys
import os
import cv2
import numpy as np

if __name__ == '__main__':

    max_iterations = 100000
    size_mutation = 2
    num_islands = 4
    num_indiv = 5
    graph_length = 200
    mutation_rate = 0.1
    sync_interval_island = 0

    if len(sys.argv)==1:
        cgp = CGPIP(graph_length,mutation_rate,size_mutation,num_islands,num_indiv,sync_interval_island,max_iterations,True,False)

        if os.path.exists('./chromo.txt'):
            cgp.load_chromosome('./chromo.txt')

        cgp.load_data('../CGP-IP-DATA/inputs','../CGP-IP-DATA/outputs')

        cgp.run()
    elif sys.argv[1]=='display':
        if os.path.exists('./display.txt'):
            chromosome = Chromosome(0,0,0)
            chromosome.fromFile('./display.txt')

            cgp = CGPIP(graph_length,mutation_rate,size_mutation,num_islands,num_indiv,sync_interval_island,max_iterations,True,False)

            cgp.load_data('../CGP-IP-DATA/inputs','../CGP-IP-DATA/outputs')

            inputs = cgp.getInputs()
            outputs = cgp.getOutputs()

            for j in range(0,len(inputs)):
                chromosome.executeChromosome(inputs[j])

                width, height = inputs[j][0].shape

                i = np.zeros([width,height,3],dtype=np.uint8)
                i[:,:,0] = np.int8(inputs[j][0])
                i[:,:,1] = np.int8(inputs[j][1])
                i[:,:,2] = np.int8(inputs[j][2])

                output_values = chromosome.getOutputValues()

                o = np.zeros([width,height,3],dtype=np.uint8)
                o[:,:,0] = np.int8(output_values[0])
                o[:,:,1] = np.int8(output_values[1])
                o[:,:,2] = np.int8(output_values[2])

                print(o)

                cv2.imshow("image", i)

                cv2.imshow("image_output", o)

                cv2.waitKey(0)


