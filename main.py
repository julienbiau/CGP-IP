from cgpip import CGPIP
import sys
import os

if __name__ == '__main__':

    max_iterations = 100
    size_mutation = 2
    num_islands = 4
    num_indiv = 5
    graph_length = 200
    mutation_rate = 0.1
    sync_interval_island = 0

    cgp = CGPIP(graph_length,mutation_rate,size_mutation,num_islands,num_indiv,sync_interval_island,max_iterations,True,False)

    if os.path.exists('./chromo.txt'):
        cgp.load_chromosome('./chromo.txt')

    cgp.load_data('../CGP-IP-DATA/inputs','../CGP-IP-DATA/outputs')

    cgp.run()
