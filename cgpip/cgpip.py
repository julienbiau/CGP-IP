from .island import Island, IslandProcess
from .chromosome import Chromosome
import random
import os
import cv2
import numpy as np
import copy
from multiprocessing import Queue, Process

class CGPIP:

    def __init__(self, graph_length, mutation_rate, size_of_mutations, num_islands, num_indiv_island, sync_interval_island, max_iterations, chromosomeOptimization, islandOptimization):
        self.graph_length = graph_length
        self.mutation_rate = mutation_rate
        self.size_of_mutations = size_of_mutations
        self.num_islands = num_islands
        self.islands = []
        self.num_indiv_island = num_indiv_island
        self.sync_interval_island = sync_interval_island
        self.max_iterations = max_iterations
        self.num_run = 0
        self.inputs = None
        self.outputs = None
        self.num_inputs = 0
        self.num_outputs = 0
        self.chromosome = None
        self.chromosomeOptimization = chromosomeOptimization
        self.islandOptimization = islandOptimization
        np.seterr(all='ignore')

    def load_data(self,input_data, output_data):
        self.inputs = []

        for filename in sorted(os.listdir(input_data)):
          input = cv2.imread(input_data+"/"+filename)
          self.inputs.append([input[:,:,0],input[:,:,1],input[:,:,2]])

        self.outputs = []

        for filename in sorted(os.listdir(output_data)):
          output = cv2.imread(output_data+"/"+filename)
          self.outputs.append([output[:,:,0],output[:,:,1],output[:,:,2]])

        self.num_inputs = 3
        self.num_outputs = 3

        self.data_loaded = True

    def load_chromosome(self,filename):
        self.chromosome = Chromosome(0,0,0)
        self.chromosome.fromFile(filename)

    def run(self):
        if not self.data_loaded:
            # load data
            print("Load data first")

        for i in range(0,self.num_islands):
            # create island
            island = Island(self.chromosome,self.num_inputs,self.num_outputs,self.graph_length,self.mutation_rate,self.num_indiv_island)
            self.islands.append(island)
            island.updateParentFitness(self.inputs,self.outputs)

        for i in range(0, self.max_iterations):

            if self.islandOptimization==True:
                for j in range(0,self.num_islands):
                    self.islands[j].updateFitnessIsland(self.inputs,self.outputs)

                for j in range(0,self.num_islands):
                    self.islands[j].waitForUpdateFitnessIsland()

                    if self.num_run % 5 == 0:
                        print("Island "+str(j)+" iterations "+str(self.num_run)+" fitness: "+str(self.islands[j].getBestChromosome().getFitness()))
            elif self.chromosomeOptimization==True:
                for j in range(0,self.num_islands):
                    self.islands[j].updateFitnessChromosome(self.inputs,self.outputs)

                for j in range(0,self.num_islands):
                    self.islands[j].waitForUpdateFitnessChromosome()

                    if self.num_run % 5 == 0:
                        print("Island "+str(j)+" iterations "+str(self.num_run)+" fitness: "+str(self.islands[j].getBestChromosome().getFitness()))
            else:
                for j in range(0,self.num_islands):
                    self.islands[j].updateFitness(self.inputs,self.outputs)

                    if self.num_run % 5 == 0:
                        print("Island "+str(j)+" iterations "+str(self.num_run)+" fitness: "+str(self.islands[j].getBestChromosome().getFitness()))

            self.num_run = self.num_run + 1

            if self.sync_interval_island>0 and self.num_run % self.sync_interval_island:
                islands_best = []
                # update all island with best chromosome
                for j in range(0,self.num_islands):
                    islands_best.append(self.islands[j].getBestChromosome())

                best_chromosome = islands_best[0]

                for j in range(1,self.num_islands):
                    if best_chromosome.getFitness()>islands_best[j].getFitness():
                        best_chromosome = islands_best[j]

                print("Fitness: "+str(best_chromosome.getFitness()))

                for j in range(0,self.num_islands):
                    self.islands[j].updateBestChromosome(best_chromosome)

            for j in range(0,self.num_islands):
                self.islands[j].doEvolution()
