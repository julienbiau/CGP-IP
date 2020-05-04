from .island import Island, IslandProcess
from .chromosome import Chromosome
import random
import os
import cv2
import numpy as np
import copy
from multiprocessing import Queue, Process

class CGPIP:

    def __init__(self, graph_length, mutation_rate, size_of_mutations, num_islands, num_indiv_island, sync_interval_island, max_iterations, chromosomeOptimization, islandOptimization, fitnessFunction):
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
        self.fitnessFunction = fitnessFunction
        #np.seterr(all='ignore')

    def load_data(self,input_data, output_data, num_inputs, num_outputs):
        self.inputs = input_data

        self.outputs = output_data

        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

        self.data_loaded = True

    def load_chromosome(self,filename):
        self.chromosome = Chromosome(0,0,0,self.fitnessFunction)
        self.chromosome.fromFile(filename)

    def set_parent_chromosome(self,chromosome):
        self.chromosome = chromosome

    def getInputs(self):
        return self.inputs

    def getOutputs(self):
        return self.outputs

    def run(self):
        if not self.data_loaded:
            # load data
            print("Load data first")

        if self.islandOptimization==True:
            print("Island optimization")
        elif self.chromosomeOptimization==True:
            print("Chromosome optimization")
        
        for i in range(0,self.num_islands):
            # create island
            island = Island(self.chromosome,self.num_inputs,self.num_outputs,self.graph_length,self.mutation_rate,self.num_indiv_island,self.fitnessFunction)
            self.islands.append(island)
            island.updateParentFitness(self.inputs,self.outputs)

        print("islands created")

        for i in range(0, self.max_iterations):

            if self.islandOptimization==True:
                for j in range(0,self.num_islands):
                    self.islands[j].updateFitnessIsland(self.inputs,self.outputs)

                for j in range(0,self.num_islands):
                    self.islands[j].waitForUpdateFitnessIsland()

                    if self.num_run % 5 == 0:
                        print("Island "+str(j)+" iterations "+str(self.num_run)+" fitness: "+str(self.islands[j].getBestChromosome().getFitness())+" active nodes: "+str(self.islands[j].getBestChromosome().getNbActiveNodes()))
            elif self.chromosomeOptimization==True:
                for j in range(0,self.num_islands):
                    self.islands[j].updateFitnessChromosome(self.inputs,self.outputs)

                for j in range(0,self.num_islands):
                    self.islands[j].waitForUpdateFitnessChromosome()

                    if self.num_run % 5 == 0:
                        print("Island "+str(j)+" iterations "+str(self.num_run)+" fitness: "+str(self.islands[j].getBestChromosome().getFitness())+" active nodes: "+str(self.islands[j].getBestChromosome().getNbActiveNodes())+" duration: "+str(self.islands[j].getBestChromosome().getDuration()))
            else:
                for j in range(0,self.num_islands):
                    self.islands[j].updateFitness(self.inputs,self.outputs)

                    if self.num_run % 5 == 0:
                        print("Island "+str(j)+" iterations "+str(self.num_run)+" fitness: "+str(self.islands[j].getBestChromosome().getFitness())+" active nodes: "+str(self.islands[j].getBestChromosome().getNbActiveNodes()))

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

        islands_best = []
        # update all island with best chromosome
        for j in range(0,self.num_islands):
            islands_best.append(self.islands[j].getBestChromosome())

        best_chromosome = islands_best[0]
        
        for j in range(1,self.num_islands):
            if best_chromosome.getFitness()>islands_best[j].getFitness():
                best_chromosome = islands_best[j]

        best_chromosome.saveFile('./best.txt')
