from .chromosome import Chromosome, ChromosomeProcess
import copy
from multiprocessing import Queue

class Island:

    def __init__(self,chromosome,num_inputs,num_outputs,graph_length,mutation_rate,num_indiv):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.graph_length = graph_length
        self.mutation_rate = mutation_rate
        self.emu = 1
        self.elambda = num_indiv - self.emu
        self.parent = Chromosome(self.num_inputs,self.num_outputs,self.graph_length)
        self.childs = []
        self.best_chromosome = None
        self.process = []
        self.queues = []
        if chromosome==None:
            self.parent.random()
        else:
            self.parent = copy.deepcopy(chromosome)

        for i in range(0,self.elambda):
            self.childs.append(Chromosome(self.num_inputs,self.num_outputs,self.graph_length))

            if chromosome==None:
                self.childs[i].random()
            else:
                self.childs[i] = copy.deepcopy(chromosome)

    def updateParentFitness(self,input_data,output_data):
        self.parent.calculateFitnessMeanError(input_data,output_data)

    def updateFitness(self,input_data,output_data):
        for i in range(0,self.elambda):
            self.childs[i].calculateFitnessMeanError(input_data,output_data)

        self.setBestChromosome()

    def updateFitnessChromosome(self,input_data,output_data):
        for i in range(0,self.elambda):
            q = Queue()
            c = ChromosomeProcess(self.childs[i],input_data,output_data,q)
            c.start()
            self.process.append(c)
            self.queues.append(q)

    def waitForUpdateFitnessChromosome(self):
        for i in range(0,len(self.process)):
            self.process[i].join()

        for i in range(0,len(self.process)):
            self.childs[i].setFitness(self.queues[i].get())

        self.setBestChromosome()

        self.process = []
        self.queues = []

    def setBestChromosome(self):
        self.best_chromosome = self.parent
        for i in range(0,self.elambda):
            if self.childs[i].getFitness()<=self.best_chromosome.getFitness():
                self.best_chromosome = self.childs[i]

    def doEvolution(self):
        self.parent = copy.deepcopy(self.best_chromosome)

        for i in range(0,self.elambda):
            self.childs[i] = copy.deepcopy(self.parent)
        
        for i in range(0,self.elambda):
            self.childs[i].mutate(self.mutation_rate)

    
    def getBestChromosome(self):
        return self.best_chromosome

    def updateBestChromosome(self,chromosome):
        self.best_chromosome = chromosome
