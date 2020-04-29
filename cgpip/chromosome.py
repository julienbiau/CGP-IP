from .functions import Functions
import random
import math
import numpy as np
from multiprocessing import Process, Queue

class ChromosomeProcess(Process):
    def __init__(self,chromosome,input_data,output_data,queue):
        super(ChromosomeProcess, self).__init__()
        self.chromosome = chromosome
        self.input_data = input_data
        self.output_data = output_data
        self.queue = queue

    def run(self):
        self.queue.put(self.chromosome.calculateFitnessMeanError(self.input_data,self.output_data))

class Node:
    # Function Int
    # Connection 0 Int
    # Connection 1 Int
    # Parameter 0 Real no limitation
    # Parameter 1 Int [−16, +16]
    # Parameter 2 Int [−16, +16]
    # Gabor Filter Frequ. Int [0, 16]
    # Gabor Filter Orient. Int [−8, +8]

    def __init__(self,num_inputs,num_outputs,graph_length,random_init=True):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.graph_length = graph_length

        if random_init:
            self.getRandomFunction()
            self.getRandomConnection0()
            self.getRandomConnection1()

            self.getRandomParameter0()
            self.getRandomParameter1()
            self.getRandomParameter2()

            self.getRandomGaborFilterFrequence()
            self.getRandomGaborFilterOrientation()

    def setValues(self,function,connection0,connection1,parameter0,parameter1,parameter2,gaborFilterFrequence,gaborFilterOrientation):
        self.function = function
        self.connection0 = connection0
        self.connection1 = connection1
        self.parameter0 = parameter0
        self.parameter1 = parameter1
        self.parameter2 = parameter2
        self.gaborFilterFrequence = gaborFilterFrequence
        self.gaborFilterOrientation = gaborFilterOrientation

    def getRandomFunction(self):
        self.function = Functions.getRandomFunction()

    def getRandomConnection0(self):
        try:
            self.connection0 = random.randrange(1,self.graph_length+self.num_inputs)
        except ValueError:
            self.connection0 = 1

    def getRandomConnection1(self):
        try:
            self.connection1 = random.randrange(1,self.graph_length+self.num_inputs)
        except ValueError:
            self.connection1 = 1

    def getRandomParameter0(self):
        self.parameter0 = random.uniform(-2^32,2^32)

    def getRandomParameter1(self):
        self.parameter1 = random.randrange(-16, 16, 1)

    def getRandomParameter2(self):
        self.parameter2 = random.randrange(-16, 16, 1)

    def getRandomGaborFilterFrequence(self):
        self.gaborFilterFrequence = random.randrange(0, 16, 1)

    def getRandomGaborFilterOrientation(self):
        self.gaborFilterOrientation = random.randrange(-8, 8, 1)

    def getFunction(self):
        return self.function

    def getConnection0(self):
        return self.connection0

    def getConnection1(self):
        return self.connection1

    def getParameter0(self):
        return self.parameter0

    def getParameter1(self):
        return self.parameter1

    def getParameter2(self):
        return self.parameter2

    def getGaborFilterFrequence(self):
        return self.gaborFilterFrequence

    def getGaborFilterOrientation(self):
        return self.gaborFilterOrientation

    def execute(self,value0,value1):
        return Functions.execute(self.function,value0,value1,self.parameter0,self.parameter1,self.parameter2,self.gaborFilterFrequence,self.gaborFilterOrientation)

class Chromosome:

    def __init__(self,num_inputs,num_outputs,graph_length):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.graph_length = graph_length
        self.output_nodes = []
        self.output_values = []
        self.inputs_index = 0
        self.nodes = []
        self.active_nodes = []
        self.nodes_value = []
        self.fitness = None

    def fromFile(self,filename):
        file = open(filename,"r")

        lines = file.readlines()

        line = lines[0].strip('\n')
        self.num_inputs = int(line.split(' ')[0])
        self.num_outputs = int(line.split(' ')[1])
        self.graph_length = int(line.split(' ')[2])

        for i in range(1,len(lines)):
            line = lines[i].strip('\n')

            data = line.split(' ')
            if len(data)==1:
                self.output_nodes.append(int(data[0]))
            else:
                node = Node(self.num_inputs,self.num_outputs,i-1,False)
                node.setValues(int(data[0]),int(data[1]),int(data[2]),float(data[3]),int(data[4]),int(data[5]),int(data[6]),int(data[7]))
                self.nodes.append(node)

        file.close()

        self.updateActiveNodes()

    def random(self):
        for i in range(0, self.graph_length):
            self.nodes.append(Node(self.num_inputs,self.num_outputs,i,True))

        for i in range(0,self.num_outputs):
            self.output_nodes.append(random.randrange(1,self.graph_length))

        self.updateActiveNodes()

    def getOutputValues(self):
        return self.output_values
        
    def mutate(self,mutation_rate):
        nb_mutations = math.floor((self.graph_length*8 + self.num_outputs)*mutation_rate)

        for i in range(0,nb_mutations):
            index = random.randrange(0, self.graph_length*8 + self.num_outputs, 1)

            if index < self.graph_length*8:
                parameter = index % 8
                node_index = int((index - parameter)/8)

                # nodes
                if parameter == 0:
                    self.nodes[node_index].getRandomFunction()
                elif parameter == 1:
                    self.nodes[node_index].getRandomConnection0()
                elif parameter == 2:
                    self.nodes[node_index].getRandomConnection1()
                elif parameter == 3:
                    self.nodes[node_index].getRandomParameter0()
                elif parameter == 4:
                    self.nodes[node_index].getRandomParameter1()
                elif parameter == 5:
                    self.nodes[node_index].getRandomParameter2()
                elif parameter == 6:
                    self.nodes[node_index].getRandomGaborFilterFrequence()
                elif parameter == 7:
                    self.nodes[node_index].getRandomGaborFilterOrientation()
            else:
                # outputs
                self.output_nodes[index-self.graph_length*8] = random.randrange(1,self.graph_length,1)

        self.updateActiveNodes()


    def calculateFitnessMCC(self,output_data):
        return 0

    def calculateFitnessMeanError(self,input_data,output_data):
        mean = 0

        for i in range(0, len(input_data)):
            self.executeChromosome(input_data[i])

            for j in range(0, len(output_data[i])):
                mean = mean + abs(output_data[i][j]-self.output_values[j]).sum()/output_data[i][j].size

        mean = mean / len(input_data)

        self.fitness = mean

        return mean

    def updateActiveNodes(self):
        nodes_to_check = []
        self.active_nodes = []

        for i in self.output_nodes:
            nodes_to_check.append(self.graph_length+self.num_inputs-i)

        while len(nodes_to_check)>0:
            node_to_check = nodes_to_check.pop(0)

            self.active_nodes.append(node_to_check)

            if node_to_check-self.nodes[node_to_check-self.num_inputs].getConnection0()>=self.num_inputs:
                nodes_to_check.append(node_to_check-self.nodes[node_to_check-self.num_inputs].getConnection0())

            if node_to_check-self.nodes[node_to_check-self.num_inputs].getConnection1()>=self.num_inputs:
                nodes_to_check.append(node_to_check-self.nodes[node_to_check-self.num_inputs].getConnection1())

        self.active_nodes = list(set(self.active_nodes))
        self.active_nodes.sort()
        
    def executeChromosome(self,input_data):
        try:
            self.nodes_value = [None] * (self.num_inputs+self.graph_length)
            self.output_values = []

            for i in range(0, len(input_data)):
                self.nodes_value[i] = input_data[i]

            for i in self.active_nodes:
                # INP
                if self.nodes[i-self.num_inputs].getFunction()==1:
                    self.inputs_index = (self.inputs_index+1)%self.num_inputs
                    self.nodes_value[i] = self.nodes_value[self.inputs_index]
                # INPP
                elif self.nodes[i-self.num_inputs].getFunction()==2:
                    self.inputs_index = (self.inputs_index-1)%self.num_inputs
                    self.nodes_value[i] = self.nodes_value[self.inputs_index]
                # SKIP
                elif self.nodes[i-self.num_inputs].getFunction()==3:
                    self.inputs_index = (self.inputs_index+math.floor(self.nodes[i-self.num_inputs].getParameter0()))%self.num_inputs
                    self.nodes_value[i] = self.nodes_value[self.inputs_index]
                else:
                    self.nodes_value[i] = self.nodes[i-self.num_inputs].execute(self.nodes_value[i-self.nodes[i-self.num_inputs].getConnection0()],self.nodes_value[i-self.nodes[i-self.num_inputs].getConnection1()])

            for i in range(0,len(self.output_nodes)):
                self.output_values.append(self.nodes_value[self.graph_length+self.num_inputs-self.output_nodes[i]])
        except:
            print("Exception node "+str(i))
            print(str(self.nodes[i-self.num_inputs].getFunction()))
            print(str(self.nodes[i-self.num_inputs].getParameter0()))
            print(str(self.nodes[i-self.num_inputs].getParameter1()))
            print(str(self.nodes[i-self.num_inputs].getParameter2()))
            print(self.nodes_value[i-self.nodes[i-self.num_inputs].getConnection0()])
            print(self.nodes_value[i-self.nodes[i-self.num_inputs].getConnection1()])
            self.saveFile()
            raise

    def getFitness(self):
        return self.fitness

    def setFitness(self,fitness):
        self.fitness = fitness

    def print(self):
        print("Nb inputs: "+str(self.num_inputs))
        for i in range(0, self.graph_length):
            print("Node "+str(i+self.num_inputs)+" "+str(self.nodes[i].getFunction())+" "+str(self.nodes[i].getConnection0())+" "+str(self.nodes[i].getConnection1())+" "+str(self.nodes[i].getParameter0())+" "+str(self.nodes[i].getParameter1())+" "+str(self.nodes[i].getParameter2()))

        print("Nb outputs: "+str(self.num_outputs))
        for i in range(0, self.num_outputs):
            print(str(self.output_nodes[i]))
    
    def saveFile(self):
        file = open("./chromo.txt","w")

        file.write(str(self.num_inputs)+" "+str(self.num_outputs)+" "+str(self.graph_length)+"\n")

        for i in range(0,self.graph_length):
            file.write(str(self.nodes[i].getFunction())+" "+str(self.nodes[i].getConnection0())+" "+str(self.nodes[i].getConnection1())
                +" "+str(self.nodes[i].getParameter0())+" "+str(self.nodes[i].getParameter1())+" "+str(self.nodes[i].getParameter2())
                +" "+str(self.nodes[i].getGaborFilterFrequence())+" "+str(self.nodes[i].getGaborFilterOrientation())+"\n")

        for i in range(0,self.num_outputs):
            file.write(str(self.output_nodes[i])+"\n")

        file.close()