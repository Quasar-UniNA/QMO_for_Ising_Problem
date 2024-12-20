from ising_problem import Ising
from GA_Optimization import GA_Optimizer
import random
from deap import creator, base, tools
from qiskit import IBMQ
import quantum_mating_operator as QMO
from init import getPop
import io
import sys


#CUSTOM OPERATORS
def quantum_mating(offspring, cx_pb, mut_pb, prob_1=0, prob_2=0, p0given1=0.1, p1given0=0.05, grid_size=5):
    #Build Custom Noise Model
    noise_model = QMO.noise_model(prob_1=0, prob_2=0, p0given1=p0given1, p1given0=p1given0)
    #Define QMO operator
    QMO.qmo(pop=offspring, ind_size=(grid_size**2), cx_pb=cx_pb, m_pb=mut_pb, draw_qc=False,
            creator_ind=GA.deap_creator.Individual, size_sub_probl=10, noise_model=noise_model)
    return offspring

def one_point(offspring, cx_pb):
    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < cx_pb:
            tools.cxOnePoint(child1, child2)
            del child1.fitness.values
            del child2.fitness.values

def two_point(offspring, cx_pb):
    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < cx_pb:
            tools.cxTwoPoint(child1, child2)
            del child1.fitness.values
            del child2.fitness.values

def uniform_x(offspring, cx_pb):
    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < cx_pb:
            tools.cxUniform(child1, child2, cx_pb)
            del child1.fitness.values
            del child2.fitness.values

def a2s(arr):
    string = ""
    for i in arr:
        string += (str(i)+" ")
    return string + "\n"
def getInfo(conf):
    file = open(conf, "r")
    a = file.read()
    file.close()
    a = a.split("\n")
    a.pop()
    b = [float(i) for i in a]
    return int(a[0]), b[1], b[2::]

class GA_for_Ising:
    def __init__(self, conf, popfile=None, popsize=10):
        self.conf = conf
        self.popfile = popfile
        self.popsize = popsize
    def execute(self, operator="qmo", nlev = 0):
        text_trap = io.StringIO()
        sys.stdout = text_trap
        d, s, farr = getInfo(self.conf)
        ip = Ising(d, self.conf)
        ip.setup()
        global GA
        GA = GA_Optimizer(problem_size=(d**2),  verbose=True)
        GA.set_Fitness_Function(ip.evaluate)
        bf = nlev/20
        if operator=="uniform":
            uniform_x1 = GA.toolbox.register('custom_cx', uniform_x, cx_pb=0.8)
            if self.popfile != None:
                GA.start_GA(pop_size=self.popsize, pop_list=getPop(self.popfile, 0))
            else:
                GA.start_GA(pop_size=self.popsize)
            GA.optimize(elitism=True, sel=True,  cx=True, mut=True, max_gen=100, max_evals=1e5, custom_cx=uniform_x1, mut_pb=0.2)
        if operator=="1-point":
            opoint = GA.toolbox.register('custom_cx', one_point, cx_pb=0.8)
            if self.popfile != None:
                GA.start_GA(pop_size=self.popsize, pop_list=getPop(self.popfile, 0))
            else:
                GA.start_GA(pop_size=self.popsize)
            GA.optimize(elitism=True, sel=True,  cx=True, mut=True, max_gen=100, max_evals=1e5, custom_cx=opoint, mut_pb=0.3)
        if operator=="2-point":
            tpoint = GA.toolbox.register('custom_cx', two_point, cx_pb=0.8)
            if self.popfile != None:
                GA.start_GA(pop_size=self.popsize, pop_list=getPop(self.popfile, 0))
            else:
                GA.start_GA(pop_size=self.popsize)
            GA.optimize(elitism=True, sel=True,  cx=True, mut=True, max_gen=100, max_evals=1e5, custom_cx=tpoint, mut_pb=0.2)
        if operator=="qmo":
            qmat = GA.toolbox.register('custom_cx', quantum_mating, cx_pb=0.7, grid_size=d, mut_pb=0.15, p0given1=bf, p1given0=bf)
            if self.popfile != None:
                GA.start_GA(pop_size=self.popsize, pop_list=getPop(self.popfile, 0))
            else:
                GA.start_GA(pop_size=self.popsize)
            GA.optimize(elitism=True, sel=True,  cx=True, mut=True, max_gen=100, max_evals=1e5, custom_cx=qmat)
            
        sys.stdout = sys.__stdout__
        return GA.getBest()
alg = GA_for_Ising(conf="conf1.txt")#Devo capire se devo passare le variabili pure dopo class
print(alg.execute("qmo"))
