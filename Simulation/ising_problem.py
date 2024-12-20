import random, statistics
import numpy as np

def converter(sol, n):
    out = []
    for i in range(n):
        o = []
        for j in range(n):
            o.append(int(2*(sol[i*n+j]-0.5))) #Map 0s to -1s (is the best way?)
        out.append(o)
    return np.asarray(out)
def confLoad(conf):
    f = open(conf, "r")
    a = f.read()
    f.close()
    a = a.split("\n")
    n = int(a[0])
    R = np.zeros((n,n-1))
    C = np.zeros((n-1,n))
    for i in range(n):
        for j in range(n-1):
            R[i,j]=float(a[(n-1)*i+j])
    for i in range(n-1):
        for j in range(n):
            C[i,j]=float(a[n*i+j+n*(n-1)])
    return R,C

def val(i, j, S, n): #Return the value of S in that point, if the point does not exist returns 0.
    if i>=0 and i<n and j>=0 and j<n:
        return S[i, j]
    else:
        return 0

def fitness(inp, n, conf):
    S = converter(inp, n)
    R,C = confLoad(conf) 
    file = open(conf, "r")
    a = file.read()
    file.close()
    a = a.split("\n")
    H = 0 #Hamiltonian value of the system (external field=0)
    for i in range(n):
        for j in range(n):
            if i>0:
                H+=C[i-1, j]*S[i-1,j]*S[i,j]
            if i<(n-1):
                H+=C[i, j]*S[i+1, j]*S[i,j]
            if j>0:
                H+=R[i, j-1]*S[i, j-1]*S[i,j]
            if j<(n-1):
                H+=R[i, j]*S[i, j+1]*S[i,j]
    return -1*(H/2)

def rn():
    if random.random()>0.5:
        return 1
    else:
        return 0

class Ising():
    """
    Class Implementing the Ising problem
    """
    def __init__(self, gs, conf):
        """
        :param gs: grid size (grid size)
        """
        self.gs = gs
        self.N = gs**2 #To see if this line is needed...
        self.conf = conf
    def setup(self, spin=None):
        """
        Setup the Ising problem.
        By default the configuration of the system will be randomly initialized.
        You can specify a initial solution passing it in object creation.
        :param spin: spin array (0 for -1 and 1 for +1)
        :return:None
        """

        if spin==None:
            self.spin=[rn() for i in range(0, self.gs**2)]
        else:
            self.spin = spin
    def evaluate(self, solution, verbose=False):
        """
        Evaluate a solution
        :param solution: candidate solution
        :param verbose: False by default
        :return: value of the solution. -1 if it is invalid.
        """
        if verbose:
            print('candidate solution ', solution)
        return fitness(solution, self.gs, self.conf)

