import random,math
from qiskit import QuantumCircuit, QuantumRegister, execute, Aer, IBMQ, BasicAer
from qiskit.test.mock import FakeSydney
from qiskit.providers.aer.noise import NoiseModel, depolarizing_error
from qiskit.providers.aer.noise import ReadoutError
import copy
from qiskit.providers.aer.backends import QasmSimulator



def generate_ind_from_count(creator_ind, final_pop, counts):
    """
    Function appending to final_pop the states in the counts vector as list of
    integers in the correct order.
    ...
    :param final_pop: pop in which the new individual will be appended
    :param counts: counts vector for qc measurement
    ...
    :return: None
    """
    for state in list(counts.keys()):
        #print(state)
        ind = []
        for b in range(len(state)):
            ind.append(int(state[-1-b]))
        for occurence in range(counts[state]):
            final_pop.append(creator_ind(ind))


def compute_frequencies(ind_list):
    """
    Function compunting the occorence of ones position
    by position of the individuals in the list ind_list.
    ...
    :param ind_list: list of individuals to mate
    ...
    :return: one_frequences as dictionary.
    """
    ind_size = len(ind_list[0])
    one_frequences = {i:0 for i in range(ind_size)}
    for bit in range(ind_size):
        sum_bit = 0
        for ind in ind_list:
            sum_bit = sum_bit + ind[bit]
        one_frequences[bit] = sum_bit/len(ind_list)
    return one_frequences


def noise_model(prob_1=0.001, prob_2=0.01, p0given1=0.1, p1given0=0.05):
    """
    Build noise model for simulation.
    ...
    :param prob_1: probability single qubit gate error
    :param prob_2: probability two-qubits gate error
    :param p0given1: readout probability that 1 is flipped in 0
    :param p1given0: readout probability that 0 is flipped in 1
    :return: noise model
    """

    # Depolarizing quantum errors
    error_1 = depolarizing_error(prob_1, 1)
    error_2 = depolarizing_error(prob_2, 2)

    # Measurement miss-assignement probabilities
    error_rd = ReadoutError([[1 - p1given0, p1given0], [p0given1, 1 - p0given1]])

    # Add errors to noise model
    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(error_1, ['u1', 'u2', 'u3'])
    noise_model.add_all_qubit_quantum_error(error_2, ['cx'])
    noise_model.add_all_qubit_readout_error(error_rd)

    # Get basis gates from noise model
    #basis_gates = noise_model.basis_gates
    return noise_model



def qmo(pop, ind_size, cx_pb, m_pb, creator_ind, draw_qc=False, **kwargs):
    """
    Function implementing QMO operator. By default, QMO works simulating ideally the quantum circuit created.
    Real quantum devices from IBM Quantum can be used specifying the backend argument.
    QMO modifies in place pop.
    QMO can be distributed according the D-NISQ reference model, by specifying the sub problems size with size_sub_prob
    parameter. It will create several quantum circuits with size at maximum equals to size_sub_prob.
    ...
    :param (list) pop: genetic population to mate;
    :param (int) ind_size: problem size;
    :param (float) cx_pb: probability of crossover;
    :param (float) m_pb: probability of mutation;
    :param (DEAP Creator) creator_ind: Individual Creator Object from Deap;
    :param (Bool - default False) draw_qc: Show QMO quantum circuit if TRUE.
    ...
    :keyword (int) **size_sub_prob: sub problem size;
    :keyword **provider: IBMQ provider if real backands or IBMQ simulators have to be used;
    :keyword **backend: IBMQ backend object;
    :keyword **noise_model: Qiskit Noise Model Object. If specified, it will be considered in the simulation;
    ...
    :return:
    """

    # Check if QMO has to be distributed according to D-NISQ.
    if 'size_sub_prob' in kwargs: size_sub_prob = kwargs['size_sub_prob']
    else: size_sub_prob = ind_size

    to_consider, to_not_consider = [],[]
    # if random < cx_pb then consider the individual for crossover
    for ind in pop:
        if random.random() < cx_pb:
            del ind.fitness.values
            to_consider.append(ind)
            pop.remove(ind)
    # Build Quantum Circuit
    if len(to_consider)>0:
        rotations = compute_frequencies(to_consider)
        sub_rot = [list(rotations.values())[i:i+size_sub_prob] for i in range(0,ind_size, size_sub_prob)]
        for iteration in range(len(to_consider)):

            qr = [QuantumRegister(len(sub_rot[i])) for i in range(len(sub_rot))]
            qc = [QuantumCircuit(i) for i in qr]
            counts_list = []
            for sub_prob in range(len(sub_rot)):
                for bit in range(len(sub_rot[sub_prob])):
                    #print('bit', bit)
                    qc[sub_prob].ry(math.pi*sub_rot[sub_prob][bit], qr[sub_prob][bit])
                    if random.random() < m_pb:
                        #print('target', target)
                        qc[sub_prob].ry(math.pi*random.random(),qr[sub_prob][bit])
                qc[sub_prob].measure_all()
                if draw_qc:
                    if 'size_sub_prob' in kwargs:
                        print('plotting list of sub circuits')
                    else: print('plot QMO circuit')
                    for i in range(size_sub_prob):
                        qc[i].draw('mpl').show()

                # Execute qc
                if 'backend' not in kwargs:
                    # QASM SIMULATOR
                    backend = Aer.get_backend('qasm_simulator')
                    if 'noise_model' in kwargs:
                        job = execute(qc[sub_prob], backend, noise_model=kwargs['noise_model'], shots=1, seed_simulator=random.randint(1, 150))
                    else:
                        job = execute(qc[sub_prob], backend, shots=1, seed_simulator=random.randint(1, 150))
                    result = job.result()
                    counts = result.get_counts()
                    counts_list.append(counts)

                # CUSTOM BACKEND
                else:
                    if 'noise_model' in kwargs:
                        job = execute(qc[sub_prob], kwargs['backend'], noise_model=kwargs['noise_model'], shots=1,
                                      seed_simulator=random.randint(1, 150))
                    else:
                        job = execute(qc[sub_prob], kwargs['backend'], shots=1, seed_simulator=random.randint(1, 150))
                    result = job.result()
                    counts = result.get_counts()
                    counts_list.append(counts)
            final_count = list(counts_list[0].keys())[0]
            for i in range(1, len(counts_list)):
                final_count += list(counts_list[i].keys())[0]
            generate_ind_from_count(creator_ind, pop, {final_count:1})
