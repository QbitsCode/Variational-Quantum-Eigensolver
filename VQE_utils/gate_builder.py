from math import pi

from qiskit import execute, Aer, QuantumRegister, QuantumCircuit
from qiskit.quantum_info.operators.symplectic.pauli import Pauli

from VQE_utils import build_HF
from myutils import getStrFinalRes


def init_gate(instance, Measure_State=False):
    hfstate = build_HF(instance)
    excstate1 = {"H2": {'sto-3g': [0, 1], 'H-1s-sto3g & H-1s-sto3g': [0, 1]}, "LiH": {'H-1s-sto3g & Li-1,2s-ccpvdz': [0, 3, 4, 5], 'H-1s-sto3g & Li-1,2,3s-ccpvdz': [0, 1, 2, 4]}}
    excstate2p = {"LiH": {'H-1s-sto3g & Li-1,2s-ccpvdz': [0, 1, 3, 5]}}
    excstate3 = {"LiH": {'H-1s-sto3g & Li-1,2s-ccpvdz': [0, 2, 3, 5]}}
    excstate5 = {"LiH": {'H-1s-sto3g & Li-1,2s-ccpvdz': [2, 3, 4, 5]}}
    excstate7 = {"LiH": {'H-1s-sto3g & Li-1,2s-ccpvdz': [1, 2, 4, 5]}}

    init_qr = QuantumRegister(instance.nqbits)
    gate = QuantumCircuit(init_qr)
    if instance.refstate == 'HF':
        gate.x(hfstate)
    elif instance.refstate == 'exc1':
        gate.x(excstate1[instance.molecule.name][instance.molecule.basis_name])
    elif instance.refstate == 'exc2':
        if instance.molecule.name == 'H2':
            gate.h([0])
            gate.x([2, 1])
            gate.cx(0, 1)                                                       #makes 0.707 ( |0110> + |1001> )
            gate.cx(0, 2)
            gate.cx(0, 3)
        elif instance.molecule.name == 'LiH':
            gate.h([1])
            gate.x([2, 4])
            gate.cx(1, 2)                                                       #makes 0.707 ( |011101> + |101011> )
            gate.cx(1, 4)
            gate.cx(1, 5)
            gate.x([0, 3])
    elif instance.refstate == 'exc2p':
        gate.x(excstate2p[instance.molecule.name][instance.molecule.basis_name])
    elif instance.refstate == 'exc3':
        gate.x(excstate3[instance.molecule.name][instance.molecule.basis_name])
    elif instance.refstate == 'exc5':
        gate.x(excstate5[instance.molecule.name][instance.molecule.basis_name])
    elif instance.refstate == 'exc7':
        gate.x(excstate7[instance.molecule.name][instance.molecule.basis_name])
    else:
        raise NotImplementedError('refstate ' + instance.refstate)              #could add a initMP2

    if Measure_State:
        _backend = Aer.get_backend(instance.backend)                            #Simulation          qasm_simulator  ;  statevector_simulator
        result = execute(gate, _backend).result().data(0)                       #DEBUG
        return getStrFinalRes(result)

    Uinit = gate.to_gate(label='Init')
    if instance.wordiness > 2:
        print(gate.draw())
    return Uinit


def ent_gate(instance, ):                                                       #Entanglement
    ent_qr = QuantumRegister(instance.nqbits)
    ent_gate = QuantumCircuit(ent_qr)
    ent_gate.cx(0, 1)
    ent_gate.cx(1, 2)
    ent_gate.cx(2, 3)
    ent_gate.cx(1, 2)
    ent_gate.cx(0, 1)
    Uent = ent_gate.to_gate(label='Uent')
    if instance.wordiness > 2:
        print(ent_gate.draw())                                                  #DEBUG
    return Uent


def postrot_gate(instance, POps):
    pr_qr = QuantumRegister(instance.nqbits)
    pr_gate = QuantumCircuit(pr_qr)
    prstr = instance.postRots_dict[POps[0]][0]
    if prstr == 'i' * instance.nqbits:
        pass
    else:
        for n in range(instance.nqbits):
            if prstr[instance.nqbits - 1 - n] == 'i':
                pass
            elif prstr[instance.nqbits - 1 - n] == 'h':
                pr_gate.h(n)
            elif prstr[instance.nqbits - 1 - n] == 'r':
                pr_gate.rx(-pi / 2, n)
            else:
                print("couldn't build post rotations")

    if instance.wordiness > 2:
        print(pr_gate.draw())                                                   #DEBUG
    postRots = pr_gate.to_gate(label='postRots' + str(POps))
    return postRots


def UCC_gate(instance, angles, na):
    ucc_qr = QuantumRegister(instance.nqbits)
    gate = QuantumCircuit(ucc_qr)

    for idx in range(instance.nexc):
        for term in instance.qUCCops[idx]:
            Pstring = term.paulis[0]
            coeff = term.coeffs[0]
            activeqbits = []                                                    # this contains all the qubits involved in the pauli string
            Xrotated = []
            Yrotated = []
            for n in range(instance.nqbits):                                    # pre-rotations to put everyone in Z basis
                if Pstring[n] == Pauli('I'):
                    pass
                elif Pstring[n] == Pauli('Z'):
                    activeqbits.append(n)
                elif Pstring[n] == Pauli('X'):
                    activeqbits.append(n)
                    Xrotated.append(n)
                    gate.h(n)
                elif Pstring[n] == Pauli('Y'):
                    activeqbits.append(n)
                    Yrotated.append(n)
                    gate.rx(-pi / 2, n)
                else:
                    print('Unknown Pauli')

            nactive_qbits = len(activeqbits)

            for j in range(nactive_qbits - 1):                                  #CNots ladder
                gate.cx(activeqbits[j], activeqbits[j + 1])

            gate.rz(coeff.real * angles[na * instance.nexc + idx], activeqbits[nactive_qbits - 1]) #Rz

            for j in range(nactive_qbits - 1, 0, -1):                           #CNots ladder
                gate.cx(activeqbits[j - 1], activeqbits[j])

            gate.h(Xrotated)                                                    #Undo the prerotations
            gate.rx(pi / 2, Yrotated)

    if instance.wordiness > 2:
        print('UCC GATE')
        print(gate.draw())
    UCCsd = gate.to_gate(label='UCC' + str(angles))
    return UCCsd
