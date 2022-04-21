from abc import ABC
from unicodedata import name
from qiskit_nature.drivers import Molecule
import pyscf


class build_molecule(ABC):

    def __init__(self, geometry, basis, charge, multiplicity, name, unit='Ang') -> None: #geometry but be  [('element', [x,y,z]),('element', [x,y,z])]
        self.geometry = geometry

        if isinstance(basis, build_basis):
            self.basis = basis.b_dict
            self.basis_name = str(" & ".join([basis.name_dict[atom[0]] for atom in self.geometry]))
        else:
            self.basis = basis
            self.basis_name = str(basis)

        self.name = name

        pyscfstring = ""
        qiskitlist = []
        for atom in self.geometry:
            pyscfstring = pyscfstring + atom[0] + ' ' + str(atom[1][0]) + ' ' + str(atom[1][1]) + ' ' + str(atom[1][2]) + ' ' + ';'
            qiskitlist.append([atom[0], [atom[1][0], atom[1][1], atom[1][2]]])

        mlc = pyscf.M(
            atom=pyscfstring,
            basis=self.basis,
            unit=unit,
            spin=multiplicity - 1,
            charge=charge,
        )

        self.ao_h1 = mlc.intor('int1e_nuc') + mlc.intor('int1e_kin')            # 1 body integrals
        self.ao_h2 = mlc.intor('int2e')                                         # 2 body integrals
        self.mo_h1 = pyscf.ao2mo.kernel(mlc, self.ao_h1)

        self.nuclear_energy = mlc.energy_nuc()

        self.norbitals = sum([mlc.atom_nshells(i) for i in range(mlc.natm)])
        self.nspinorbitals = self.norbitals * 2                                 #Spin degeneracy

        self.molecule = Molecule(geometry=qiskitlist, charge=charge, multiplicity=multiplicity)


class build_basis(ABC):

    def __init__(self, b_dict, name_dict) -> None:
        self.b_dict = b_dict
        self.name_dict = name_dict

    def __str__(self) -> str:
        return str(" & ".join(self.name_dict.values()))

    def __repr__(self) -> str:
        return self.b_dict


def build_HF(instance):
    return [i for i in range(instance.nα)] + [int(instance.nqbits / 2 + i) for i in range(instance.nβ)]
