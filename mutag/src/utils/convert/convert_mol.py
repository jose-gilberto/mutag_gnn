from rdkit import Chem as Chem

atoms_type = {
    0: 'C',
    1: 'N',
    2: 'O',
    3: 'F',
    4: 'I',
    5: 'Cl',
    6: 'Br'
}

bonds_type = {
    0: Chem.rdchem.BondType.AROMATIC,
    1: Chem.rdchem.BondType.SINGLE,
    2: Chem.rdchem.BondType.DOUBLE,
    3: Chem.rdchem.BondType.TRIPLE
}


def graph_to_mol(atoms_list, edge_index, edge_attr):
    mol = Chem.RWMol()
    node_to_idx = {}

    atoms_list = [atoms_type[atom.index(1)] for atom in atoms_list]
    for i in range(len(atoms_list)):
        a = Chem.Atom(atoms_list[i])
        mol_idx = mol.AddAtom(a)
        node_to_idx[i] = mol_idx

    bond_list = list(zip(edge_index[0], edge_index[1]))
    bond_list = list({tuple(sorted(item)) for item in bond_list})
    for i, t in enumerate(bond_list):
        btype = edge_attr[i].index(1)
        btype = bonds_type[btype]
        mol.AddBond(t[0], t[1], btype)

    mol = mol.GetMol()
    return mol
