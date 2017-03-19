'''
MMTF export with PyMOL

Copyright (C) Schrodinger, Inc.
Author: Thomas Holder
License: BSD-3-Clause
'''

import pymol


def get_mmtfstr(selection='all', state=1, _self=pymol.cmd):
    '''
DESCRIPTION

    Get an atom selection as MMTF format.
    '''
    from simplemmtf import mmtfstr, from_atoms

    state = int(state)

    if state == 0:
        states = range(1, _self.count_states(selection) + 1)
    else:
        states = [state]

    ss_map = {
        'H': 2,  # alpha helix
        'S': 3,  # extended
    }

    bonds = []
    atoms = []

    if True:
        numBonds = 0

        for state in states:
            m = _self.get_model(selection, state)

            for a in m.atom:
                atom = {
                    u'modelIndex': state,
                    u'chainId': mmtfstr(a.segi),
                    u'chainName': mmtfstr(a.chain),
                    u'groupId': a.resi_number,
                    u'groupName': mmtfstr(a.resn),
                    u'atomName': mmtfstr(a.name),
                    u'element': mmtfstr(a.symbol),
                    u'coords': a.coord,
                }

                if a.resi[-1:].isalpha():
                    atom[u'insCode'] = mmtfstr(a.resi[-1])

                if a.alt:
                    atom[u'altLoc'] = mmtfstr(a.alt)

                if a.formal_charge:
                    atom[u'formalCharge'] = int(a.formal_charge)

                if a.b:
                    atom[u'bFactor'] = a.b

                if a.q != 1.0:
                    atom[u'occupancy'] = a.q

                if a.ss:
                    atom[u'secStruct'] = ss_map.get(a.ss, -1)

                atoms.append(atom)

            for b in m.bond:
                bonds.append((b.index[0] + numBonds, b.index[1] + numBonds,
                              b.order))

            numBonds += len(m.atom)

    d_out = from_atoms(atoms, bonds)

    return d_out.encode()


try:
    pymol.exporting.savefunctions['mmtf'] = get_mmtfstr
except AttributeError:
    print('Error: registering mmtf export requires PyMOL >= 1.8.6')
