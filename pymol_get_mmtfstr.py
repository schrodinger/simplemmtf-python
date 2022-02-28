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
    import simplemmtf

    state = int(state)

    try:
        # register PyMOL-specific spec extensions
        simplemmtf.levels['atom']['pymolReps'] = 0
        simplemmtf.levels['atom']['pymolColor'] = 0
        simplemmtf.encodingrules['pymolRepsList'] = (7, 0)
    except Exception as e:
        print(e)

    ss_map = {
        'H': 2,  # alpha helix
        'S': 3,  # extended
    }

    bonds = _self.get_bonds(selection, state)
    atoms = []

    def callback(state, segi, chain, resv, resi, resn, name, elem,
            x, y, z, reps, color, alt, formal_charge, b, q, ss):
        atoms.append({
            "modelIndex": state,
            "chainId": mmtfstr(segi),
            "chainName": mmtfstr(chain),
            "groupId": resv,
            "groupName": mmtfstr(resn),
            "atomName": mmtfstr(name),
            "element": mmtfstr(elem),
            "coords": (x, y, z),
            "altLoc": mmtfstr(alt),
            "formalCharge": formal_charge,
            "bFactor": b,
            "occupancy": q,
            "secStruct": ss_map.get(ss, -1),
            "insCode": mmtfstr(resi.lstrip("0123456789")),
            "pymolReps": reps,
            "pymolColor": color,
        })

    _self.iterate_state(
        state,
        selection,
        "callback(state, segi, chain, resv, resi, resn, name, elem, "
        "x, y, z, reps, color, alt, formal_charge, b, q, ss)",
        space={"callback": callback})

    d_out = from_atoms(atoms, bonds)

    return d_out.encode()


try:
    pymol.exporting.savefunctions['mmtf'] = get_mmtfstr
except AttributeError:
    print('Error: registering mmtf export requires PyMOL >= 1.8.6')
