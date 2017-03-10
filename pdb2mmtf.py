#!/usr/bin/env python
'''
Example program which converts PDB to MMTF

Copyright (C) Schrodinger, Inc.
Author: Thomas Holder
License: BSD-3-Clause

'''


def pdb2mmtf(handle):
    '''
    @type handle: open file handle for reading
    @rtype: bytes
    '''

    from simplemmtf import mmtfstr, from_atoms

    def atom_gen():
        state = 0

        for line in handle:
            rec = line[:6]

            if rec == 'MODEL ':
                state += 1
            elif rec in ('ATOM  ', 'HETATM'):
                yield {
                    u'modelIndex': state,
                    u'atomId': int(line[6:11]),
                    u'atomName': mmtfstr(line[12:16].strip()),
                    u'altLoc': mmtfstr(line[16:17].rstrip()),
                    u'groupName': mmtfstr(line[17:20].strip()),
                    u'chainName': mmtfstr(line[21:22].rstrip()),
                    u'groupId': int(line[22:26]),
                    u'insCode': mmtfstr(line[26:27].rstrip()),
                    u'xCoord': float(line[30:38]),
                    u'yCoord': float(line[38:46]),
                    u'zCoord': float(line[46:54]),
                    u'bFactor': float(line[60:66]),
                    u'occupancy': float(line[54:60]),
                    u'chainId': mmtfstr(line[72:76].strip()),
                    u'element': mmtfstr(line[76:78].lstrip()),
                }

    d_out = from_atoms(atom_gen())

    return d_out.encode()


if __name__ == '__main__':
    import sys
    for filename in sys.argv[1:]:
        outfilename = filename + '.mmtf'
        with open(outfilename, 'wb') as handle:
            handle.write(pdb2mmtf(open(filename)))
        print(outfilename)
