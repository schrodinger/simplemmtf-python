# MMTF I/O Python Library

**simplemmtf** is a lightweight [MMTF](http://mmtf.rcsb.org/) encoding and decoding module written in Python. It was originally developed for PyMOL 1.8.4 to prototype MMTF load support.

This implementation is similar (but unrelated) to [mmtf-python](https://github.com/rcsb/mmtf-python).

## Dependencies

* Python 3.6+
* [msgpack-python](https://github.com/msgpack/msgpack-python) or [u-msgpack-python](https://github.com/vsergeev/u-msgpack-python)
* numpy (optional)

## Examples

* [mmtf2cif.py](mmtf2cif.py): Convert MMTF to mmCIF (without bonds)
* [pdb2mmtf.py](pdb2mmtf.py): Convert PDB to MMTF (without bonds)
* [pymol_get_mmtfstr.py](pymol_get_mmtfstr.py): Register MMTF export with PyMOL 1.8.6

## Example Code

```python
>>> import simplemmtf
>>> d = simplemmtf.fetch('1rx1')
>>> d.get('mmtfVersion')
'0.2.0'
```

```python
>>> next(d.atoms())
{u'atomName': u'N',
 u'chainId': u'A',
 u'chemCompType': u'L-PEPTIDE LINKING',
 u'coords': (12.284, 42.763, 10.037),
 u'element': u'N',
 u'formalCharge': 0,
 u'groupId': 1,
 u'groupName': u'MET',
 u'modelIndex': 0,
 u'singleLetterCode': u'M'}
```

## License

Copyright 2017 [Schrodinger, Inc.](http://www.schrodinger.com/)

Published under the **BSD-3-Clause** license.
