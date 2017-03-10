from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import simplemmtf
from simplemmtf import noiter, requiredfields


def write(dat, filename):
    print('Writing to ' + filename)
    with open(filename, 'wb') as handle:
        handle.write(dat.encode())


def nowrite(dat, _):
    dat.encode()


def test_file(infilename):
    dat = simplemmtf.from_url(infilename)

    assert dat.get('mmtfVersion') is not None

    write(dat, infilename + '-repacked.mmtf')

    # recode
    for key in dat.keys():
        dat[key] = noiter(dat.get(key))

    write(dat, infilename + '-recoded.mmtf')

    # remove non-required fields
    dat.set(u'numBonds', dat.get('numBonds') - len(dat.get('bondOrderList', ())))
    for key in list(dat.keys()):
        if key not in requiredfields:
            del dat[key]

    write(dat, infilename + '-onlyrequired.mmtf')


def make_empty_files():
    # make empty MMTF file with required fields
    dat = simplemmtf.MmtfDict()

    dat.set(u"mmtfProducer", u"Thomas Holder")
    dat.set(u"numBonds", 0)
    dat.set(u"numAtoms", 0)
    dat.set(u"numGroups", 0)
    dat.set(u"numChains", 0)
    dat.set(u"numModels", 0)
    dat.set(u"groupList", [])
    dat.set(u"xCoordList", [])
    dat.set(u"yCoordList", [])
    dat.set(u"zCoordList", [])
    dat.set(u"groupIdList", [])
    dat.set(u"groupTypeList", [])
    dat.set(u"chainIdList", [])
    dat.set(u"groupsPerChain", [])
    dat.set(u"chainsPerModel", [])

    write(dat, 'empty-all0.mmtf')

    # empty with 1 model
    dat.set(u"numModels", 1)
    dat.set(u"chainsPerModel", [0])

    write(dat, 'empty-numModels1.mmtf')

    # empty with 1 chain
    dat.set(u"numChains", 1)
    dat.set(u"chainIdList", [u"A"])
    dat.set(u"groupsPerChain", [0])
    dat.set(u"chainsPerModel", [1])

    write(dat, 'empty-numChains1.mmtf')


def test_encode_recode():
    # test encode/decode
    dat = simplemmtf.MmtfDict()

    import numpy

    for (key, arr) in [
        (u"altLocList", ['', '', 'A', 'A', 'B', '']),
        (u"atomIdList", list(range(1000, 1010))),
        (u"bFactorList", numpy.random.random(10) * 100),
        (u"bondAtomList", numpy.random.randint(1, 1000, 10)),
        (u"bondOrderList", numpy.random.randint(1, 5, 10)),
        (u"chainIdList", ['A', 'A', 'A', 'BCDE', 'BCDE', 'foo', 'bar']),
        (u"occupancyList", [1.0] * 5 + numpy.random.random(5).tolist()),
        (u"secStructList", list(range(-1, 8)) + [2] * 5),
        (u"xCoordList", [1.23, 4.56, 7.89, 1000023.4, 1000045.6, 0.5, 0.0]),
        (u"unknownUserKey", list(range(100))),
    ]:
        dat.set(key, arr)
        arr_test = noiter(dat.get(key))

        if isinstance(arr[0], str):
            testval = (arr == arr_test)
        else:
            testval = numpy.allclose(arr, arr_test, 1e-6, 1e-2)

        if not testval:
            print(arr)
            print(arr_test)
            raise UserWarning(key)


def test():
    for fn in sys.argv[1:]:
        test_file(fn)

    test_encode_recode()


if __name__ == '__main__':
    import timeit
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument('--no-empty', action="store_true")
    ap.add_argument('--no-write', '-r', action="store_true")
    ap.add_argument('--timeit-number', '-n', type=int, default=3)

    options, sys.argv[1:] = ap.parse_known_args()

    if options.no_write:
        write = nowrite

    for np_switch in (True, False):
        simplemmtf.use_numpy(np_switch)
        t = timeit.timeit(test, number=options.timeit_number)
        print('numpy', np_switch, t)

    if not options.no_empty:
        make_empty_files()
