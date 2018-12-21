r'''
Experimental MMTF (Macromolecular Transmission Format) I/O library

Copyright (C) Schrodinger, Inc.
Author: Thomas Holder
License: BSD-3-Clause

Implements a pure standard Python library version and a numpy version
of the encoding and decoding routines. The numpy version is generally
faster, but performance may depend on how an application uses the data.
Select the preferred implementation with `use_numpy(bool)`.

Load MMTF file from working directory

    >>> d = simplemmtf.from_url('1rx1.mmtf.gz')

Access data

    >>> d.get(u'mmtfVersion')
    '0.2.0'
    >>> simplemmtf.noiter(d.get(u'xCoordList'))[:5]
    [12.284, 13.039, 14.463, 15.086, 12.961]

Traversal of atoms and bonds

    >>> bonds = []
    >>> for a in list(d.atoms(bonds))[:3]:
    ...     print(a['groupName'], a['groupId'], a['atomName'], a['coords'])
    ...
    MET 1 N (12.284, 42.763, 10.037)
    MET 1 CA (13.039, 42.249, 11.169)
    MET 1 C (14.463, 42.676, 11.106)
    >>> for b in bonds[:3]:
    ...     print(b)
    ...
    (8, 2, 1)
    (16, 10, 1)
    (22, 18, 1)

Compare return types of different APIs

    >>> simplemmtf.use_numpy(False)
    >>> type(d.get(u'xCoordList'))
    <class 'list'>
    >>> simplemmtf.use_numpy(True)
    >>> type(d.get(u'xCoordList'))
    <class 'numpy.ndarray'>

Encode to binary MessagePack format

    >>> d.encode()[:50]
    b'\xde\x00&\xabmmtfVersion\xa50.2.0\xacmmtfProducer\xd9FRCSB-PDB Gener'

Serialize atom table to MMTF:

    >>> atoms = [
    ...   {
    ...     u'atomName': u'N',
    ...     u'chainId': u'A',
    ...     u'coords': (12.284, 42.763, 10.037),
    ...     u'element': u'N',
    ...     u'groupName': u'MET',
    ...   },
    ...   {
    ...     u'atomName': u'CA',
    ...     u'chainId': u'A',
    ...     u'coords': (13.039, 42.249, 11.169),
    ...     u'element': u'C',
    ...     u'groupName': u'MET',
    ...   }
    ... ]
    >>> d = simplemmtf.from_atoms(atoms)

'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import itertools
import struct

try:
    import msgpack
    _KWARGS_UNPACK = {'encoding': 'utf-8'}
    _KWARGS_PACK = {'use_bin_type': True}
except ImportError:
    import umsgpack as msgpack
    _KWARGS_UNPACK = {}
    _KWARGS_PACK = {}

if sys.version_info[0] < 3:
    from urllib import urlopen
    izip = itertools.izip
    izip_longest = itertools.izip_longest
    _next_method_name = 'next'
    chr = unichr
else:
    # python3
    from urllib.request import urlopen
    izip = zip
    izip_longest = itertools.zip_longest
    buffer = lambda s, i=0: memoryview(s)[i:]
    xrange = range
    _next_method_name = '__next__'
    unicode = str


def mmtfstr(s):
    '''Cast `s` to MMTF compatible string'''
    if isinstance(s, bytes):
        return s.decode('ascii')
    return unicode(s)


class _apiproxy:
    def __call__(self, obj):
        setattr(self, obj.__name__, obj)


apinumpy = _apiproxy()
apigeneric = _apiproxy()
apigeneric.simpleiter = iter


# should be replaced with a more efficient numpy-array aware iterator
@apinumpy
def simpleiter(iterable):
    if isinstance(iterable, numpy.ndarray):
        # Iterating over a numpy array is unreasonably slow. Iterating
        # over its list copy is much faster!
        return iter(iterable.tolist())
    return iter(iterable)


def asarray(arr, dtype='i'):
    if hasattr(arr, '__len__'):
        return numpy.asarray(arr, dtype)
    return numpy.fromiter(arr, dtype)


def aslist(iterable):
    if isinstance(iterable, list):
        return iterable
    if hasattr(iterable, 'tolist'):
        return iterable.tolist()
    return list(iterable)


def noiter(iterable):
    if not hasattr(iterable, '__len__') and \
            hasattr(iterable, _next_method_name):
        return list(iterable)
    return iterable


MMTF_ENDIAN = '>'  # big-endian

########### ENCODINGS ###########################################


class RunLength:
    @staticmethod
    def encode(iterable):
        in_iter = api.simpleiter(iterable)
        try:
            curr = next(in_iter)
        except StopIteration:
            return
        counter = 1
        for item in in_iter:
            if item == curr:
                counter += 1
            else:
                yield curr
                yield counter
                curr = item
                counter = 1
        yield curr
        yield counter

    @staticmethod
    def decode(iterable):
        in_iter = api.simpleiter(iterable)
        out = []
        extend = out.extend
        for item in in_iter:
            extend([item] * next(in_iter))
        return out


@apigeneric
class Delta:
    @staticmethod
    def encode(iterable):
        prev = 0
        for i in api.simpleiter(iterable):
            yield i - prev
            prev = i

    @staticmethod
    def decode(iterable):
        running = 0
        for i in api.simpleiter(iterable):
            running += i
            yield running


@apinumpy
class Delta:
    @staticmethod
    def encode(iterable):
        arr = asarray(iterable, 'i4')
        out = arr.copy()
        out[1:] -= arr[:-1]
        return out

    @staticmethod
    def decode(iterable):
        return asarray(iterable).cumsum(dtype='i4')


class RecursiveIndex:
    def __init__(self, min, max):
        self.limits = (min, max)

    def encode(self, iterable):
        min, max = self.limits
        for curr in api.simpleiter(iterable):
            while curr >= max:
                yield max
                curr -= max
            while curr <= min:
                yield min
                curr -= min
            yield curr

    def decode(self, iterable):
        min, max = self.limits
        decoded_val = 0
        for item in api.simpleiter(iterable):
            decoded_val += item
            if item != max and item != min:
                yield decoded_val
                decoded_val = 0


@apigeneric
class IntegerFloats:
    def __init__(self, factor):
        self.factor = factor

    def encode(self, in_floats):
        factor = self.factor
        return [int(f * factor) for f in api.simpleiter(in_floats)]

    def decode(self, in_ints):
        factor = float(self.factor)
        return [i / factor for i in api.simpleiter(in_ints)]


@apinumpy
class IntegerFloats:
    def __init__(self, factor):
        self.factor = factor

    def encode(self, in_floats):
        return (asarray(in_floats, 'f4') * self.factor).astype('i4')

    def decode(self, in_ints):
        return asarray(in_ints, 'f4') / self.factor


class IntegerChars:
    @staticmethod
    def encode(in_chars):
        return [(ord(x) if x else 0) for x in in_chars]

    @staticmethod
    def decode(in_ints):
        return [(chr(x) if x else u'') for x in api.simpleiter(in_ints)]


######## BUFFERS ################################################


@apigeneric
class NumbersBuffer:
    typemap = {}

    def __init__(self, basetype='i', dectype=''):
        if not self.typemap:
            from array import array
            self.__class__.array = staticmethod(array)
            for code in 'bhil':
                self.typemap['i' + str(array(code).itemsize)] = code
            for code in 'fd':
                self.typemap['f' + str(array(code).itemsize)] = code
            self.typemap['f'] = self.typemap['f4']
            self.typemap['i'] = self.typemap['i4']

        self.enctype = self.typemap[basetype]

    def decode(self, in_bytes):
        a = self.array(self.enctype)
        a.fromstring(in_bytes)
        if sys.byteorder != 'big':
            a.byteswap()
        return a

    def encode(self, in_ints):
        a = self.array(self.enctype, in_ints)
        if sys.byteorder != 'big':
            a.byteswap()
        return a.tostring()


@apinumpy
class NumbersBuffer:
    def __init__(self, basetype='i', dectype=''):
        self.enctype = numpy.dtype(MMTF_ENDIAN + basetype)
        self.dectype = numpy.dtype(dectype or basetype)

    def decode(self, in_bytes):
        return numpy.frombuffer(in_bytes, self.enctype).astype(self.dectype)

    def encode(self, in_ints):
        return asarray(in_ints, self.enctype).tostring()


@apigeneric
class StringsBuffer:
    def __init__(self, nbytes, encoding='utf-8'):
        self.nbytes = nbytes
        self.encoding = encoding

    def decode(self, in_bytes):
        n = self.nbytes
        e = self.encoding
        return [
            bytes(in_bytes[i:i + n]).rstrip(b'\0').decode(e)
            for i in xrange(0, len(in_bytes), n)
        ]

    def encode(self, strings):
        n = self.nbytes
        e = self.encoding
        b = bytearray(len(strings) * n)
        i = 0
        for s in strings:
            s = s.encode(e)
            L = len(s)
            if L > n:
                s = s[:n]
                L = n
            b[i:i + L] = s
            i += n
        return bytes(b)


@apinumpy
class StringsBuffer:
    def __init__(self, nbytes, encoding='utf-8'):
        self.enctype = numpy.dtype('S' + str(nbytes))
        self.encoding = encoding

    def decode(self, in_bytes):
        bstrings = numpy.frombuffer(in_bytes, self.enctype)
        return [b.decode(self.encoding) for b in bstrings]

    def encode(self, strings):
        s_it = (s.encode(self.encoding) for s in strings)
        bstrings = numpy.fromiter(s_it, self.enctype, len(strings))
        return bstrings.tostring()


########## NUMPY API ############################################


def use_numpy(value=True):
    '''
    Use numpy for list encoding and decoding. If `value` is False, then
    use built-in types and generic routines.

    Inital setup depends on USE_NUMPY environment variable and defaults
    to False (generic implementation).
    '''
    global numpy, api

    if value:
        try:
            import numpy
        except ImportError:
            print('numpy not available')
        else:
            api = apinumpy
            return

    api = apigeneric


use_numpy(os.getenv('USE_NUMPY', 0))

########## STRATEGIES ###########################################


def _PackedIntBufStrategy(nbytes=1, dectype='i4'):
    m = 1 << (nbytes * 8 - 1)
    return [
        api.NumbersBuffer('i' + str(nbytes), dectype),
        RecursiveIndex(-m, m - 1),
    ]

strategies = {
    1: lambda _: [api.NumbersBuffer('f4')],
    2: lambda _: [api.NumbersBuffer('i1')],
    3: lambda _: [api.NumbersBuffer('i2')],
    4: lambda _: [api.NumbersBuffer('i4')],
    5: lambda length: [api.StringsBuffer(length)],
    6: lambda _: [api.NumbersBuffer('i4'), RunLength, IntegerChars],
    7: lambda _: [api.NumbersBuffer('i4'), RunLength],
    8: lambda _: [api.NumbersBuffer('i4'), RunLength, api.Delta],
    9: lambda factor: [api.NumbersBuffer('i4'), RunLength, api.IntegerFloats(factor)],
   10: lambda factor: _PackedIntBufStrategy(2) + [api.Delta, api.IntegerFloats(factor)],
   11: lambda factor: [api.NumbersBuffer('i2'), api.IntegerFloats(factor)],
   12: lambda factor: _PackedIntBufStrategy(2) + [api.IntegerFloats(factor)],
   13: lambda factor: _PackedIntBufStrategy(1) + [api.IntegerFloats(factor)],
   14: lambda _: _PackedIntBufStrategy(2),
   15: lambda _: _PackedIntBufStrategy(1),
}

# optional parameters format (defaults to 'i' -> one int32 argument)
strategyparamsfmt = {}

########## MEDIUM LEVEL ARRAY ENCODE/DECODE API #################


def check_encodable(arr, codec, param=0):
    '''Check if an array is encodable with the requested strategy.

    Example of type 5 "fixed-length string array" with string length 4:

        >>> check_encodable(["ABCD"], 5, 4)
        True
        >>> check_encodable(["ABCDE"], 5, 4)
        False
    '''
    if codec == 5:
        return all(len(s) <= param for s in arr)

    return True


def encode_array(arr, codec, param=0):
    strategy = strategies[codec](param)

    buf = struct.pack(MMTF_ENDIAN + 'iii', codec, len(arr), param)

    for handler in reversed(strategy):
        arr = handler.encode(arr)

    buf += arr
    return buf


def decode_array(value):
    codec, length = struct.unpack(MMTF_ENDIAN + 'ii', value[:8])

    fmt = strategyparamsfmt.get(codec, 'i')
    params = struct.unpack(MMTF_ENDIAN + fmt, value[8:12])
    strategy = strategies[codec](*params)

    value = buffer(value, 12)
    for handler in strategy:
        value = handler.decode(value)

    return value


def _get_array_length(value):
    if isinstance(value, bytes):
        return struct.unpack(MMTF_ENDIAN + 'i', value[4:8])[0]
    return len(value)


############### SPEC ############################################

MMTF_SPEC_VERSION = (1, 0)

encodingrules = {
    u"altLocList": (6, 0),
    u"atomIdList": (8, 0),
    u"bFactorList": (10, 100),
    u"bondAtomList": (4, 0),
    u"bondOrderList": (2, 0),
    u"chainIdList": (5, 4),
    u"chainNameList": (5, 4),
    u"groupIdList": (8, 0),
    u"groupTypeList": (4, 0),
    u"insCodeList": (6, 0),
    u"occupancyList": (9, 100),
    u"secStructList": (2, 0),
    u"sequenceIndexList": (8, 0),
    u"xCoordList": (10, 1000),
    u"yCoordList": (10, 1000),
    u"zCoordList": (10, 1000),
}

requiredfields = [
    u"mmtfVersion",
    u"mmtfProducer",
    u"numBonds",
    u"numAtoms",
    u"numGroups",
    u"numChains",
    u"numModels",
    u"groupList",
    u"xCoordList",
    u"yCoordList",
    u"zCoordList",
    u"groupIdList",
    u"groupTypeList",
    u"chainIdList",
    u"groupsPerChain",
    u"chainsPerModel",
]

levels = {
    # top-level {key}List len=numChains
    u'chain': {
        u'chainName': u'',
        u'chainId': u'',
    },
    # top-level {key}List len=numGroups
    u'group': {
        u'sequenceIndex': -1,
        u'groupId': -1,
        u'insCode': u'',
        u'secStruct': -1,
    },
    # groupType-level {key}
    u'grouptype': {
        u'groupName': u'',
        u'singleLetterCode': u'?',
        u'chemCompType': u'',
    },
    # groupType-level {key}List
    u'grouptypeatom': {
        u'formalCharge': 0,
        u'atomName': u'',
        u'element': u'',
    },
    # top-level {key}List len=numAtoms
    u'atom': {
        u'bFactor': 0.0,
        u'occupancy': 1.0,
        u'altLoc': u'',
        u'atomId': -1,
    },
}


def assert_consistency(mmtfdict, acceptempty=False):
    '''Raise KeyError if any non-optional field is missing.
    Raise IndexError if any array has the wrong length.

    @param mmtfdict: container to check
    @type mmtfdict: MmtfDict or dict
    @param acceptempty: accept length zero for non-required lists
    @type acceptempty: bool
    '''

    if isinstance(mmtfdict, MmtfDict):
        d_data = mmtfdict._data
    else:
        d_data = mmtfdict

    # check for missing fields
    for key in requiredfields:
        if key in (u'mmtfVersion', u'mmtfProducer'):
            # fields set on `encode()`
            continue

        if key not in d_data:
            raise KeyError(key)

    # check for correct list lengths
    for numkey, keys in [
        (u'numAtoms', [
            u'xCoordList',
            u'yCoordList',
            u'zCoordList',
            u'bFactorList',
            u'occupancyList',
            u'altLocList',
            u'atomIdList',
        ]),
        (u'numGroups', [
            u'sequenceIndexList',
            u'groupIdList',
            u'insCodeList',
            u'secStructList',
            u'groupTypeList',
        ]),
        (u'numChains', [
            u'chainIdList',
            u'chainNameList',
            u'groupsPerChain',
        ]),
        (u'numModels', [u'chainsPerModel']),
    ]:
        num = mmtfdict.get(numkey)
        for key in keys:
            if key not in d_data:
                continue
            length = _get_array_length(d_data[key])
            if acceptempty and length == 0 and key not in requiredfields:
                continue
            if length != num:
                raise IndexError('length mismatch %s=%d len(%s)=%d' %
                                 (numkey, num, key, length))

    # check pointers in groupTypeList
    numgrouptypes = _get_array_length(d_data[u'groupList'])
    maxgrouptype = max(mmtfdict.get(u'groupTypeList'))
    if maxgrouptype >= numgrouptypes:
        raise IndexError('max(groupTypeList)=%d len(groupList)=%d' %
                         (maxgrouptype, numgrouptypes))

    # check toplevel bonds
    bondAtomList = noiter(mmtfdict.get(u'bondAtomList', ()))
    numatomindices = len(bondAtomList)
    if numatomindices:
        if max(bondAtomList) >= mmtfdict.get(u'numAtoms'):
            raise IndexError('max(bondAtomList) >= numAtoms')
        if u'bondOrderList' in d_data:
            numbondorders = _get_array_length(d_data[u'bondOrderList'])
            if numatomindices != numbondorders * 2 and not (
                    acceptempty and numbondorders == 0):
                raise IndexError('len(bondAtomList)=%d len(bondOrderList)=%d' %
                                 (numatomindices, numbondorders))


############### HIGH LEVEL API ##################################


class MmtfDict:
    '''
    Toplevel MMTF container

    Data is stored encoded and is decoded when accessed with `get(...)`.

    Get a fully decoded copy with `to_dict()`.
    '''

    def __init__(self, data=None):
        '''
        @param data: file contents or open file handle, optionally gzipped
        @type data: bytes, stream or dict
        '''
        if data is None:
            self._data = {}
            return

        if isinstance(data, dict):
            self._data = {}
            for key, value in data.items():
                # discard non-required lists of length zero
                if key not in requiredfields and hasattr(
                        value, '__len__') and len(value) == 0:
                    continue

                self.set(key, value)
            return

        if isinstance(data, bytes):
            if data[:2] != b'\x1f\x8b':  # gzip magic number
                self._set_data(msgpack.unpackb(data, **_KWARGS_UNPACK))
                return

            import io, gzip
            data = gzip.GzipFile(fileobj=io.BytesIO(data))

        self._set_data(msgpack.unpack(data, **_KWARGS_UNPACK))

    def _set_data(self, data):
        v = mmtfstr(data.get(u'mmtfVersion', ''))
        v_major = int(v.split(u'.')[0] or 0)

        if v_major > MMTF_SPEC_VERSION[0]:
            raise NotImplementedError('Unsupported version: ' + v)

        self._data = data

    @classmethod
    def from_url(cls, url):
        '''Load an MMTF file from disk or URL'''
        handle = open(url, 'rb') if os.path.isfile(url) else urlopen(url)
        return cls(handle.read())

    def __setitem__(self, key, value):
        '''Set a field, using the default encoding according to the spec'''
        self.set(key, value)

    def __delitem__(self, key):
        '''Remove a field'''
        try:
            del self._data[key]
        except KeyError:
            del self._data[mmtfstr(key)]

    def __contains__(self, key):
        '''True if the field exists in this container'''
        return key in self._data

    def keys(self):
        '''Iterator over toplevel field names'''
        return iter(self._data)

    def to_dict(self):
        '''Fully decoded copy of this instance'''
        return dict((k, noiter(self.get(k))) for k in self._data)

    def get(self, key, default=None):
        '''Look up a toplevel field by name, and decode on the fly.
        The return value of encoded lists is iterable, the exact return
        type is not guaranteed (can be iterator, list, or numpy array).
        Use `simplemmtf.noiter` to ensure accessing a sequence.
        '''
        try:
            value = self._data[key]
        except KeyError:
            return default

        if not (key.endswith(u'List') and isinstance(value, bytes)):
            return value

        return decode_array(value)

    def get_iter(self, key, default=()):
        '''Like `get(...)` but always return an iterator'''
        return api.simpleiter(self.get(key, default))

    def get_noiter(self, key, default=()):
        '''Like `get(...)` but never return an iterator'''
        return noiter(self.get(key, default))

    def get_table_iter(self, keys, defaults=None):
        '''Treat a set of fields like columns of a table and return an
        iterator over the table rows. A field may be empty or missing,
        in which case the column will be filled with `None` or with the
        value specified by `defaults`.

        @param keys: sequence of field names
        @param defaults: row with default values to fill up short or
        missing columns.
        '''
        if defaults is None:
            return izip_longest(*[self.get_iter(k) for k in keys])
        return izip(*[
            self.get_iter(k, itertools.repeat(d))
            for (k, d) in zip(keys, defaults)
        ])

    def set(self, key, value, codec=-1, param=0):
        '''Set a toplevel field and store in decoded form.
        
        If `codec` is 0, then the data is not encoded. If `codec` is -1,
        then look up the default encoding for `key` in the MMTF spec.
        '''
        if codec == -1:
            codec, param = encodingrules.get(key, (0, 0))

        if codec != 0 and check_encodable(value, codec, param):
            value = encode_array(value, codec, param)

        self._data[mmtfstr(key)] = value

    def encode(self):
        '''@rtype: bytes'''
        self.set(u'mmtfVersion', u'%d.%d' % MMTF_SPEC_VERSION)

        if not self.get(u'mmtfProducer'):
            self.set(u'mmtfProducer', u'simplemmtf')

        return msgpack.packb(self._data, **_KWARGS_PACK)

    @classmethod
    def from_atoms(cls, atom_iter, bond_iter=None):
        '''
        Serialize atom and bond tables to MMTF format.

        @param atom_iter: iterator over type "dict" atoms with MMTF vocabulary
        items, e.g. {u'chainName': u'A', u'groupId': 1, ...}

        @param bond_iter: optional iterator over (index1, index2, order) tuples

        @rtype: MmtfDict
        '''
        raw = _from_atoms(atom_iter, bond_iter)
        return cls(raw)

    def atoms(self, bonds=None):
        '''Iterator over atoms.

        @param bonds: Optional output variable for bonds as (index1, index2, order)
        @type bonds: list

        @rtype: generator
        '''
        return _atoms_iter(self, bonds)

    def groups(self):
        '''Iterator over groups.'''
        return _atoms_iter(self, _just_groups=True)


from_atoms = MmtfDict.from_atoms
decode = lambda data: MmtfDict(data)
from_url = MmtfDict.from_url


def fetch(code):
    '''Download from RCSB web service'''
    return from_url("http://mmtf.rcsb.org/v1.0/full/" + code + ".mmtf.gz")


############# TRAVERSAL ######################################


def _atoms_iter(data, bonds=None, _just_groups=False):
    from itertools import islice

    def add_bond(i1, i2, order, offset=0):
        bonds.append((i1 + offset, i2 + offset, order))

    if bonds is not None:
        bondAtomList_iter = data.get_iter(u'bondAtomList')

        for order in data.get_iter(u'bondOrderList'):
            i1 = next(bondAtomList_iter)
            i2 = next(bondAtomList_iter)
            add_bond(i1, i2, order)

    coord_iter = data.get_table_iter([
        u'xCoordList',
        u'yCoordList',
        u'zCoordList',
    ])

    leveliters = lambda level: [
        (key, data.get_iter(key + u'List'))
        for (key, default) in levels[level].items()
        if key + u'List' in data
    ]

    chain_list_iters = leveliters(u'chain')
    group_iters = leveliters(u'group')
    atom_iters = leveliters(u'atom')

    n_groups_iter = data.get_iter(u'groupsPerChain')
    groupType_iter = data.get_iter(u'groupTypeList')

    groupList = data.get(u'groupList')

    atom = {u'modelIndex': -1}
    offset = 0

    for n_chains in data.get_iter(u'chainsPerModel'):
        atom[u'modelIndex'] += 1

        for n_groups in islice(n_groups_iter, n_chains):

            for key, it in chain_list_iters:
                atom[key] = next(it)

            for groupType in islice(groupType_iter, n_groups):

                for key, it in group_iters:
                    atom[key] = next(it)

                group = groupList[groupType]

                for (key, default) in levels[u'grouptype'].items():
                    atom[key] = group.get(key, default)

                if _just_groups:
                    yield atom.copy()
                    continue

                if bonds is not None:
                    group_bond_iter = izip(
                        group[u'bondAtomList'][0::2],
                        group[u'bondAtomList'][1::2],
                        group[u'bondOrderList'], )

                    for (i1, i2, order) in group_bond_iter:
                        add_bond(i1, i2, order, offset)

                group_atom_iters = [
                    (key, iter(group[key + u'List']))
                    for (key, default) in levels[u'grouptypeatom'].items()
                ]

                for _ in group[u'atomNameList']:
                    offset += 1

                    for key, it in group_atom_iters:
                        atom[key] = next(it)

                    for key, it in atom_iters:
                        atom[key] = next(it)

                    # use "coords" instead of xCoord, yCoord, zCoord
                    atom[u'coords'] = next(coord_iter)

                    yield atom.copy()


############### TABLE SERIALIZATION ######################


def dict_as_tuple(d):
    def gen():
        for (k, v) in sorted(d.items()):
            if isinstance(v, list):
                v = tuple(v)
            elif isinstance(v, dict):
                v = dict_as_tuple(v)
            yield (k, v)

    return tuple(gen())


def dict_subset_equal(d1, d2, keys):
    for key in keys:
        if d1.get(key) != d2.get(key):
            return False
    return True


class optionallist(list):
    '''List type with default value and overloaded `append` method which
    will not append anything until the first non-default non-None value
    is appended.

        >>> L = optionallist(5)
        >>> L.append(5)
        >>> L.append(None)
        >>> L
        []
        >>> L.append(3)
        >>> L.append(None)
        >>> L
        [5, 5, 3, 5]

    '''

    def __init__(self, default, size=0):
        self._default = default
        self._size = size

    def append(self, value):
        if self:
            list.append(self, self._default if value is None else value)
        elif value is not None and value != self._default:
            self[:] = [self._default] * self._size
            list.append(self, value)
        self._size += 1


def _from_atoms(atom_iter, bond_iter=None):
    '''@rtype: dict
    '''
    import collections

    bonds = collections.defaultdict(list)
    numBonds = 0

    for bond in (bond_iter or ()):
        numBonds += 1
        order = bond[2] if len(bond) > 2 else 1
        if bond[0] < bond[1]:
            bonds[bond[1]].append((bond[0], order))
        else:
            bonds[bond[0]].append((bond[1], order))

    raw = {
        # numAtoms
        u'xCoordList': [],
        u'yCoordList': [],
        u'zCoordList': [],

        # numGroups
        u'groupIdList': [],  # auth_seq_id
        u'groupTypeList': [],  # groupList indices

        # numChains
        u'chainIdList': [],
        u'groupsPerChain': [],

        # numModels
        u'chainsPerModel': [],

        # num group types
        u'groupList': [],

        # bonds
        u'numBonds': numBonds,
        u'bondAtomList': [],
        u'bondOrderList': optionallist(1),
    }

    for level in (u'chain', u'group', u'atom'):
        for key, default in levels[level].items():
            key += u'List'
            if key in raw:
                continue
            raw[key] = optionallist(default)

    groupHash = {}
    residue = {}
    residue_first_atom = 0
    prev_atom = {u'modelIndex': object()}  # unique value

    def handleGroupType(group):
        if not group:
            return

        hashed = dict_as_tuple(group)

        try:
            groupType = groupHash[hashed]
        except KeyError:
            groupType = len(raw[u'groupList'])
            groupHash[hashed] = groupType
            raw[u'groupList'].append(group)

        raw[u'groupTypeList'].append(groupType)

    for atomIndex, atom in enumerate(atom_iter):
        try:
            x, y, z = atom[u'coords']
        except KeyError:
            x = atom[u'xCoord']
            y = atom[u'yCoord']
            z = atom[u'zCoord']

        # numAtoms required
        raw[u'xCoordList'].append(x)
        raw[u'yCoordList'].append(y)
        raw[u'zCoordList'].append(z)

        # numAtoms optional
        for key, default in levels[u'atom'].items():
            raw[key + u'List'].append(atom.get(key, default))

        is_same_model = dict_subset_equal(prev_atom, atom, [u'modelIndex'])

        if not is_same_model:
            raw[u'chainsPerModel'].append(0)
            is_same_chain = False
        else:
            is_same_chain = dict_subset_equal(prev_atom, atom,
                                              levels[u'chain'])

        if not is_same_chain:
            raw[u'chainsPerModel'][-1] += 1
            raw[u'groupsPerChain'].append(0)  # increment with every group
            for key, default in levels[u'chain'].items():
                raw[key + u'List'].append(atom.get(key, default))
            is_same_residue = False
        else:
            is_same_residue = dict_subset_equal(prev_atom, atom,
                                                levels[u'group'])

        if not is_same_residue:
            raw[u'groupsPerChain'][-1] += 1

            handleGroupType(residue)

            for key, default in levels[u'group'].items():
                raw[key + u'List'].append(atom.get(key, default))

            residue = {
                u'bondAtomList': [],
                u'bondOrderList': [],
            }

            residue.update((key + u'List', [])
                           for key in levels['grouptypeatom'])

            for key, default in levels[u'grouptype'].items():
                residue[key] = atom.get(key, default)

            residue_first_atom = atomIndex

        for key, default in levels[u'grouptypeatom'].items():
            residue[key + u'List'].append(atom.get(key, default))

        for bond in bonds.get(atomIndex, ()):
            if bond[0] < residue_first_atom:
                raw[u'bondAtomList'].append(atomIndex)
                raw[u'bondAtomList'].append(bond[0])
                raw[u'bondOrderList'].append(bond[1])
            else:
                residue[u'bondAtomList'].append(atomIndex - residue_first_atom)
                residue[u'bondAtomList'].append(bond[0] - residue_first_atom)
                residue[u'bondOrderList'].append(bond[1])

        prev_atom = atom

    handleGroupType(residue)

    raw[u'numAtoms'] = len(raw[u'xCoordList'])
    raw[u'numGroups'] = len(raw[u'groupIdList'])
    raw[u'numChains'] = len(raw[u'chainIdList'])
    raw[u'numModels'] = len(raw[u'chainsPerModel'])

    return raw


############### SIMPLE FILE READ TEST ####################

if __name__ == '__main__':
    for fn in sys.argv[1:]:
        print('loading', fn)
        d = from_url(fn)
        d.to_dict()
        for a in list(d.atoms())[:3]:
            print(a[u'groupName'], a[u'groupId'], a[u'atomName'], a[u'coords'])
        # re-serialization
        bonds = []
        atoms = list(d.atoms(bonds))
        d_out = from_atoms(atoms, bonds)
        del atoms, bonds
        assert d.get(u'numAtoms') == d_out.get(u'numAtoms')
        assert d.get(u'numBonds') == d_out.get(u'numBonds')
        assert d.get(u'numModels') == d_out.get(u'numModels')
        assert_consistency(d_out)
