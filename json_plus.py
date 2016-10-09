#!/usr/bin/env python

# Copyright 2014 Camiolog Inc.
# Authors: Luca de Alfaro and Massimo Di Pierro

import base64
import datetime
import importlib
import json
import numbers
import numpy
import unittest
import collections


fallback = {}
remapper = {}


class Storage(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


def smartcmp(a,b,
             types=(int, long, basestring, float, bool, tuple)):
    is_a_primitive = isinstance(a[1],types)
    is_b_primitive = isinstance(b[1],types)
    if is_a_primitive and not is_b_primitive:
        return -1
    elif not is_a_primitive and is_b_primitive:
        return +1
    else:
        return cmp(a[0],b[0])


class Serializable(object):

    # We mimick a dict.
    def __getitem__(self, key):
        return getattr(self, key)
    def __setitem__(self, key, value):
        setattr(self, key, value)
    def __delitem__(self, key):
        del self.__dict__[key]
    def keys(self):
        return self.__dict__.keys()
    def items(self):
        return self.__dict__.items()
    def values(self):
        return self.__dict__.values()
    def update(self, d):
        self.__dict__.update(d)
    def __len__(self):
        return len(self.__dict__)
    def __contains__(self, item):
        return item in self.__dict__
    def iteritems(self):
        return iter(self.__dict__.items())
    def __repr__(self):
        return repr(self.__dict__)

    def get(self, k, d=None):
        try:
            return getattr(self, k)
        except AttributeError:
            return d

    def __eq__(self, other):
        return hasattr(other, '__dict__') and self.__dict__ == other.__dict__

    def to_json(self, pack_ndarray=True, tolerant=True, indent=2):
        return Serializable.dumps(self, pack_ndarray=pack_ndarray, tolerant=tolerant, indent=indent)

    @staticmethod
    def dumps(obj, pack_ndarray=True, tolerant=True, indent=2):
        def custom(o):
            if isinstance(o, Serializable):
                module = o.__class__.__module__.split('campil.')[-1]
                # make sure keys are sorted
                d = collections.OrderedDict()
                d['meta_class'] = '%s.%s' % (module, o.__class__.__name__)
                d.update(sorted((item for item in o.__dict__.iteritems()
                                 if not item[0].startswith('_')), smartcmp))
                return d
            elif isinstance(o, datetime.datetime):
                d = {'meta_class': 'datetime.datetime',
                     'date': o.isoformat()}
                return d
            elif isinstance(o, set):
                d = {'meta_class': 'set',
                     'set': list(o)}
                return d
            elif isinstance(o, file):
                return '<file %r>' % o.name

            elif pack_ndarray and isinstance(o, numpy.matrix):
                # This catches both numpy arrays, and CamArray.
                d = {'meta_class': 'numpy.matrix',
                     'dtype': str(o.dtype),
                     'shape': o.shape,
                     'data': base64.b64encode(o.tostring())}
                return d

            elif pack_ndarray and isinstance(o, numpy.ndarray):
                # This catches both numpy arrays, and CamArray.
                d = {'meta_class': 'numpy.ndarray',
                     'dtype': str(o.dtype),
                     'shape': o.shape,
                     'data': base64.b64encode(o.tostring())}
                return d

            # We try to preserve numpy numbers.
            elif type(o).__module__ == numpy.__name__ and isinstance(o, numbers.Real):
                d = {'meta_class': 'numpy.number',
                     'dtype': str(o.dtype),
                     'data': base64.b64encode(o.tostring())
                     }
                return d

            # Normal Python types are unchanged
            elif isinstance(o, (int, long, basestring, float, bool, list, tuple)):
                return o
            # except dictionaries which are sorted
            elif isinstance(o, dict):
                d = collections.OrderedDict()
                d.update(sorted((item for item in o.iteritems()), smartcmp))
                return d
            # These two defaults are catch-all
            elif isinstance(o, numbers.Integral):
                return int(o)
            elif isinstance(o, numbers.Real):
                return float(o)
            elif isinstance(o, (numpy.bool, numpy.bool_)):
                return bool(o)
            elif tolerant:
                return None
            else:
                raise ValueError("Cannot encode in json object %r" % o)
        return json.dumps(obj, default=custom, indent=indent)

    @staticmethod
    def from_json(s, objectify=True, mapper={}):
        """Decodes json_plus.
         @param s : the string to decode
         @param objectify : If True, reconstructs the object hierarchy.
         @param mapper :
            - If a dictonary, then the key classes are replaced by the value classes in the
                decoding.
            - If a class, then all objects that are not dates or numpy classes are decoded to
              this class.
            - If None, then all objects that are not dates or numpy classes are decoded to
              json_plus.Serializable."""
        def hook(o):
            meta_module, meta_class = None, o.get('meta_class')
            if meta_class in ('Datetime', 'datetime.datetime'):
                # 'Datetime' included for backward compatibility
                try:
                    tmp = datetime.datetime.strptime(
                        o['date'], '%Y-%m-%dT%H:%M:%S.%f')
                except Exception, e:
                    tmp = datetime.datetime.strptime(
                        o['date'], '%Y-%m-%dT%H:%M:%S')
                return tmp
            elif meta_class == 'set':
                return set(o['set'])
            # Numpy arrays.
            elif meta_class == 'numpy.ndarray':
                data = base64.b64decode(o['data'])
                dtype = o['dtype']
                shape = o['shape']
                v = numpy.frombuffer(data, dtype=dtype)
                v = v.reshape(shape)
                obj = v.copy()
                obj.flags.writeable = True
                return obj
            elif meta_class == 'numpy.matrix':
                data = base64.b64decode(o['data'])
                dtype = o['dtype']
                shape = o['shape']
                v = numpy.frombuffer(data, dtype=dtype)
                v = v.reshape(shape)
                obj = numpy.matrix(v.copy())
                obj.flags.writeable = True
                return obj
            # Numpy numbers.
            elif meta_class == 'numpy.number':
                data = base64.b64decode(o['data'])
                dtype = o['dtype']
                v = numpy.frombuffer(data, dtype=dtype)[0]
                return v

            elif meta_class and '.' in meta_class:
                # correct for classes that have migrated from one module to another
                meta_class = mapper.get(meta_class, meta_class)
                meta_class = remapper.get(meta_class, meta_class)
                # separate the module name from the actual class name
                meta_module, meta_class = meta_class.rsplit('.',1)

            if meta_class is not None:
                del o['meta_class']
                if mapper is None:
                    obj = Serializable()
                    obj.__dict__.update(o)
                    o = obj
                elif isinstance(mapper, dict):
                    # this option is for backward compatibility in case a module is not specified
                    if meta_class in fallback:
                        meta_module = fallback.get(meta_class)

                    if meta_module is not None and objectify:
                        try:
                            module = importlib.import_module(meta_module)
                            cls = getattr(module, meta_class)
                            obj = cls()
                            obj.__dict__.update(o)
                            o = obj
                        except Exception, e:
                            # If an object is unknown, restores it as a member
                            # of this same class.
                            obj = Serializable()
                            obj.__dict__.update(o)
                            o = obj
                else:
                    # Map all to the specified class.
                    obj = mapper()
                    obj.__dict__.update(o)
                    o = obj
            elif type(o).__name__ == 'dict':
                # For convenience we deserialize dict into Storage.
                o = Storage(o)
            return o

        return json.loads(s, object_hook=hook)

    @staticmethod
    def loads(s):
        return Serializable.from_json(s)

loads = Serializable.loads
dumps = Serializable.dumps

class TestSerializable(unittest.TestCase):

    def test_simple(self):
        a = Serializable()
        a.x = 1
        a.y = 'test'
        a.z = 3.14
        b = Serializable.from_json(a.to_json())
        self.assertEqual(a, b)

    def test_datetime(self):
        a = Serializable()
        a.x = datetime.datetime(2015,1,3)
        b = Serializable.from_json(a.to_json())
        self.assertEqual(a, b)

    def test_recursive(self):
        a = Serializable()
        a.x = Serializable()
        a.x.y = 'test'
        b = Serializable.from_json(a.to_json())
        self.assertEqual(a, b)

    def test_numpy(self):
        a = Serializable()
        a.x = numpy.array([[1,2,3],[4,5,6]], dtype=numpy.int32)
        b = Serializable.from_json(a.to_json(pack_ndarray=True))
        self.assertEqual(numpy.sum(numpy.abs(a.x - b.x)), 0)

    def test_numpy_twice(self):
        a = Serializable()
        a.x = numpy.array([[1,2,3],[4,5,6]], dtype=numpy.int32)
        b = Serializable.from_json(a.to_json(pack_ndarray=True))
        self.assertEqual(numpy.sum(numpy.abs(a.x - b.x)), 0)
        c = Serializable.from_json(b.to_json(pack_ndarray=True))
        self.assertEqual(numpy.sum(numpy.abs(a.x - c.x)), 0)

    def test_numpy_direct(self):
        a = numpy.array([[1,2,3],[4,5,6]], dtype=numpy.int32)
        s = Serializable.dumps(a, pack_ndarray=True)
        c = Serializable.from_json(s)
        self.assertEqual(numpy.sum(numpy.abs(a - c)), 0)

    def test_float(self):
        x = numpy.float16(3.5)
        y = Serializable.from_json(Serializable.dumps(x))
        self.assertAlmostEqual(y, x, 2)

    def test_numpy_uint32(self):
        x = numpy.uint32(55)
        s = Serializable.dumps(x)
        y = Serializable.from_json(s)
        self.assertEqual(x, y)
        self.assertEqual(str(x.dtype), 'uint32')
        self.assertEqual(str(y.dtype), 'uint32')

    def test_numpy_float128(self):
        x = numpy.float128(55.3)
        s = Serializable.dumps(x)
        y = Serializable.from_json(s)
        self.assertAlmostEqual(x, y, 5)
        self.assertEqual(str(x.dtype), 'float128')
        self.assertEqual(str(y.dtype), 'float128')

    def test_set(self):
        s = set(['a', 'b', 'c'])
        x = Serializable.dumps(s)
        t = Serializable.loads(x)
        self.assertEqual(s, t)

    def test_multiple_dicts(self):
        d = dict(cane=4, gatto=4, uccello=2)
        d1 = Serializable.loads(Serializable.dumps(d))
        d2 = Serializable.loads(Serializable.dumps(d1))
        for k in d.keys():
            self.assertEqual(d.get(k), d2.get(k))
        for k in d2.keys():
            self.assertEqual(d.get(k), d2.get(k))

    def test_modifiable(self):
        a = numpy.zeros((10,10))
        b = loads(dumps(a))
        a[2:4, 5:6] = 1
        b[2:4, 5:6] = 1
        self.assertEqual(numpy.sum(numpy.abs(a - b)), 0)

    def test_matrices(self):
        a = numpy.matrix(numpy.ones((4,5)))
        b = numpy.matrix(numpy.ones((5, 6)))
        ab = a * b
        # print "Serialization:", dumps(a)
        aa = loads(dumps(a))
        bb = loads(dumps(b))
        # print "Deserialied types:", type(aa), type(bb)
        aabb = aa * bb
        self.assertEqual(numpy.sum(numpy.abs(ab - aabb)), 0)

if __name__ == '__main__':
    unittest.main()
