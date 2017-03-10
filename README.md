# Experimental MMTF I/O Python Library

**simplemmtf** is a lightweight [MMTF](http://mmtf.rcsb.org/) encoding and decoding module written in Python. It was originally developed for PyMOL 1.8.4 to prototype MMTF load support.

This implementation is similar (but unrelated) to [mmtf-python](https://github.com/rcsb/mmtf-python).

## Dependencies

* [msgpack-python](https://github.com/polyglotted/msgpack-python) or [u-msgpack-python](https://github.com/vsergeev/u-msgpack-python)
* numpy (optional)

## Examples

```python
>>> import simplemmtf
>>> d = simplemmtf.fetch('1rx1')
>>> d.get('mmtfVersion')
'0.2.0'
```

## License

Copyright 2017 [Schrodinger, Inc.](http://www.schrodinger.com/)
Published under the **BSD-3-Clause** license.
