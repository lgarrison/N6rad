N6rad
=====
A GPU implementation of a simple radiation transfer algorithm that considers all N^6 pairwise cell interactions for a N^3 grid.

This is currently experimental and doesn't do any radiation transfer, just a dummy pairwise operation.

Usage
-----
Set the parameters at the top of `n6rad.cu`.  To run:
```console
$ make
$ ./n6rad
```

Author
------
Lehman Garrison

License
-------
Apache-2.0
