# Dependencies Installation

## [PyFusion](https://github.com/griegler/pyfusion)
* Step 1: make sure the gcc version is no more than 7.
* Step 2: follow the instructions in the README of the PyFusion.
* Step 3: set path as follows.
```
    export LD_LIBRARY_PATH=path_to/pyfusion/build:$LD_LIBRARY_PATH
    export PYTHONPATH=path_to/pyfusion:$PYTHONPATH
    export PYTHONPATH=path_to/:$PYTHONPATH
```