# Library for rapid prototyping

The library is supposed to be used in the interactive python shell.
That is, we run the python shell, import relevant parts of the library,
and then we can quickly and conveniently create and run experiments.

We briefly summarize each submodule's role in the library.

`testdata.py` contains the code for loading and preprocessing the
training and validation data. It provides methods for accessing these
data to other modules.

`functional.py` contains various functions that are relevant for
the neural networks. For example, it contains the "flatten" function,
which flattens the input tensor. Also contains functions whose role
is to measure a certain metric, such as accuracy or cross entropy.

`training.py` contains the code for training neural networks and
validating them on the test data.

`util.py` contains various utilities, such as plotting graphs of
training and validation error/accuracy, plotting images, saving models
onto the hard drive and model inspection, such as tracking gradients
flowing through a weight during computation.

The folder `models` contains the logic used to create models. `general.py`
contains subclasses of torch.nn.Module that can be used inside models.
`creation.py` and `constructors.py` enable us to take a more functional
approach towards creating models: instead of having to write a class
which subclasses torch.nn.Module, we just call the right function
with the right arguments (some of which are function with their own
arguments, ...).

The folder `experiments` contains modules which are meant to be imported
from the python shell. These modules load everything necessary for the
user to be able to work with data, train a specific network architecture
with a specific combination of enhancements, ...
