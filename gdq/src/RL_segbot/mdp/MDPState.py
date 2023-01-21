import numpy as np


class MDPStateClass(object):
    def __init__(self, data, is_terminal=False):
        self.data = data
        self._is_terminal = is_terminal

    # Accessors

    def get_data(self):
        return self.data

    def is_terminal(self):
        return self._is_terminal

    # Setters

    def set_data(self, data):
        self.data = data

    def set_terminal(self, is_terminal=True):
        self._is_terminal = is_terminal

    # Core

    def __hash__(self):
        if type(self.data).__module__ == np.__name__:
            # Numpy arrays
            return hash(str(self.data))
        elif self.data.__hash__ is None:
            return hash(tuple(self.data))
        else:
            return hash(self.data)

    def __eq__(self, other):
        assert isinstance(other, MDPStateClass), "Arg object is not in " + type(self).__module__
        return self.data == other.data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

