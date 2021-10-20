import tensorflow as tf

__all__ = ['TFDict']


class TFDict:

    def __init__(self):

        self.tf_keys = []

        self.tfd_graph = tf.Graph()

    def find(self, key):

        nodes = []

        for i in range(len(self.tf_keys)):

            if tf.math.reduce_all(self.tf_keys[i].deref() == key.deref()):

                nodes += self.tfd_graph.get_collection(name=self.tf_keys[i])

        return nodes

    def get(self, key):

        nodes = self.tfd_graph.get_collection(name=key)

        if len(nodes) != 0:

            return nodes

        return self.find(key)

    def build(self, keys, values):

        n = len(keys)

        def _add(i):

            if i >= n:

                return

            self.add(keys[i], values[i])

        _add(0)

    def add(self, key, value):

        self.tf_keys.append(key)

        self.tfd_graph.add_to_collection(name=self.tf_keys[-1], value=value)

    def __len__(self):

        return len(self.tf_keys)

    def __getitem__(self, key):

        return self.get(key)
