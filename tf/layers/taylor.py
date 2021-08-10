import tensorflow as tf

__all__ = ['NthOrderDiff', 'AutoTaylorExpansion']


class NthOrderDiff:

    def __init__(self, x, func, none_to_zero=False):

        self.diff_list = None

        self.y = None
        self.func = func

        self.x = x
        self.none_to_zero = none_to_zero

    def compute_diff(self, order):

        self.reset()

        def nth_autodiff(ith_tape, _ord):

            if _ord == 0:
                return

            with ith_tape:

                ith_tape.watch(self.x)

                nth_autodiff(tf.GradientTape(), _ord - 1)

                if _ord == 1:

                    self.y = self.func(self.x)

            self.y = ith_tape.gradient(self.y, self.x)

            if self.y is None and self.none_to_zero:
                self.y = tf.cast(0.0, dtype=tf.float32)

            self.diff_list.append(self.y)

        start = tf.GradientTape()

        nth_autodiff(ith_tape=start, _ord=order)

    def reset(self):

        self.diff_list = []

        self.y = None


class AutoTaylorExpansion(tf.keras.layers.Layer):

    def __init__(self, a, func, n_terms, *arg, **kwargs):

        """

        """

        self.a = tf.constant(a, dtype=tf.float32)

        self.func = func
        self.n_terms = n_terms

        self._super_tape = NthOrderDiff(x=self.a, func=self.func, none_to_zero=True)
        self._super_tape.compute_diff(order=self.n_terms)

        super(AutoTaylorExpansion, self).__init__(*arg, **kwargs)

    @staticmethod
    def fac(i):
        return tf.exp(tf.math.lgamma(i + 1.0))

    def _find_expansion(self, inputs):

        diff = self._super_tape.diff_list
        expansion = []

        def _cond(i):

            return i < self.n_terms

        def loop(i, cond):

            if not cond(i):
                return

            p = tf.cast(i, dtype=tf.float32)

            c0 = (inputs - self.a) ** p
            c1 = AutoTaylorExpansion.fac(p)
            c2 = tf.gather(diff, i)

            t = (c0 * c2) / c1

            expansion.append(t)

            loop(i + 1, cond)

        loop(0, _cond)

        expansion = tf.stack(expansion, axis=-1)

        return expansion

    def call(self, inputs):

        expansion = self._find_expansion(inputs)

        return expansion
