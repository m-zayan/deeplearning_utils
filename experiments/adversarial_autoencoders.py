import tensorflow as tf

from tensorflow.keras import optimizers

from tensorflow.keras.layers import Input, Dense, Activation, Concatenate, GaussianDropout
from tensorflow.keras.models import Model

from . import functional

from utils.external.common import OS

__all__ = ['AdversarialAutoencoder']


class AdversarialAutoencoder:

    def __init__(self, input_dim, hidden_dim, style_dim, n_classes, **kwargs):

        self.input_dim = input_dim

        self.hidden_dim = hidden_dim

        self.style_dim = style_dim

        self.n_classes = n_classes

        self.dropout_rate = kwargs.pop('input_dropout_rate', None)

        self.encoder_activation = kwargs.pop('encoder_activation', 'relu')

        self.style_interloper_units = kwargs.pop('style_interloper_units', 1)
        self.y_interloper_units = kwargs.pop('style_interloper_units', 1)

        # -------------------------

        self.base_encoder = None

        self.style_discriminator = None

        self.y_discriminator = None

        self.decoder = None

        self.style_interloper = None
        self.y_interloper = None

        # -------------------------

        self.__model_attr__ = ['base_encoder',  'style_discriminator', 'y_discriminator',
                               'decoder', 'style_interloper', 'y_interloper']

    def style_gaussian_sample(self, batch_size, mu=0.0, sigma=1.0, seed=None):

        sample = tf.random.normal((batch_size, self.style_dim), mean=mu, stddev=sigma, seed=seed)

        return sample

    def cat_multinomial_sample(self, batch_size, seed=None):

        sample = tf.random.uniform((batch_size,), minval=0, maxval=self.n_classes, dtype=tf.int32, seed=seed)

        return sample

    def cat_one_hot_sample(self, batch_size, seed=None):

        sample = self.cat_multinomial_sample(batch_size, seed)

        sample = tf.one_hot(sample, self.n_classes)

        return sample

    def encoder_block(self, inputs):

        z = Dense(self.hidden_dim)(inputs)
        z = Activation(self.encoder_activation)(z)

        z = Dense(self.hidden_dim)(z)
        z = Activation(self.encoder_activation)(z)

        return z

    def build_base_encoder(self):

        """ phase [-1]: base-encoder q(z|x) & q(y|x) """

        inputs = Input(shape=(self.input_dim,))

        if self.dropout_rate is not None:

            z = GaussianDropout(self.dropout_rate)(inputs)

        else:

            z = inputs

        z = self.encoder_block(z)

        self.base_encoder = Model(inputs, z)

    def build_style_discriminator(self):

        """ phase [0]: q(z|x) """

        inputs = Input(shape=(self.hidden_dim,))

        z_style = Dense(self.style_dim, name='z_style')(inputs)

        self.style_discriminator = Model(inputs, z_style)

    def build_y_discriminator(self):

        """ phase [0]: q(y|x) """

        inputs = Input(shape=(self.hidden_dim,))

        y_posterior = Dense(self.n_classes)(inputs)
        y_posterior = Activation('softmax', name='y_posterior')(y_posterior)

        self.y_discriminator = Model(inputs, y_posterior)

    def build_decoder(self, activation):

        """ phase [1]: decoder """

        inputs = Input(shape=(self.input_dim,))

        z = self.base_encoder(inputs)

        z_style = self.style_discriminator(z)
        y_posterior = self.y_discriminator(z)

        merge = Concatenate(name='merge_yz')([y_posterior, z_style])

        z = self.encoder_block(merge)

        x_posterior = Dense(self.input_dim)(z)
        x_posterior = Activation(activation, name='x_posterior')(x_posterior)

        self.decoder = Model(inputs, x_posterior)

    def build_style_interloper(self, activation):

        """ phase [2]: q(z|x) matches arbitrary prior """

        inputs = Input(shape=(self.style_dim,))

        z = self.encoder_block(inputs)

        # e.g. discrete-style
        new_style = Dense(self.style_interloper_units)(z)
        new_style = Activation(activation, name='imposed_z')(new_style)

        self.style_interloper = Model(inputs, new_style)

    def build_y_interloper(self, activation):

        """ phase [2]: q(y|x) matches categorical distribution """

        inputs = Input(shape=(self.n_classes, ))

        z = self.encoder_block(inputs)

        # categorical distribution
        new_style = Dense(self.y_interloper_units)(z)
        new_style = Activation(activation, name='imposed_y')(new_style)

        self.y_interloper = Model(inputs, new_style)

    def build(self, **kwargs):

        decoder_activation = kwargs.pop('decoder_activation', 'sigmoid')
        style_interloper_activation = kwargs.pop('style_interloper_activation', 'sigmoid')
        y_interloper_activation = kwargs.pop('y_interloper_activation', 'sigmoid')

        self.build_base_encoder()

        self.build_style_discriminator()

        self.build_y_discriminator()

        self.build_decoder(activation=decoder_activation)

        self.build_style_interloper(activation=style_interloper_activation)
        self.build_y_interloper(activation=y_interloper_activation)

    def compile_style_discriminator(self, use_sgd, **kwargs):

        if use_sgd:

            learning_rate = kwargs.pop('style_discriminator_lr', 0.1)
            momentum = kwargs.pop('style_discriminator_momentum', 0.9)

            optimizer = optimizers.SGD(learning_rate=learning_rate, momentum=momentum)

        else:

            learning_rate = kwargs.pop('style_discriminator_lr', 1e-3)
            beta_1 = kwargs.pop('style_discriminator_beta_1', 0.9)
            beta_2 = kwargs.pop('style_discriminator_beta_2', 0.999)

            optimizer = optimizers.Adam(learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2)

        self.style_discriminator.compile(optimizer=optimizer)

    def compile_y_discriminator(self, use_sgd, **kwargs):

        if use_sgd:

            learning_rate = kwargs.pop('y_discriminator_lr', 0.1)
            momentum = kwargs.pop('y_discriminator_momentum', 0.9)

            optimizer = optimizers.SGD(learning_rate=learning_rate, momentum=momentum)

        else:

            learning_rate = kwargs.pop('y_discriminator_lr', 1e-3)
            beta_1 = kwargs.pop('y_discriminator_beta_1', 0.9)
            beta_2 = kwargs.pop('y_discriminator_beta_2', 0.999)

            optimizer = optimizers.Adam(learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2)

        self.y_discriminator.compile(optimizer=optimizer)

    def compile_decoder(self, use_sgd, **kwargs):

        if use_sgd:

            learning_rate = kwargs.pop('decoder_lr', 0.1)
            momentum = kwargs.pop('decoder_momentum', 0.9)

            optimizer = optimizers.SGD(learning_rate=learning_rate, momentum=momentum)

        else:

            learning_rate = kwargs.pop('decoder_lr', 1e-3)
            beta_1 = kwargs.pop('decoder_beta_1', 0.9)
            beta_2 = kwargs.pop('decoder_beta_2', 0.999)

            optimizer = optimizers.Adam(learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2)

        self.decoder.compile(optimizer=optimizer)

    def compile_style_interloper(self, use_sgd, **kwargs):

        if use_sgd:

            learning_rate = kwargs.pop('style_interloper_lr', 0.1)
            momentum = kwargs.pop('style_interloper_momentum', 0.1)

            optimizer = optimizers.SGD(learning_rate=learning_rate, momentum=momentum)

        else:

            learning_rate = kwargs.pop('style_interloper_lr', 1e-3)
            beta_1 = kwargs.pop('style_interloper_beta_1', 0.9)
            beta_2 = kwargs.pop('style_interloper_beta_2', 0.999)

            optimizer = optimizers.Adam(learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2)

        self.style_interloper.compile(optimizer=optimizer)

    def compile_y_interloper(self, use_sgd, **kwargs):

        if use_sgd:

            learning_rate = kwargs.pop('y_interloper_lr', 0.1)
            momentum = kwargs.pop('y_interloper_momentum', 0.9)

            optimizer = optimizers.SGD(learning_rate=learning_rate, momentum=momentum)

        else:

            learning_rate = kwargs.pop('y_interloper_lr', 1e-3)
            beta_1 = kwargs.pop('y_interloper_beta_1', 0.9)
            beta_2 = kwargs.pop('y_interloper_beta_2', 0.999)

            optimizer = optimizers.Adam(learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2)

        self.y_interloper.compile(optimizer=optimizer)

    def compile(self, use_sgd=False, **kwargs):

        self.compile_style_discriminator(use_sgd=use_sgd, **kwargs)

        self.compile_y_discriminator(use_sgd=use_sgd, **kwargs)

        self.compile_decoder(use_sgd=use_sgd, **kwargs)

        self.compile_style_interloper(use_sgd=use_sgd, **kwargs)
        self.compile_y_interloper(use_sgd=use_sgd, **kwargs)

    def style_interloper_predict(self, x, training=False):

        z = self.base_encoder(x, training=training)
        z_style = self.style_discriminator(z, training=training)

        z_posterior = self.style_interloper(z_style, training=training)

        return z_posterior

    def y_interloper_predict(self, x, training=False):

        z = self.base_encoder(x, training=training)
        y_posterior = self.y_discriminator(z, training=training)

        cat_posterior = self.y_interloper(y_posterior, training=training)

        return cat_posterior

    def decoder_predict(self, x, training=False):

        x_posterior = self.decoder(x, training=training)

        return x_posterior

    def style_discriminator_predict(self, x, training=False):

        z = self.base_encoder(x, training=training)
        z_style = self.style_discriminator(z, training=training)

        return z_style

    def y_discriminator_predict(self, x, training=False):

        z = self.base_encoder(x, training=training)
        y_posterior = self.y_discriminator(z, training=training)

        return y_posterior

    def supervised_train_step(self, x_prior, y_prior, sparse_y=True):

        """ i.e. semi-supervised or supervised """

        if sparse_y:

            y_prior = tf.one_hot(y_prior, self.n_classes)

        # -------------------------------------------------------

        # forward pass
        with tf.GradientTape() as tape:

            y_posterior = self.y_discriminator_predict(x_prior, training=True)

            loss = functional.cross_entropy_loss(y_prior, y_posterior)

        # -------------------------------------------------------

        # backward pass

        trainable_vars = self.base_encoder.trainable_variables
        trainable_vars += self.y_discriminator.trainable_variables

        gradients = tape.gradient(loss, trainable_vars)

        self.y_discriminator.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # -------------------------------------------------------

        return {'loss': loss}

    def style_adversarial_train_step(self, x_prior, arbitrary_style, update_discriminator=False):

        """ i.e. unsupervised, semi-supervised or supervised """

        # -------------------------------------------------------

        # forward pass
        with tf.GradientTape() as tape:

            z_posterior = self.style_interloper_predict(x_prior, training=True)

            z_prior = self.style_interloper(arbitrary_style, training=True)

            loss = functional.cross_entropy_loss(z_prior, z_posterior)

        # -------------------------------------------------------

        # backward pass

        trainable_vars = self.base_encoder.trainable_variables

        if update_discriminator:

            trainable_vars += self.style_discriminator.trainable_variables

        trainable_vars += self.style_interloper.trainable_variables

        gradients = tape.gradient(loss, trainable_vars)

        self.style_interloper.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # -------------------------------------------------------

        return {'loss': loss}

    def cat_adversarial_train_step(self, x_prior, arbitrary_cat, sparse_cat=True, update_discriminator=False):

        """ i.e. unsupervised, semi-supervised or supervised """

        if sparse_cat:

            arbitrary_cat = tf.one_hot(arbitrary_cat, self.n_classes)

        # -------------------------------------------------------

        # forward pass
        with tf.GradientTape() as tape:

            cat_posterior = self.y_interloper_predict(x_prior, training=True)

            cat_prior = self.y_interloper(arbitrary_cat, training=True)

            loss = functional.cross_entropy_loss(cat_prior, cat_posterior)

        # -------------------------------------------------------

        # backward pass

        trainable_vars = self.base_encoder.trainable_variables

        if update_discriminator:

            trainable_vars += self.y_discriminator.trainable_variables

        trainable_vars += self.y_interloper.trainable_variables

        gradients = tape.gradient(loss, trainable_vars)

        self.y_interloper.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # -------------------------------------------------------

        return {'loss': loss}

    def decoder_train_step(self, x_prior):

        """ i.e. unsupervised, semi-supervised or supervised """

        # -------------------------------------------------------

        # forward pass
        with tf.GradientTape() as tape:

            x_posterior = self.decoder_predict(x_prior, training=True)

            loss = functional.euclidean_norm_loss(x_prior, x_posterior)

        # -------------------------------------------------------

        # backward pass

        trainable_vars = self.decoder.trainable_variables

        gradients = tape.gradient(loss, trainable_vars)

        self.decoder.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # -------------------------------------------------------

        return {'loss': loss}

    def save_checkpoint(self, working_dir):

        working_dir = OS.realpath(working_dir)

        if not OS.dir_exists(working_dir):

            OS.make_dirs(working_dir)

        for attr_name in self.__model_attr__:

            model = getattr(self,  attr_name)

            if model is None:

                raise ValueError(f'{attr_name}=None, consider using .build(...)')

            path = OS.join(working_dir, f'{attr_name}.h5')

            model.save(path)

    def load_checkpoint(self, working_dir):

        working_dir = OS.realpath(working_dir)

        if not OS.dir_exists(working_dir):

            OS.make_dirs(working_dir)

        for attr_name in self.__model_attr__:

            model = getattr(self, attr_name)

            if model is None:

                raise ValueError(f'{attr_name}=None, consider using .build(...)')

            path = OS.join(working_dir, f'{attr_name}.h5')

            model.load_weights(path)
