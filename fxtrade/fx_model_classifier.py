from keras.layers.core import Activation
from keras.layers.core import Dense
from keras.layers.core import Dropout
from keras.models import Sequential

from keras.layers.recurrent import LSTM
from keras.callbacks import EarlyStopping
from keras.initializers import glorot_uniform
from keras.initializers import orthogonal


class FxClassifier:
    def __init__(self, maxlen, n_hidden, n_in, n_out):
        self.maxlen = maxlen
        self.n_hidden = n_hidden
        self.n_in = n_in
        self.n_out = n_out

    def create_model(self):
        model = Sequential()
        model.add(LSTM(self.n_hidden, batch_input_shape=(None, self.maxlen, self.n_in),
                       kernel_initializer=glorot_uniform(seed=20170719),
                       recurrent_initializer=orthogonal(gain=1.0, seed=20170719),
                       dropout=0.3,
                       recurrent_dropout=0.3))
        model.add(Dropout(0.3))
        model.add(Dense(self.n_out,
                        kernel_initializer=glorot_uniform(seed=20170719)))
        model.add(Activation("softmax"))
        model.compile(loss="categorical_crossentropy", optimizer="RMSprop", metrics=['categorical_accuracy'])
        return model

    # 学習
    def train(self, epochs, train_gen, train_steps, validation_gen, validation_steps):
        early_stopping = EarlyStopping(patience=0, verbose=1)
        model = self.create_model()
        '''
        history = model.fit_generator(generator=train_gen,
                                      steps_per_epoch=train_steps,
                                      epochs=epochs,
                                      verbose=1,
                                      callbacks=[early_stopping],
                                      validation_data=validation_gen,
                                      validation_steps=validation_steps)
        '''
        history = model.fit_generator(generator=train_gen,
                                      steps_per_epoch=train_steps,
                                      epochs=epochs,
                                      verbose=0,
                                      use_multiprocessing=True,
                                      workers=2,
                                      callbacks=[early_stopping])
        return model, history
