import keras
from keras.models import Sequential, Model
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers import Bidirectional, Activation, Input, Concatenate, Reshape, add, maximum
from keras.layers import dot, multiply, MaxPooling1D, TimeDistributed, MaxPooling2D
from keras_contrib.layers import CRF

class net():

    def __init__(self, embeddings_list=[], trainable_embeds=True, rnn_units=300, embed_dim=300):

        self.rnn_units = rnn_units
        self.embed_dim = embed_dim
        self.embeddings_list = embeddings_list
        self.input_cols = []
        self.train_embeds = trainable_embeds

        self.nn1 = None
        self.nn_rep = None
        self.crf = None

        print('n_classes: {}'.format(max(self.embeddings_list[-1][0].values())))
        super(net, self).__init__()


    def base(self, inputs, sample_shape=(None,), drop=.2):
        #####Variable model inputs
        inputs_dic = {}
        for it in inputs:
            inputs_dic[it] = Input(shape=sample_shape, name=str(it))
        self.input_cols = inputs_dic.keys()
        embeddings = Embedding(input_dim=len(self.embeddings_list[0][0]),
                               output_dim=self.embeddings_list[0][1],
                               weights=[self.embeddings_list[0][2]],
                               name='encoder_embeddings')
        embeddings.trainable = self.train_embeds

        ############################################################
        ####### Main encoder
        ############################################################
        encoder_lstm_inputs = None
        if len(self.input_cols) > 1:
            encoder_lstm_inputs = Concatenate()([embeddings(in_val) for in_val in inputs_dic.values()])
        else:
            encoder_lstm_inputs = embeddings(list(inputs_dic.values())[0])
        encoder_lstm = Bidirectional(LSTM(self.rnn_units,
                                          activation='relu',
                                          return_sequences=True,
                                          return_state=True,
                                          ))
        encoder_outputs, fh, fc, bh, bc = encoder_lstm(encoder_lstm_inputs)
        encoded_states = [fh, fc, bh, bc]

        ############################################################
        ####### Both decoder_layers
        ############################################################
        # Training decoder inputs
        decoder_inputs = Input(shape=(None,), name='training_decoder_input')
        decoder_embeddings = Embedding(input_dim=len(self.embeddings_list[-1][0]),
                                       output_dim=self.embeddings_list[-1][1],
                                       weights=[self.embeddings_list[-1][2]],
                                       name='decoder_embeddings')
        decoder_embeddings.trainable = self.train_embeds
        decoder_data = decoder_embeddings(decoder_inputs)

        # Training decoder layers
        decoder_LSTM = Bidirectional(LSTM(self.rnn_units,
                                            activation='relu',
                                            return_sequences=True,
                                            return_state=True,
                                            name='decoder_lstm_relu',
                                            ))

        ############################################################
        ####### NN1 LSTM output layers
        ############################################################
        #CM Lstm Add'l layers
        self.crf = CRF(len(self.embeddings_list[-1][0]), learn_mode='marginal')

        ############################################################
        ####### Both NNs implementation
        ############################################################
        # Training decoder implementation
        decoder_outputs1 = decoder_LSTM(decoder_data, initial_state=encoded_states)
        decs = Dropout(drop)(decoder_outputs1[0])

        ############################################################
        ####### NN1 implementation
        ############################################################
        train_out = self.crf(decoder_outputs2_0[0])
        train_model = Model(list(inputs_dic.values()) + [decoder_inputs], train_out)
        train_model.compile(optimizer='adam', loss=self.crf.loss_function, metrics=[self.crf.accuracy])
        print('decoder_1 (NN1): {}'.format(train_model.metrics_names))
        self.nn1 = train_model

        ############################################################
        ####### Decoder state data
        ############################################################
        #representations_output = Concatenate()(decoder_outputs2_0[1:]+decoder_outputs1[1:])
        representations = Model(list(inputs_dic.values()) + [decoder_inputs], outputs=decoder_outputs2_0[1:]+decoder_outputs1[1:] #decoder_outputs2_0+decoder_outputs1
                                )
        self.nn_rep = representations
