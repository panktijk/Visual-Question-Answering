from keras.models import Sequential
from keras.layers.core import Reshape, Activation, Dropout
from keras.layers import LSTM, Merge, Dense

def build_model():
    image_feature_size          = 1000
    word_feature_size           = 4096
    number_of_LSTM              = 3
    number_of_hidden_units_LSTM = 512
    max_length_questions        = 30
    number_of_dense_layers      = 2
    number_of_hidden_units      = 1024
    activation_function         = 'tanh'
    dropout_pct                 = 0.5


    # Image model
    model_image = Sequential()
    # model_image.add(Reshape((image_feature_size,), input_shape=(image_feature_size,)))

    # Language Model
    model_language = Sequential()
    # model_language.add(LSTM(number_of_hidden_units_LSTM, return_sequences=True, input_shape=(max_length_questions, word_feature_size)))
    # model_language.add(LSTM(number_of_hidden_units_LSTM, return_sequences=True))
    # model_language.add(LSTM(number_of_hidden_units_LSTM, return_sequences=False))

    # combined model
    model = Sequential()
    # model.add(Merge([model_language, model_image], mode='concat', concat_axis=1))
    model.add(Dense(number_of_hidden_units, input_shape=((image_feature_size+word_feature_size), ), kernel_initializer='uniform'))
    model.add(Activation(activation_function))
    model.add(Dropout(dropout_pct))

    for _ in range(number_of_dense_layers):
        model.add(Dense(number_of_hidden_units, kernel_initializer='uniform'))
        model.add(Activation(activation_function))
        model.add(Dropout(dropout_pct)) #to prevent overfitting

    model.add(Dense(1))
    model.add(Activation('softmax'))

    return model
    