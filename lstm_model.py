from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

class LSTMModel:
    def __init__(self, input_shape):
        self.model = Sequential()
        self.model.add(LSTM(units=50, activation='relu', return_sequences=True, input_shape=input_shape))
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(units=60, activation='relu', return_sequences=True))
        self.model.add(Dropout(0.3))
        self.model.add(LSTM(units=80, activation='relu', return_sequences=True))
        self.model.add(Dropout(0.4))
        self.model.add(LSTM(units=120, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(units=1))  # Output layer for regression

    def compile_model(self, learning_rate=0.001):
        # Compile with Adam optimizer
        from tensorflow.keras.optimizers import Adam
        optimizer = Adam(learning_rate=learning_rate)
        self.model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])

    def get_model(self):
        return self.model

