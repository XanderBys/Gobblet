import keras
from keras.layers import Dense
from keras.layers import Add
from keras.layers import Lambda
from keras.layers.advanced_activations import LeakyReLU
import keras.backend as K
import numpy as np

class Model:
    def __init__(self, num_states, num_actions, dueling=True):
        self.num_states = num_states
        self.num_actions = num_actions
        self.dueling = dueling
        
        # placeholders - will be initialized in define_model
        self.nn = None
        
        self.define_model()
    
    def define_model(self):
        branches = []
        ALPHA = 0.6
        inputs = keras.Input(shape=(self.num_states,))
        
        # split into two branches: one for location, one for piece
        processed_inputs = Dense(100)(inputs)
        processed_inputs = LeakyReLU(alpha=ALPHA)(processed_inputs)
        
        if self.dueling:
            state_value = Dense(1)(processed_inputs)
            state_value = LeakyReLU(alpha=ALPHA)(state_value)
            state_value = Lambda(lambda s: K.expand_dims(s[:, 0], -1), output_shape=(self.num_actions[0],))(state_value)
            
            advantage = Dense(self.num_actions[0])(processed_inputs)
            advantage = LeakyReLU(alpha=ALPHA)(advantage)
            advantage = Lambda(lambda a: a[...] - K.mean(a[...], keepdims=True), output_shape=(self.num_actions[0],))(advantage)
        
            location = Add()([state_value, advantage])
            
        else:
            location = Dense(self.num_actions[0])(processed_inputs)
            location = LeakyReLU(alpha=ALPHA)(location)
        
        #location_nn = keras.Model(inputs=inputs, outputs=location, name="Location Model")
        piece = keras.layers.Concatenate()([processed_inputs, location])
        
        if self.dueling:
            state_value = Dense(1)(piece)
            state_value = LeakyReLU(alpha=ALPHA)(state_value)
            state_value = Lambda(lambda s: K.expand_dims(s[:, 0], -1), output_shape=(self.num_actions[1],))(state_value)
            
            advantage = Dense(self.num_actions[1])(piece)
            advantage = LeakyReLU(alpha=ALPHA)(advantage)
            advantage = Lambda(lambda a: a[...] - K.mean(a[...], keepdims=True), output_shape=(self.num_actions[1],))(advantage)
        
            piece = Add()([state_value, advantage])
            
        else:
            piece = Dense(self.num_actions[1])(piece)
            piece = LeakyReLU(alpha=ALPHA)(piece)
        
        self.nn = keras.Model(inputs=inputs, outputs=[location, piece], name="Weird DDDQN Model")
        
        # use mean squared error loss and Adam optimizer
        self.nn.compile(loss=keras.losses.mean_squared_error,
                        optimizer='adam', metrics=['accuracy'])
        keras.utils.plot_model(self.nn, to_file="/home/pi/programs/Gobblet/weird_model_vizualization.png", show_shapes=True, show_layer_names=True)
    
    def copy_weights(self, other):
        # copies weights from this neural network to the another network
        for main_layer, other_layer in zip(self.nn.layers, other.nn.layers):
            weights = main_layer.get_weights()
            other_layer.set_weights(weights)
            
    def predict_one(self, board):
        return self.nn.predict(np.array(board, ndmin=2))
    
    def predict_batch(self, boards):
        return self.nn.predict(boards)
    
    def train_batch(self, x_batch, y_batch, batch, use_fit=True):
        if use_fit:
            return self.nn.fit(x_batch, y_batch, verbose=0, batch_size=batch,epochs=1)
        else:
            return self.nn.train_on_batch(x_batch, y_batch)
    
if __name__ == '__main__':
    model = Model(16, [16, 12], True)