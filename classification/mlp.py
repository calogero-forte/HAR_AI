"""
This module includes classes and functions
for a multi layer perceptron network
"""

#------------------------------
# Import
#------------------------------
from tensorflow.experimental import Optional
from keras import Sequential
from keras.layers import Input, Dense

# pyrefly: ignore [missing-import]
from classification.base_classifier import BaseClassifier

#------------------------------
# Class
#------------------------------
class MLP(BaseClassifier):

    def __init__(self, input_shape_i, hidden_layers_i: Optional[list[int]] = None, output_units_i: Optional[int] = None,
        hidden_activation_i: Optional[list[str]] = None, output_activation_i: Optional[str] = None) -> None:
        
        self.__model = Sequential(name='MLP Network')
        self.add_hidden_layers(hidden_layers_i, hidden_activation_i)
        self.add_output_layer(output_units_i, output_activation_i)
        
    
    #----------------------------------------
    
    def add_hidden_layers(self, hidden_layers_i: list[int], hidden_activation_i: Optional[list[str]] = None) -> None:
        """
        Add one or a multiple hidden layers with the specified number of units and activation function

        Parameters
        ----------
        hidden_layers_i : list[int]
            number of units for each hidden layer
        hidden_activation_i : list[str]
            activation function for each hidden layer.
            If None, ReLU is used for all hidden layers.

        Return
        ------
        None
        """

        if(hidden_activation_i == None):
            hidden_activation_i = ['relu']*len(hidden_layers_i)
        
        for i in range(len(hidden_layers_i)):
            self.__model.add(
                Dense(
                    units=hidden_layers_i[i],
                    activation=hidden_activation_i[i],
                    name=f"hidden_layer_{i}"
                )
            )    
    
    #----------------------------------------

    def add_output_layer(self, output_units_i: int, output_activation_i: Optional[str] = None) -> None:
        """
        Add the output layer with the specified number of units and activation function

        Parameters
        ----------
        output_units_i : int
            number of units for the output layer
        output_activation_i : str
            activation function for the output layer.
            If None, softmax is used for all output layers.

        Return
        ------
        None
        """

        if(output_activation_i == None):
            output_activation_i = 'softmax'
        
        self.__model.add(
            Dense(
                units=output_units_i,
                activation=output_activation_i,
                name="output_layer"
            )
        ) 

        #----------------------------------------

        def compilte(self, optimizer_i='adam', loss_i='sparse_categorical_crossentropy', metrics_i=['accuracy']) -> None:
            self.__model.compile(
                optimizer=optimizer_i,
                loss=loss_i,
                metrics=metrics_i
            )

        #----------------------------------------

        def print_summary(self) -> None:
            self.__model.summary()      

       



