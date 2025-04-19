# model.py
import torch  # import the main PyTorch library for tensors and operations
import torch.nn as nn  # import neural network modules and layers

class CustomCNN(nn.Module):  # define a custom convolutional neural network class
    def __init__(self, input_channels=3, num_filters=None, kernel_size=3,  
                 activation='relu', dense_neurons=128, num_classes=10,  
                 dropout_rate=0.3, batch_norm=True, num_conv_layers=5, input_size=224):
        super(CustomCNN, self).__init__()  # initialize base nn.Module
        self.activation = activation  # store activation function type
        self.batch_norm = batch_norm  # flag to include batch normalization
        self.dropout_rate = dropout_rate  # rate for dropout regularization

        # Prepare list of convolution filter sizes
        if num_filters is None:  # no filters provided
            num_filters = [32, 64, 128, 256, 512]  # use default filter sizes
        elif isinstance(num_filters, int):  # single integer provided
            num_filters = [num_filters] * num_conv_layers  # repeat it for each conv layer
        elif isinstance(num_filters, list):  # list provided
            assert len(num_filters) >= num_conv_layers  # ensure enough filters are defined

        # Prepare list of kernel sizes
        if isinstance(kernel_size, int):  # single kernel size provided
            kernel_size = [kernel_size] * num_conv_layers  # repeat for each layer
        elif isinstance(kernel_size, list):  # list of sizes provided
            assert len(kernel_size) >= num_conv_layers  # ensure enough sizes are defined
        else:
            raise TypeError("kernel_size must be a list or an integer.")  # invalid type

        # Prepare list of dense layer neuron counts
        if isinstance(dense_neurons, int):  # single integer provided
            dense_neurons = [dense_neurons]  # convert to list
        elif isinstance(dense_neurons, list):  # list provided
            assert all(isinstance(x, int) for x in dense_neurons)  # ensure all are ints
        else:
            raise TypeError("dense_neurons must be a list or an integer.")  # invalid type

        # Build convolutional layers list
        layers = []  # empty list to collect conv blocks
        in_channels = input_channels  # start with input image channels
        for i in range(num_conv_layers):  # iterate over conv layers
            out_channels = num_filters[i]  # number of filters for this layer
            k_size = kernel_size[i]  # kernel size for this layer
            # create a conv block and add to list
            layers.append(self.create_conv_block(in_channels, out_channels, k_size))
            in_channels = out_channels  # update input channels for next layer
        self.conv_layers = nn.Sequential(*layers)  # combine conv blocks sequentially

        # Compute flattened feature size dynamically
        with torch.no_grad():  # disable gradient tracking
            dummy_input = torch.zeros(1, input_channels, input_size, input_size)  # fake image tensor
            dummy_output = self.conv_layers(dummy_input)  # forward fake tensor
            # flatten and record feature count for dense layers
            self.flattened_size = dummy_output.view(1, -1).size(1)

        # Build fully connected (dense) layers list
        self.flatten = nn.Flatten()  # layer to flatten conv output
        fc_layers = []  # empty list to collect dense layers
        in_features = self.flattened_size  # input features for first dense layer
        for out_features in dense_neurons:  # iterate over defined dense neurons
            fc_layers.append(nn.Linear(in_features, out_features))  # add linear layer
            fc_layers.append(self.get_activation())  # add activation layer
            fc_layers.append(nn.Dropout(self.dropout_rate))  # add dropout layer
            in_features = out_features  # update feature count for next dense layer
        fc_layers.append(nn.Linear(in_features, num_classes))  # final output layer
        self.fc_layers = nn.Sequential(*fc_layers)  # combine dense layers sequentially

    def create_conv_block(self, in_channels, out_channels, kernel_size):  # helper to build a conv block
        padding = kernel_size // 2  # use same padding
        # convolutional layer
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding)]
        if self.batch_norm:  # if batch norm enabled
            layers.append(nn.BatchNorm2d(out_channels))  # add batch normalization
        layers.append(self.get_activation())  # add activation function
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))  # add max pooling
        layers.append(nn.Dropout(self.dropout_rate))  # add dropout
        return nn.Sequential(*layers)  # return the block as one module

    def get_activation(self):  # select activation based on name
        activations = {  # map of supported activations
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
            'silu': nn.SiLU(),
            'mish': nn.Mish()
        }
        # return selected activation or default to ReLU
        return activations.get(self.activation, nn.ReLU())

    def forward(self, x):  # define forward pass
        x = self.conv_layers(x)  # apply all convolutional layers
        x = self.flatten(x)  # flatten for dense layers
        x = self.fc_layers(x)  # apply all dense layers
        return x  # return output logits
