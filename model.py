"""Implementation of Depth Map Prediction from a Single Image using a Multi-Scale Deep Network in TensorFlow 2.1

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

This implements Eigen et al. Depth Map Prediction from a Single Image using a Multi-Scale Deep Network, arXiv: 1406.2283 (2014)
in TensorFlow 2.1.

    Typical usage example:

        test_tensor = tf.random.uniform(shape=(32, 304, 228, 3))
        coarse_model = CoarseNet(name="coarse_test", verbose=False)
        coarse_out_test = coarse_model(test_tensor)

        fine_model = FineNet(name="fine")
        y_pred_depth = combined_model(test_tensor, coarse_model, fine_model)

        train_losses, val_losses = train_coarse(coarse_model, train_dataset_batched,
                                                train_dataset, val_dataset, 100)

        train_fine_losses, val_fine_losses = train_fine(coarse_model, fine_model,
                                                        train_dataset_batched, train_dataset,
                                                        val_dataset, 100)

"""
from __future__ import absolute_import, division, print_function
import os
from skimage.transform import downscale_local_mean, resize
import numpy as np
import tensorflow as tf
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from mpl_toolkits.axes_grid1 import ImageGrid
tf.keras.backend.clear_session()
from tensorflow.keras import layers

class CoarseNet(tf.keras.Model):
    """Takes raw image input and converts it to a coarse depth map based on global features.
    
    This represents layers Course 1 through Course 7 in the paper. The coarse model has
    5 convolutional layers followed by two fully connected layers. The fully connected
    layers at the top of the network allow identifying features accross the entire image
    creating the global view of the network.
    
    Attributes:
            rate: A float that sets the dropout rate. Note that since TensorFlow 2.0, this
                represents the fraction of the input units to drop.
            verbose: A boolean. If True, the model will print out the shape of outputs from
                each of the layers
            name: A String that sets the name of the model. It is also used in the individual
                names of the layers. These names are used in the training loop and are hardcoded.
                If the name attribute is changed from the default, make sure to change the
                corresponding value in the training loop.
                
    """
    
    def __init__(self,
                 rate=0.2,
                 verbose=False,
                 name="Coarse"):
        """Inits CoarseNet with the dropout rate, whether to print layer shapes, and the name of the network.
        
        Since most of the layers are convolutional layers that don't depend on the shape of the inputs for
        initialization, they can be initialized here.
        """
        super(CoarseNet, self).__init__(name=name)
        self.drop_rate = rate
        self.verbose = verbose
        self.conv1 = layers.Conv2D(96, (11, 11), strides=(4,4), padding="valid", activation="relu", name=name + str(1))
        self.pool1 = layers.MaxPool2D()
        self.conv2 = layers.Conv2D(256, (5, 5), padding="same", activation="relu", name=name + str(2))
        self.pool2 = layers.MaxPool2D()
        self.conv3 = layers.Conv2D(384, (3, 3), padding="same", activation="relu", name=name + str(3))
        self.conv4 = layers.Conv2D(384, (3, 3), padding="same", activation="relu", name=name + str(4))
        self.conv5 = layers.Conv2D(256, (3, 3), strides=(2,2), padding="valid", activation="relu", name=name + str(5))
        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(4096, activation="relu", name=name + str(6))
        self.dropout = layers.Dropout(self.drop_rate)
    
    def build(self, input_shape):
        """Initializes the final dense layer whose output dimension depends on the input shape.
        
        The build method is used to calculate the final shape of the coarse model output to fit with Fine 1
        of the fine network. Note: The publication seems to have the wrong value for the width of the output
        on the KITTI dataset. It's listed as 27 and should be 41.
        
        Args:
            input_shape: Inputs to the network have shape (batches, n_H, n_W, channels).
        """
        n_H_out = int(math.floor((input_shape[1]-9)/2 + 1)/2)
        n_W_out = int(math.floor((input_shape[2]-9)/2 + 1)/2)
        n_total = n_W_out*n_H_out
        self.dense2 = layers.Dense(n_total, name=self.name + str(7))
        self.reshape = layers.Reshape((n_H_out, n_W_out))
        
    def call(self, inputs, training=None):
        """This method executes each of the layers whenever the model is run.
        
        The call method takes the inputs and sequentially passes them through each layer of the
        Coarse model architecture. The dropout layer is only used during training. It is skipped
        during inference.
        
        Args:
            inputs: A tensor with shape (batches, n_H, n_W, channels).
            training: A boolean. If true, dropout will be included before the final dense layer.
            
        Returns:
            coarse_7: A tensor with shape (batches, n_H_out, n_W_out) where n_H_out
                and n_W_out correspond the the shape of Fine 1 in the fine model.
        """
        x = self.conv1(inputs)
        coarse_1 = self.pool1(x)
        x = self.conv2(coarse_1)
        coarse_2 = self.pool2(x)
        coarse_3 = self.conv3(coarse_2)
        coarse_4 = self.conv4(coarse_3)
        coarse_5 = self.conv5(coarse_4)
        x = self.flatten(coarse_5)
        coarse_6 = self.dense1(x)
        if training:
            x = self.dropout(coarse_6)
        else:
            x = coarse_6
            build_d = self.dropout(coarse_6)
        x = self.dense2(x)
        coarse_7 = self.reshape(x)
        
        if(self.verbose):
            print("Layer shapes:\nCoarse 1: " + str(coarse_1.shape) + "\nCoarse 2: " + str(coarse_2.shape) + ", Coarse 3: " +
                  str(coarse_3.shape) + ", Coarse 4: " + str(coarse_4.shape) + "\nCoarse 5: " + str(coarse_5.shape) +
                  "\nCoarse 6: " + str(coarse_6.shape) + "\nCoarse 7: " + str(coarse_7.shape))
        
        return coarse_7


class FineNet(tf.keras.Model):
    """Takes raw image input and converts it to a coarse depth map based on fine + global features.
    
    This network is focused on incorporating fine features into the depth map prediction. It uses
    three convolutional layers for this. Additionally, it incorporates a global perspective by
    concatenating the output of the Course Model into the output of Fine 1.
    
    Attributes:
        name: A String that sets the name of the model. This is used to name the individual layers
            and is used in the training loop. If the name is changed from the default value, the names
            need to be changed in the training loop. They are hard coded in the training loop for
            assigning layer-wise learning rates
    """
    
    def __init__(self,
                 name="Fine"):
        """Inits FineNet with the name of the model."""
        super(FineNet, self).__init__(name=name)
        self.conv1 = layers.Conv2D(63, (9, 9), strides=(2,2), padding="valid", activation="relu", name = name + str(1))
        self.pool1 = layers.MaxPool2D()
        self.conc = layers.Concatenate()
        self.conv2 = layers.Conv2D(64, (5, 5), padding="same", activation="relu", name=name + str(3))
        self.conv3 = layers.Conv2D(1, (5, 5), padding="same", activation="linear", name=name + str(4))
        
    def build(self, input_shape):
        """Initializes the reshape layer for changing the shape of the outpute from the Coarse Model.
        
        The coarse model output is of shape (batches, n_H_coarse, n_W_coarse). To be concatenated with
        the Fine 1, it needs to be of shape (batches, n_H_coarse, n_W_coarse, 1).
        
        Args:
            input_shape: A dict with keys "orig" and "coarse". input_shape["coarse"] contains the dimensions
                of the coarse model output and has shape (batches, n_H_coarse, n_W_coarse).
        """
        coarse_shape = input_shape['coarse']
        n_H_coarse = coarse_shape[1]
        n_W_coarse = coarse_shape[2]
        self.reshape = layers.Reshape((n_H_coarse, n_W_coarse, 1))
        
        
    def call(self, inputs):
        """This method executes each of the layers whenever the model is run.
        
        The call method takes the inputs and sequentially passes them through each layer of the
        Fine model architecture to create a depth map based on fine and global features.
        
        Args:
            inputs: A dict with keys "orig" and "coarse". inputs["orig"] is a tensor of shape
                (batches, n_H, n_W, channels). inputs["coarse"] is a tensor of shape
                (batches, n_H_coarse, n_W_coarse).
            
        Returns:
            fine_4: A tensor with shape (batches, n_H_out, n_W_out, 1) where n_H_out
                and n_W_out are the dimensions of the output depth map from the network.
        """
        orig_input = inputs["orig"]
        coarse_input = inputs["coarse"]
        x = self.conv1(orig_input)
        fine_1 = self.pool1(x)
        coarse_input = self.reshape(coarse_input)
        fine_2 = self.conc([fine_1, coarse_input])
        fine_3 = self.conv2(fine_2)
        fine_4 = self.conv3(fine_3)
        
        return fine_4


def combined_model(input_image, coarse_model, fine_model,
                   training_coarse=False, training_fine=False):
    """Combines the coarse and fine models together.
    
    Args:
        input_image: A tensor of shape (batches, n_H, n_W, 3).
        coarse_model: A keras Model that implements the coarse model architecture.
        fine_model: A keras Model that implements the fine model architecture.
        training_coarse: A boolean. If True, sets training equal to True in the Coarse Model.
            This will have the effect of implementing dropout before the final dense layer.
        training_fine: A boolean. If True, sets training equal to True in the Fine Model. Currently,
            this will have no effect but could be used if something like dropout were added to the fine
            model in the future.
            
    Returns:
        y_pred: A tensor representing the predicted depth map. It has shape (batches, n_H_out, n_W_out, 1)
    """
    coarse_out = coarse_model(input_image, training = training_coarse)
    fine_in = {
        "orig": input_image,
        "coarse": coarse_out,
    }
    y_pred = fine_model(fine_in, training = training_fine)
    
    return y_pred


def scale_invariant_MSE(y_true, y_pred, mask):
    """This calculates the scale invariant error given ground truths, predictions, and masks.
    
    This function implements the scale invariant MSE defined in Eigen et. al. it is essentially
    the sum of the squared difference between all of the distances of the ground truth values
    and the distances of all of the corresponding prediction values. Because it is the difference
    of differences, it is invariant to the mean of the ground truths and predictions. Eigen et al.
    chose a scale invariant error because they noticed that a change in the mean log depth
    impacted the RMSE to a large degree. With this loss function, chaning the scale does not
    change the error.
    
    Both n_H_pred and n_W_pred, the outputs of the fine model, are smaller than n_H and n_W, respectively.
    Because of the difference in the size of the tensors, to compare them, y_pred is resized to be the
    same size as y_true.
    
    The mask is used so that errors where the depth map didn't have a reading are not counted into
    the loss function.
    
    Args:
        y_true: A tensor representing the ground truth depths. It has shape (batches, n_H, n_W).
        y_pred: A tensor representing the predicted depths. It has shape (batches, n_H_pred, n_W_pred, 1).
        mask: A tensor masking the parts of loss where y_true didn't have a reading.
            It has shape (batches, n_H, n_W)
            
    Returns:
        batch_loss: the sum of scale invariant losses for the batch divided by the number of samples in
            the batch.
    """
    n_H = y_true.shape[1]
    n_W = y_true.shape[2]
    n_H_pred = y_pred.shape[1]
    n_W_pred = y_pred.shape[2]
    y_true = layers.Reshape((n_H, n_W, 1))(y_true)
    y_pred = layers.Reshape((n_H_pred, n_W_pred, 1))(y_pred)
    
    log_pred = tf.image.resize(y_pred, (n_H, n_W), method="nearest")
    log_true = tf.math.log(y_true + 1e-30) #adding 1e-30 prevents -inf
    mask = tf.dtypes.cast(mask, tf.float32)
    mask = layers.Reshape((n_H, n_W, 1))(mask)
    log_pred = log_pred*mask
    log_true = log_true*mask
        
    d = tf.math.subtract(log_true, log_pred)
    n = tf.math.reduce_sum(mask, [1, 2, 3])
    d2 = d**2
    D_sum1 = tf.math.reduce_sum(d2, [1, 2, 3])
    D_sum2 = tf.math.reduce_sum(d, [1, 2, 3])**2
    lamb = 0.5
    D = (1./n)*D_sum1 - (lamb/(n*n))*D_sum2
    batch_loss = tf.math.reduce_sum(D)
    batch_loss = batch_loss/D.shape[0]
    
    return batch_loss


def train_coarse(model, train_dataset_batched, train_dataset, val_dataset, epochs,
                 lr12345=0.001, lr67=0.1):
    """This is the training loop for the coarse model.
    
    This training loop uses layer-wise training rates to train the course model. Specifically,
    layers 1 through 5 have a default training rate of 0.001 while layers 6 and 7 have a default
    training rate of 0.1. These default values do have better training performance than when using
    a constant training rate.
    
    Note: The layer-wise learning rates depend on the layer names. These are based on the default
    values set in the CoarseNet and FineNet classes.
    
    Args:
        model: A keras Model that defines the coarse model.
        train_dataset_batched: A tf.data.Dataset that contains batched training data with tensors
            of shape (batches, n_H, n_W, channels).
        train_dataset: A tensor containing the entire training dataset with shape
            (training_size, n_H, n_W, channels). This is used for calculating the overall training set
            loss after each epoch.
        val_dataset: A tensor containing the entire validation dataset with shape
            (validation_size, n_H, n_W, channels). This is used for calculating the overall validation set
            loss after each epoch.
        epochs:
            An int that sets how many times the training loop should itterate through all of the data.
        lr12345: A float that sets the learning rate for layers 1 through 5.
        lr67: A float that sets the learning rate for layers 6 through 7.
        
    Returns:
        train_losses: A list of the training losses after each epoch
        val_losses: A list of the validation losses after each epoch
    """
    train_losses = []
    val_losses = []
    
    optimizer_c12345 = tf.keras.optimizers.SGD(learning_rate=lr12345, momentum=0.9,
                                               nesterov=False, name='SGD1')
    optimizer_c67 = tf.keras.optimizers.SGD(learning_rate=lr67, momentum=0.9,
                                            nesterov=False, name='SGD2')
    
    for epoch in range(epochs):        
        for step, (x_batch_train, y_batch_train, m_batch_train) in enumerate(train_dataset_batched):
            with tf.GradientTape() as tape:
                y_pred = model(x_batch_train, training=True)
                
                loss_value = scale_invariant_MSE(y_batch_train, y_pred, m_batch_train)
                
            grads = tape.gradient(loss_value, model.trainable_weights)
            
            trainable_weights = model.trainable_weights            
            coarse12345_weights = []
            coarse12345_grads = []
            coarse67_weights = []
            coarse67_grads = []
            
            for i, weight in enumerate(trainable_weights):
                w_name = weight.name
                grad = grads[i]
                
                if("Coarse6" in w_name or "Coarse7" in w_name):
                    coarse67_weights.append(weight)
                    coarse67_grads.append(grad)
                else:
                    coarse12345_weights.append(weight)
                    coarse12345_grads.append(grad)
            
            optimizer_c12345.apply_gradients(zip(coarse12345_grads, coarse12345_weights))
            optimizer_c67.apply_gradients(zip(coarse67_grads, coarse67_weights))
            
        x_train, y_train, m_train = train_dataset
        y_pred_train = model(x_train, training=False)
        train_loss = scale_invariant_MSE(y_train, y_pred_train, m_train).numpy()
        train_losses.append(train_loss)
        
        x_val, y_val, m_val = val_dataset
        y_pred_val = model(x_val, training=False)
        val_loss = scale_invariant_MSE(y_val, y_pred_val, m_val).numpy()
        val_losses.append(val_loss)
        
        if epoch % 5 == 0:
            print("After epoch " + str(epoch) + ", Training Loss: "
                  + str(train_loss) + ", Validation Loss: " + str(val_loss))
            
    return train_losses, val_losses


def train_fine(coarse_model, fine_model, train_dataset_batched, train_dataset, val_dataset, epochs,
               lr14 = 0.001, lr3 = 0.01):
    """This is the training loop for the fine model.
    
    The coarse model is fixed while the loop trains the fine model.
    
    This training loop uses layer-wise training rates to train the fine model. Specifically,
    layers 1 and 4 have a default training rate of 0.001 while layer 3 has a default training
    rate of 0.1.
    
    Note: The layer-wise learning rates depend on the layer names. These are based on the default
    values set in the CoarseNet and FineNet classes.
    
    Args:
        coarse_model: A keras Model that defines the coarse model.
        fine_model: A keras Model that defines the fine model.
        train_dataset_batched: A tf.data.Dataset that contains batched training data with tensors
            of shape (batches, n_H, n_W, channels).
        train_dataset: A tensor containing the entire training dataset with shape
            (training_size, n_H, n_W, channels). This is used for calculating the overall training set
            loss after each epoch.
        val_dataset: A tensor containing the entire validation dataset with shape
            (validation_size, n_H, n_W, channels). This is used for calculating the overall validation set
            loss after each epoch.
        epochs:
            An int that sets how many times the training loop should itterate through all of the data.
        lr14: A float that sets the learning rate for layers 1 and 4.
        lr3: A float that sets the learning rate for layer 3.
        
    Returns:
        train_losses: A list of the training losses after each epoch
        val_losses: A list of the validation losses after each epoch
    """
    train_losses = []
    val_losses = []
    
    optimizer_f14 = tf.keras.optimizers.SGD(learning_rate=lr14, momentum=0.9,
                                            nesterov=False, name='SGD3')
    optimizer_f3 = tf.keras.optimizers.SGD(learning_rate=lr3, momentum=0.9,
                                           nesterov=False, name='SGD4')
    
    for epoch in range(epochs):        
        for step, (x_batch_train, y_batch_train, m_batch_train) in enumerate(train_dataset_batched):
            with tf.GradientTape() as tape:
                
                y_pred = combined_model(x_batch_train, coarse_model, fine_model, training_fine=True)                
                loss_value = scale_invariant_MSE(y_batch_train, y_pred, m_batch_train)
                
            grads = tape.gradient(loss_value, fine_model.trainable_weights)
            
            trainable_weights = fine_model.trainable_weights            
            fine14_weights = []
            fine14_grads = []
            fine3_weights = []
            fine3_grads = []
            
            for i, weight in enumerate(trainable_weights):
                w_name = weight.name
                grad = grads[i]
                
                if("Fine3" in w_name):
                    fine3_weights.append(weight)
                    fine3_grads.append(grad)
                else:
                    fine14_weights.append(weight)
                    fine14_grads.append(grad)
            
            #the two layers below are where I implement layer-wise learning rates
            optimizer_f14.apply_gradients(zip(fine14_grads, fine14_weights))
            optimizer_f3.apply_gradients(zip(fine3_grads, fine3_weights))
            
        x_train, y_train, m_train = train_dataset
        y_pred_train = combined_model(x_train, coarse_model, fine_model)
        train_loss = scale_invariant_MSE(y_train, y_pred_train, m_train).numpy()
        train_losses.append(train_loss)
        
        x_val, y_val, m_val = val_dataset
        y_pred_val = combined_model(x_val, coarse_model, fine_model)
        val_loss = scale_invariant_MSE(y_val, y_pred_val, m_val).numpy()
        val_losses.append(val_loss)
        
        if epoch % 5 == 0:
            print("After epoch " + str(epoch) + ", Training Loss: "
                  + str(train_loss) + ", Validation Loss: " + str(val_loss))
            
    return train_losses, val_losses