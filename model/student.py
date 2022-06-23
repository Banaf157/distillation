import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization

class Student(tf.keras.Model):
    def __init__(self):
        
        self.res_block_1 = ResBlock(16, 3, 3)




    def encoder(self, input):
        pass
    
    def call(self, input):
        code = self.encoder(input)
        
        return code

class ResBlock(tf.keras.Model):
    def __init__(self, nb_filter, filter_size,  input_channels, strides=(1,1)):
        self.conv_1 = Conv2D(filters=nb_filter, kernel_size=filter_size, padding='same')
        self.batch_norm_1 = BatchNormalization()
        self.conv_2 = Conv2D(filters=input_channels, kernel_size=filter_size, padding='same')
        self.batch_norm_2 = BatchNormalization()
        self.output_conv = Conv2D(filters=nb_filter, kernel_size=filter_size, padding='same', strides=strides)

    def call(self, input):
        x = self.conv_1(input)
        x = self.batch_norm_1(x)
        x = tf.keras.activations.relu(x)
        x = self.conv_2(input)
        x = self.batch_norm_2(x)
        x = tf.keras.activations.relu(x)

        return self.output_conv(x + input)     
