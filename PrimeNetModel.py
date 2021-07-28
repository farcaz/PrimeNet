import tensorflow as tf
class Capsule(tf.keras.Model):
	def __init__(self,filters_reduce,filters,scale):
		super(Capsule, self).__init__()

		self.filters_reduce=filters_reduce
		self.filters=filters
		self.scale=scale

		self.kernel_init = tf.keras.initializers.glorot_uniform()
		self.bias_init = tf.keras.initializers.Constant(value=0.2)

		#self.conv_1x1 = tf.keras.layers.Conv2D(64, (1, 1), padding='same', activation='relu', kernel_initializer=self.kernel_init, bias_initializer=self.bias_init)

		self.conv_1 = tf.keras.layers.Conv2D(self.filters_reduce, (1, 1), padding='same', kernel_initializer=self.kernel_init, bias_initializer=self.bias_init)
		self.conv_2 = tf.keras.layers.Conv2D(self.filters, (scale, scale), padding='same', kernel_initializer=self.kernel_init, bias_initializer=self.bias_init)

		self.batch_norm_1 = tf.keras.layers.BatchNormalization()
		self.leakyrelu_1 = tf.keras.layers.LeakyReLU()
		self.flatten=tf.keras.layers.Flatten()
		self.final_dense = tf.keras.layers.Dense(28*28*1, activation='tanh')
		self.reshape = tf.keras.layers.Reshape((28, 28, 1))

	def call(self,input,training=True):
		#x=self.conv_1x1(input)
		x=self.conv_1(input)
		Conv2D=self.conv_2(x)
		x=self.batch_norm_1(Conv2D,training)
		x=self.leakyrelu_1(x)
		Flatx=self.flatten(x)

		Densex=self.final_dense(Flatx)
		Outx=self.reshape(Densex)
		return x,Outx

class PrimeNet(tf.keras.Model):
	def __init__(self):
		super(PrimeNet, self).__init__()
		self.kernel_init = tf.keras.initializers.glorot_uniform()
		self.bias_init = tf.keras.initializers.Constant(value=0.2)
		
		self.pool_proj1=tf.keras.layers.MaxPool2D((3, 3), strides=(1, 1), padding='same')
		self.pool_proj2=tf.keras.layers.Conv2D(64, (1, 1), padding='same', activation='relu',kernel_initializer=self.kernel_init, bias_initializer=self.bias_init)

		self.pool1=tf.keras.layers.MaxPool2D((2,2))
		self.bn=tf.keras.layers.BatchNormalization()
		self.conv1=tf.keras.layers.Conv2D(128,(3,3),activation='relu',kernel_initializer=self.kernel_init, bias_initializer=self.bias_init)
		self.conv2=tf.keras.layers.Conv2D(256,(3,3),activation='relu',kernel_initializer=self.kernel_init, bias_initializer=self.bias_init)
		self.pool2=tf.keras.layers.MaxPool2D((2,2))
		self.globalPool=tf.keras.layers.GlobalAveragePooling2D()
		self.flatten=tf.keras.layers.Flatten(name="flatten")
		self.dense=tf.keras.layers.Dense(10,activation='softmax', name='output')

	def call(self, inputs, training=True):
		
		proj=self.pool_proj1(inputs)
		proj=self.pool_proj2(proj)
		x=tf.keras.layers.Concatenate([proj,inputs])

		x=self.bn(inputs,training)
		x=self.pool1(x)
		
		x=self.conv1(x)
		x=self.pool2(x)
		return self.globalPool(x)
		#x=self.flatten(x)
		#return self.dense(x)


if __name__ == "__main__":
    pass


