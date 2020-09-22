# Preparind the hourses and zebra dataset
'''
The "A" refers to house and "B" refers to zebra
Below we will load all photographs from the train and test folders and create an array of images for 
category A and another for category B
Both arrays are then saved to a new file in compressed NumPy array formet
'''
# Preparing the horses and zebra dataset
import subprocess
import sys
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip","install",package])
install("git+https://www.github.com/keras-team/keras-contrib.git")
# !pip install git+https://www.github.com/keras-team/keras-contrib.git
from os import listdir
from numpy import asarray
from numpy import vstack
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from numpy import savez_compressed

# Load all images in a directory into memory
def load_images(path, size=(256,256)):
    data_list = list()
    # enumerate filenames in directory, assume all are images
    for filename in listdir(path):
        # load and resize the image
        pixels = load_img(path + filename, target_size=size)
        # convert to numpy array
        pixels = img_to_array(pixels)
        # sote
        data_list.append(pixels)
    return asarray(data_list)

# Dataset path
path = '../input/cyclegan/horse2zebra/horse2zebra/'
# load dataset A
dataA1 = load_images(path + 'trainA/')
dataAB = load_images(path + 'testA/')
dataA = vstack((dataA1, dataAB))
print('Loaded dataA: ', dataA.shape)
# load dtaset B
dataB1 = load_images(path + 'trainB/')
dataB2 = load_images(path + 'testB/')
dataB = vstack((dataB1, dataB2))
print('Loaded dataB: ', dataB.shape)
# save as compressed numpy array
filename = 'horse2zebra_256.npz'
savez_compressed(filename, dataA, dataB)
print('Saved dataseet:', filename)



# =======================================================================================================================




# example of training a cyclegan on the horse2zebra dataset
from random import random
from numpy import load
from numpy import zeros
from numpy import ones
from numpy import asarray
from numpy.random import randint
from keras.optimizers import Adam
from keras.initializers import RandomNormal
from keras.models import Model
from keras.models import Input
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import Activation
from keras.layers import Concatenate
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from matplotlib import pyplot
import tensorflow as tf

# detect and init the TPU
tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
tf.config.experimental_connect_to_cluster(tpu)
tf.tpu.experimental.initialize_tpu_system(tpu)

# instantiate a distribution strategy
tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)



'''
The discriminator is deep CNN that performs image classification. Two disciminator models are used, one
for Domain-A and one for Domain-B. 
The disciminator design is based on the effective receptive field of the
model, which definew the relationship between on output of the model to the number of pixels in the input image.
This is called a PatchGAN model and is carefully designed so that each output prediction of the model maps to a 70*70
square or patch of the input image. The benefit of this approach is that the same model can be applied to input images
of different size, ex larger or smaller than 256*256 pixels.

The output of the model depends on the size of the input image but may be one value or a square acivation map of values.
Each value is a probability for the likelihood that a patch in the input image is real. These values can be averaged to 
give an overlall likelihood or classification score if needed.

A pattern of convolutional-batchNorm-LeakyReLU layers is used in the model, which is common to deep convolutional 
discriminator models. Unilike other models, the CycleGan discriminator uses InstanceNormalization instead of BatchNormalization.
It is a very simple type of normalization and involves standardizing(eg Scaling to a standard Gaussian) the values on
each output feature map, rather than across features in a batch.
'''
# define the discriminator model
def define_discriminator(image_shape):
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# source image input
	in_image = Input(shape=image_shape)
	# C64	
	d = Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(in_image)
	d = LeakyReLU(alpha=0.2)(d)
	# C128
	d = Conv2D(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
	d = InstanceNormalization(axis=-1)(d)
	# axis argument is set to -1 to ensure that features are normalized per feature map.
	d = LeakyReLU(alpha=0.2)(d)
	# C256
	d = Conv2D(256, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
	d = InstanceNormalization(axis=-1)(d)
	d = LeakyReLU(alpha=0.2)(d)
	# C512
	d = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
	d = InstanceNormalization(axis=-1)(d)
	d = LeakyReLU(alpha=0.2)(d)
	# second last output layer
	d = Conv2D(512, (4,4), padding='same', kernel_initializer=init)(d)
	d = InstanceNormalization(axis=-1)(d)
	d = LeakyReLU(alpha=0.2)(d)
	# patch output
	patch_out = Conv2D(1, (4,4), padding='same', kernel_initializer=init)(d)
	# define model
	model = Model(in_image, patch_out)
	# compile model
	model.compile(loss='mse', optimizer=Adam(lr=0.0002, beta_1=0.5), loss_weights=[0.5])
	return model



'''
The generator is an encoder-decoder model architecute. The model takes a source image and generates a target image.
It does it by first downsampling or encoding the input image down to a bottleneck layer, then interpreting the encoding 
with a number of ResNet layers that use skit connections, followed by a series of layers that upsample or decode the 
representation to the size of the output image.

First we need a function to define the ResNet blocks. These are blocks comprised of two 3x3 CNN layers where the input to
the block is concatenated to the output of the block, cannel-wise.

Following fuction creates two convolution-InstanceNorm blocks with 3x3 filters and 1x1 stride and without a Relu Activation
after the second block.
'''

# generator a resnet block
def resnet_block(n_filters, input_layer):
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# first layer convolutional layer
	g = Conv2D(n_filters, (3,3), padding='same', kernel_initializer=init)(input_layer)
	g = InstanceNormalization(axis=-1)(g)
	g = Activation('relu')(g)
	# second convolutional layer
	g = Conv2D(n_filters, (3,3), padding='same', kernel_initializer=init)(g)
	g = InstanceNormalization(axis=-1)(g)
	# concatenate merge channel-wise with input layer
	g = Concatenate()([g, input_layer])
	return g

'''
Now, we can define a fuction that will create the 9-resent block version for 256x256 input images. 

The model outputs pixel values with the shape as the input and pixel values are in the rang[-1,1], typlica for GAN
generator models.
'''

# define the standalone generator model
def define_generator(image_shape, n_resnet=9):
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# image input
	in_image = Input(shape=image_shape)
	# c7s1-64
	g = Conv2D(64, (7,7), padding='same', kernel_initializer=init)(in_image)
	g = InstanceNormalization(axis=-1)(g)
	g = Activation('relu')(g)
	# d128
	g = Conv2D(128, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(g)
	g = InstanceNormalization(axis=-1)(g)
	g = Activation('relu')(g)
	# d256
	g = Conv2D(256, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(g)
	g = InstanceNormalization(axis=-1)(g)
	g = Activation('relu')(g)
	# R256
	for _ in range(n_resnet):
		g = resnet_block(256, g)
	# u128
	g = Conv2DTranspose(128, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(g)
	g = InstanceNormalization(axis=-1)(g)
	g = Activation('relu')(g)
	# u64
	g = Conv2DTranspose(64, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(g)
	g = InstanceNormalization(axis=-1)(g)
	g = Activation('relu')(g)
	# c7s1-3
	g = Conv2D(3, (7,7), padding='same', kernel_initializer=init)(g)
	g = InstanceNormalization(axis=-1)(g)
	out_image = Activation('tanh')(g)
	# define model
	model = Model(in_image, out_image)
	return model

'''
The disciminator models are trained directly on real and generated images, whereas the genrator models are not.

Instread, the genrator models are trained via their related discriminator models. Specifically, they are updated to 
minimize the loss predicted by the discriminator for generated images marked as 'real', called adverssarial loss. As such
they are encouraged to generate images that better fit into the target domain.

The genrator models are alos updated based on how effective they are at the regeneration of a source image when used with
the other generator model, called cycle loss. Finally, a generator model is expected to output an imgage without translation
when provided an example from the target domain, called identity loss.

Altogether, each generator model is optimized via the combination of four outputs with four loss fuctions:
	Adversarial loss(L2 or mean squared error)
	Identity loss(L1 or mean absolute error)
	Forward cycle loss(L1 or mean absolute error)
	Backward cycle loss(L1 or mean absolute error)

This can be achieved by defining a composite model used to train each generator model that is responsible for only
updating the weights of that generator model, although it is requited to share the weights with the related disciminator
model and the other generator model.

This is implemented int the define_composite_model() fuction below that takes a defined generator model(g_model_1) as 
well as the defined discriminator model for the generator models output(d_model) and the other generator model(g_model_2).
The weights of the other models are marked as not trainable as we are only interested in updating the first generator model,
i.e the focus of this comosite mode.

The disciminator is connected to the output of the genrator in order to classify generated images as real or fake. A Second
input of the composite model is defined as an image from the target domain (instead of the source domain), which the 
generator is expected to output without translation for the identity mapping. Next, forward cycle loss involves connecting the 
output of the generator to the other generator, which will reconstruct the source image. Finally, the backward cycle loss
involves the image from the target domain used for the identity mapping that is also passed through the other generaor 
whose output is connected to out main generator as input and outputs a reconstructed version of that image from the target domain.

Summery: a composite model has two inputs for the real photos from Domain-A and Domain-B, and four outputs for the discriminator
output, identity generated image, forward cycle generated image, and bakward cycle generated image.

Only the weights of the first or main generator model are updated for the composite model and this is done via the weighted sum
of all loss functions. The cycle loss is given more weight (10-times) than the adversarial loss, and the identity loss
is always used with a weighting half of the cycle loss(5-times).
'''

# define a composite model for updating generators by adversarial and cycle loss
def define_composite_model(g_model_1, d_model, g_model_2, image_shape):
	# ensure the model we're updating is trainable
	g_model_1.trainable = True
	# mark discriminator as not trainable
	d_model.trainable = False
	# mark other generator model as not trainable
	g_model_2.trainable = False
	# discriminator element
	input_gen = Input(shape=image_shape)
	gen1_out = g_model_1(input_gen)
	output_d = d_model(gen1_out)
	# identity element
	input_id = Input(shape=image_shape)
	output_id = g_model_1(input_id)
	# forward cycle
	output_f = g_model_2(gen1_out)
	# backward cycle
	gen2_out = g_model_2(input_id)
	output_b = g_model_1(gen2_out)
	# define model graph
	with tpu_strategy.scope():
		model = Model([input_gen, input_id], [output_d, output_id, output_f, output_b])
	# define optimization algorithm configuration
		opt = Adam(lr=0.0002, beta_1=0.5)
	# compile model with weighting of least squares loss and L1 loss
		model.compile(loss=['mse', 'mae', 'mae', 'mae'], loss_weights=[1, 5, 10, 10], optimizer=opt)
	return model

'''
We need to create a composite mode for each generator mode, e.g. the Generator-A for zebra to house translation, and the
Generator-B for horse to zebra translation.
'''

'''
We can load our paired images dataset in compressed NumPy array format. This will return a list of two NumPy arrays:
the first for source images and the second for corresponding target images.
'''


# load and prepare training images
def load_real_samples(filename):
	# load the dataset
	data = load(filename)
	# unpack arrays
	X1, X2 = data['arr_0'], data['arr_1']
	# scale from [0,255] to [-1,1]
	X1 = (X1 - 127.5) / 127.5
	X2 = (X2 - 127.5) / 127.5
	return [X1, X2]

'''
Each training iteration we will requtire a sample of real images from each domain as input to the discriminator and
composite generator models. This can be achieved by selecting a random batch of samples.

The generate_real_samples() function implements this, taking a NumPy array for a domain as input and returning the
requested number of randomly selected images, as well as the target for the PatchGAN discriminator model indicating the
images are real(target=1.0). As such, the shape of the PatchGAN output is also provided, which in the case of 256x256
images will be 16, or a 16x16x1 activation map, defined by the patch_shape function argument.
'''

# select a batch of random samples, returns images and target
def generate_real_samples(dataset, n_samples, patch_shape):
	# choose random instances
	ix = randint(0, dataset.shape[0], n_samples)
	# retrieve selected images
	X = dataset[ix]
	# generate 'real' class labels (1)
	y = ones((n_samples, patch_shape, patch_shape, 1))
	return X, y

'''
Similarly, a sample of generated images is required to update each discriminator model in each training iteration.

The generate_fake_samples() function below generates this sample given a generator model and the sample of real images 
from the source domain. Again, target values for each generated image are provided with the correct shape of the PatchGAN,
indicating that they are fake or generted (target=0.0)
'''

# generate a batch of images, returns images and targets
def generate_fake_samples(g_model, dataset, patch_shape):
	# generate fake instance
	X = g_model.predict(dataset)
	# create 'fake' class labels (0)
	y = zeros((len(X), patch_shape, patch_shape, 1))
	return X, y

'''
Typically, GAN model do not converge; instead, an equibirium is found between the generator and discriminator models.
As such, we cannot easily judge whether training should stop. Therefore, we can save the model and use it to generate 
sample image-to-image translaations periodically during training, such as every one or five training epochs.

We can then review the generated images at the end of training and use the image quality to choose a final model. 
The save_models() fuction bellow will save each generator model to the current directory in H5 format, including the 
training iteration number in the filename.
'''

# save the generator models to file
def save_models(step, g_model_AtoB, g_model_BtoA):
	# save the first generator model
	filename1 = 'g_model_AtoB_%06d.h5' % (step+1)
	g_model_AtoB.save(filename1)
	# save the second generator model
	filename2 = 'g_model_BtoA_%06d.h5' % (step+1)
	g_model_BtoA.save(filename2)
	print('>Saved: %s and %s' % (filename1, filename2))

'''
The summarize_performance() function below uses a given generator model to generate translated version of a few randomly
selected source photographs and saaves the plot to file.

The source image are plotted on the first row and the generated images are plotted on the second row.
'''

# generate samples and save as a plot and save the model
def summarize_performance(step, g_model, trainX, name, n_samples=5):
	# select a sample of input images
	X_in, _ = generate_real_samples(trainX, n_samples, 0)
	# generate translated images
	X_out, _ = generate_fake_samples(g_model, X_in, 0)
	# scale all pixels from [-1,1] to [0,1]
	X_in = (X_in + 1) / 2.0
	X_out = (X_out + 1) / 2.0
	# plot real images
	for i in range(n_samples):
		pyplot.subplot(2, n_samples, 1 + i)
		pyplot.axis('off')
		pyplot.imshow(X_in[i])
	# plot translated image
	for i in range(n_samples):
		pyplot.subplot(2, n_samples, 1 + n_samples + i)
		pyplot.axis('off')
		pyplot.imshow(X_out[i])
	# save plot to file
	filename1 = '%s_generated_plot_%06d.png' % (name, (step+1))
	pyplot.savefig(filename1)
	pyplot.close()

'''
The discriminator models are updated directly on real and generated images, although in an effort to further manage 
how quickly the discriminator models learn, a pool of fake images is maintained.

The paper defines an image pool of 50 generated images for each discriminator model that is fist populated and 
probabilistically either adds new images to the pool by replacing and existing image or uses a generated image directly.
we can implement this as a Python list of images for each discriminator and use the updatate_image_pool() function below to 
maintain each pool list.
'''

# update image pool for fake images
def update_image_pool(pool, images, max_size=50):
	selected = list()
	for image in images:
		if len(pool) < max_size:
			# stock the pool
			pool.append(image)
			selected.append(image)
		elif random() < 0.5:
			# use image, but don't add it to the pool
			selected.append(image)
		else:
			# replace an existing image and use replaced image
			ix = randint(0, len(pool))
			selected.append(pool[ix])
			pool[ix] = image
	return asarray(selected)


'''
Now we can define the training of each of the generator models.

The train() function below takes all six models(two discriminator, two generator, and two composite models) as 
arguments along with the dataset and trains the models.

The batch size is fixed at one image to match the description in the paper and the models are fit for 100 epochs. 
Give that the houses dataset has 1187 images, one epoch is defined as 1187 batches and the same number of training 
iterations. Images are generated using both generators each epoch and models are saved every five epochs or (1187*5)
=5935 training iterations.

The order of model updates is implemented to match the official Torch implementation. First, a batch of real images 
from each domain is selected, then a batch of fake images for each domain is generated. The fake images are then used 
to update each discriminator's fake image pool.

Next, the Generator-A model(zebras to horses) is updated via the composite model, followed by the Discriminator-A model(horses).
Then the Generator-B (houses to zebras) composite model and Discriminator-B(zebras) model are updated.

Loss for each of the updated models is then reported at the end of the training iteration. Importantly, only the 
weighted average loss used to update each generator is reported.
'''


# train cyclegan models
def train(d_model_A, d_model_B, g_model_AtoB, g_model_BtoA, c_model_AtoB, c_model_BtoA, dataset):
	# define properties of the training run
	n_epochs, n_batch, = 100, 1
	# determine the output square shape of the discriminator
	n_patch = d_model_A.output_shape[1]
	# unpack dataset
	trainA, trainB = dataset
	# prepare image pool for fakes
	poolA, poolB = list(), list()
	# calculate the number of batches per training epoch
	bat_per_epo = int(len(trainA) / n_batch)
	# calculate the number of training iterations
	n_steps = bat_per_epo * n_epochs
	# manually enumerate epochs
	for i in range(n_steps):
		# select a batch of real samples
		X_realA, y_realA = generate_real_samples(trainA, n_batch, n_patch)
		X_realB, y_realB = generate_real_samples(trainB, n_batch, n_patch)
		# generate a batch of fake samples
		X_fakeA, y_fakeA = generate_fake_samples(g_model_BtoA, X_realB, n_patch)
		X_fakeB, y_fakeB = generate_fake_samples(g_model_AtoB, X_realA, n_patch)
		# update fakes from pool
		X_fakeA = update_image_pool(poolA, X_fakeA)
		X_fakeB = update_image_pool(poolB, X_fakeB)
		# update generator B->A via adversarial and cycle loss
		g_loss2, _, _, _, _  = c_model_BtoA.train_on_batch([X_realB, X_realA], [y_realA, X_realA, X_realB, X_realA])
		# update discriminator for A -> [real/fake]
		dA_loss1 = d_model_A.train_on_batch(X_realA, y_realA)
		dA_loss2 = d_model_A.train_on_batch(X_fakeA, y_fakeA)
		# update generator A->B via adversarial and cycle loss
		g_loss1, _, _, _, _ = c_model_AtoB.train_on_batch([X_realA, X_realB], [y_realB, X_realB, X_realA, X_realB])
		# update discriminator for B -> [real/fake]
		dB_loss1 = d_model_B.train_on_batch(X_realB, y_realB)
		dB_loss2 = d_model_B.train_on_batch(X_fakeB, y_fakeB)
		# summarize performance
		print('>%d, dA[%.3f,%.3f] dB[%.3f,%.3f] g[%.3f,%.3f]' % (i+1, dA_loss1,dA_loss2, dB_loss1,dB_loss2, g_loss1,g_loss2))
		# evaluate the model performance every so often
		if (i+1) % (bat_per_epo * 1) == 0:
			# plot A->B translation
			summarize_performance(i, g_model_AtoB, trainA, 'AtoB')
			# plot B->A translation
			summarize_performance(i, g_model_BtoA, trainB, 'BtoA')
		if (i+1) % (bat_per_epo * 5) == 0:
			# save the models
			save_models(i, g_model_AtoB, g_model_BtoA)

'''
The loss is reported at each training iteration, including the Discriminator-A loss on real and fake examples(dA),
Discriminator-B loss on real and fake examples(dB), and Generaotr-AtoB and Generator-BtoA loss, each of which is a 
weighted average of adversarial, identity, forward, and backward cycle loss(g).
'''

# load image data
dataset = load_real_samples('horse2zebra_256.npz')
print('Loaded', dataset[0].shape, dataset[1].shape)
# define input shape based on the loaded dataset
image_shape = dataset[0].shape[1:]
# generator: A -> B
g_model_AtoB = define_generator(image_shape)
# generator: B -> A
g_model_BtoA = define_generator(image_shape)
# discriminator: A -> [real/fake]
d_model_A = define_discriminator(image_shape)
# discriminator: B -> [real/fake]
d_model_B = define_discriminator(image_shape)
# composite: A -> B -> [real/fake, A]
c_model_AtoB = define_composite_model(g_model_AtoB, d_model_B, g_model_BtoA, image_shape)
# composite: B -> A -> [real/fake, B]
c_model_BtoA = define_composite_model(g_model_BtoA, d_model_A, g_model_AtoB, image_shape)
# train models
train(d_model_A, d_model_B, g_model_AtoB, g_model_BtoA, c_model_AtoB, c_model_BtoA, dataset)