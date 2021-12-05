import matplotlib.image as img
import numpy as np
import os
import matplotlib.pyplot as plt
import scipy.io
import mnist
import random


	
def GetRegions(img, dim, stride): #generates all dim X dim regions of a 2d numpy array
	height = img.shape[0]
	width = img.shape[1]
	for i in range(    (height - dim + 1) // stride      ):
		for j in range(    (width - dim + 1) // stride     ):  #misses edge when even, check if matters
			region = img[i*stride:(i*stride+dim), j*stride:(j*stride+dim)]
			yield region, i ,j

class Conv:
	def __init__(self, nfilters, filterDim, stride):
		self.nfilters = nfilters
		self.filterDim = filterDim
		self.filters = np.random.randn(nfilters, filterDim, filterDim) / (filterDim * filterDim) #division step reduces variance. This is important for avoiding bad gradients. https://www.quora.com/What-is-an-intuitive-explanation-of-the-Xavier-Initialization-for-Deep-Neural-Networks
		self.stride = stride
		self.LastInput = 0
	def convolve(self, img): #takes 2d images. Runs it through each filter and returns 3d array
		#img = (img /255) - 0.5   #important for avoiding bad gradients
		self.LastInput = img
		height, width = img.shape
		output = np.zeros(  ( (height - self.filterDim + 1)//self.stride ,  (width - self.filterDim + 1)//self.stride, self.nfilters )  )
		for region, i ,j in GetRegions(img, self.filterDim, self.stride):
			output[i,j] = np.sum(region * self.filters, axis=(1,2))
		return output
	def back(self, dLdOut, LearnRate = .005):
		dLdFilters = np.zeros(self.filters.shape)
		for region, i, j in GetRegions(self.LastInput, self.filterDim, self.stride):
			for f in range(self.nfilters):
				dLdFilters[f] += dLdOut[i, j, f] * region
		self.filters -= LearnRate * dLdFilters

class Max:
	def __init__(self, dim, stride):
		self.dim = dim
		self.stride = stride
		self.LastInput = 0
	def maxpool(self, img):
		self.LastInput = img
		height, width, NumFilters = img.shape
		output = np.zeros(  ( (height - self.dim + 1)//self.stride ,  (width - self.dim + 1)//self.stride, NumFilters )  )
		for region, i, j in GetRegions(img, self.dim, self.stride):
			output[i,j] = np.amax(region, axis=(0,1))
		return output
	def back(self, dLdOut): #dLdInput is shaped, for now it should be 12 x 12 x 8
		dLdInput = np.zeros(self.LastInput.shape)
		for region, i, j, in GetRegions(self.LastInput, self.dim, self.stride):
			height, width, filters = region.shape
			amax = np.amax(region, axis=(0,1))
			for h in range(height):
				for w in range(width):
					for f in range(filters):
						if (region[h,w,f] == amax[f]):
							dLdInput[i * self.stride + h, j * self.stride + w, f] = dLdOut[i,j,f] #check that it should be stride, not dim
		return dLdInput

class Soft: 
	def __init__(self, inputSize, classes):        #just pass input.size
		self.weights = np.random.randn(inputSize, classes) / inputSize
		self.biases = np.zeros(classes)
		self.LastInputShape = 0
		self.LastInput = 0
		self.LastExp = 0
	def activate(self, img):    #img can be 3d, right now we are doing 12 x 12 x 8
		self.LastInputShape = img.shape
		img = img.flatten()
		self.LastInput = img
		product = np.dot(img, self.weights) + self.biases
		product = np.clip(product, -500, 500)
		exp = np.exp(product - np.max(product)) #subtraction prevents bad gradients
		self.LastExp = exp         
		S = np.sum(exp, axis=0)
		probabilities = exp / S
		return probabilities
	def back(self, dLdProb, LearnRate = .005):
		for i, gradient in enumerate(dLdProb):
			if gradient == 0:
				continue
			S = np.sum(self.LastExp, axis = 0)
			dProbdProduct = -self.LastExp[i] * self.LastExp / (S ** 2)
			dProbdProduct[i] = self.LastExp[i] * (S - self.LastExp[i]) / (S ** 2)
			dProductdW = self.LastInput
			dProductdB = 1
			dProductdInput = self.weights
			dLdProduct = dLdProb * dProbdProduct
			dLdW = dProductdW[np.newaxis].T @ dLdProduct[np.newaxis]
			dLdB = dLdProduct * dProductdB
			dLdInput = dProductdInput @ dLdProduct
			self.weights -= LearnRate * dLdW
			self.biases -= LearnRate * dLdB
			return dLdInput.reshape(self.LastInputShape)


			
			

trainX = mnist.train_images()
trainy = mnist.train_labels()
testX = mnist.test_images()
testy = mnist.test_labels()

trainX = trainX[:1000]
trainy = trainy[:1000]
testX = testX[:1000]
testy = testy[:1000]
	
			
c = Conv(8, 3, 1)  #8 filters, 3 x 3, stride is 1 
m = Max(2, 2)      #2x2, stide of 2
out = c.convolve(  (trainX[0]/255)- 0.5  )
out = m.maxpool(out)
sm = Soft(out.size, 10)  #needs to know input size ahead of time, so just calculate one input



def forward(image, label): #im is 2d matrix, label is an int 

  out = c.convolve(  (image/255)- 0.5  ) # division prevents exploding/imploding gradients
  out = m.maxpool(out)
  out = sm.activate(out)

  loss = -np.log(out[label])   
  acc = 1 if np.argmax(out) == label else 0

  return out, loss, acc

def train(image, label, LearnRate=.005): #im is 2d matrix, label is an int 
  
  out, loss, acc = forward(image, label)

  gradient = np.zeros(10)
  gradient[label] = -1 / out[label]

  gradient = sm.back(gradient, LearnRate)
  gradient = m.back(gradient)
  gradient = c.back(gradient, LearnRate)

  return loss, acc


for epoch in range(3):
  print('Epoch %d : ' % (epoch + 1))

  permutation = np.random.permutation(len(trainX))
  trainX = trainX[permutation] #shuffles data
  trainy = trainy[permutation]

  loss = 0
  NumberCorrect = 0
  for i, (image, label) in enumerate(zip(trainX, trainy)):
    if i > 0 and i % 100 == 99:
      print('%d : Average Loss (last 100 batches) %.3f ; Accuracy: %d%%' % (i + 1, loss / 100, NumberCorrect))
      loss = 0
      NumberCorrect = 0

    l, acc = train(image, label)
    loss = loss + l
    NumberCorrect += acc
