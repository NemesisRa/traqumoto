require 'torch'
require 'image'
require 'nn'
require 'trepl'

n1 = 15
n2 = 64
N = n1 + n2
l = 70
L = 140

dataset = torch.load('train_data.t7','ascii')

net = nn.Sequential();  -- make a multi-layer perceptron
inputs = 3; outputs = 2; -- parameters

net = nn.Sequential()
net:add(nn.SpatialConvolution(inputs, 6, 5, 10)) -- 3 input image channels, 6 output channels, 5x10 convolution kernel
net:add(nn.ReLU())				-- non-linearity 
net:add(nn.SpatialMaxPooling(2,2,2,2))		-- A max-pooling operation that looks at 2x2 windows and finds the max.
net:add(nn.SpatialConvolution(6, 16, 5, 10))	-- 6 input image channels, 16 output channels, 5x10 convolution kernel
net:add(nn.ReLU())				-- non-linearity 
net:add(nn.SpatialMaxPooling(2,2,2,2))		-- A max-pooling operation that looks at 2x2 windows and finds the max.
net:add(nn.SpatialConvolution(16, 32, 5, 10))	-- 16 input image channels, 32 output channels, 5x10 convolution kernel
net:add(nn.ReLU())				-- non-linearity 
net:add(nn.SpatialMaxPooling(2,2,2,2))		-- A max-pooling operation that looks at 2x2 windows and finds the max.
net:add(nn.View(32*9*5))			-- reshapes from a 3D tensor into 1D tensor
net:add(nn.Linear(32*9*5, 200))			-- fully connected layer (matrix multiplication between input and weights)
net:add(nn.ReLU())				-- non-linearity 
net:add(nn.Linear(200, 120))			-- fully connected layer (matrix multiplication between input and weights)
net:add(nn.ReLU())				-- non-linearity 
net:add(nn.Linear(120, 84))			-- fully connected layer (matrix multiplication between input and weights)
net:add(nn.ReLU())				-- non-linearity 
net:add(nn.Linear(84, outputs))			-- the number of outputs of the network
net:add(nn.LogSoftMax())			-- converts the output to a log-probability. Useful for classification problems

criterion = nn.ClassNLLCriterion()
trainer = nn.StochasticGradient(net, criterion)
trainer.learningRate = 0.001
trainer.maxIteration = 300
trainer:train(dataset)

torch.save('network.t7', net, 'ascii')
