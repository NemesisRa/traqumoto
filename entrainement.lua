require 'torch'
require 'image'
require 'nn'
require 'trepl'

nt = 10
n1 = 140
n2 = 500
N = n1 + n2
l = 60
L = 120

dataset = torch.load('dataset.t7')

function dataset:size()
    return N*nt
end

net = nn.Sequential();  -- make a multi-layer perceptron
inputs = 1; outputs = 1; -- parameters

net = nn.Sequential()
net:add(nn.SpatialConvolution(inputs, 4, 5, 5)) -- 1 input image channels, 3 output channels, 5x5 convolution kernel
net:add(nn.ReLU())				-- non-linearity 
net:add(nn.SpatialMaxPooling(2,2,2,2))		-- A max-pooling operation that looks at 2x2 windows and finds the max.
net:add(nn.SpatialConvolution(4, 16, 5, 5))	-- 6 input image channels, 16 output channels, 5x5 convolution kernel
net:add(nn.ReLU())				-- non-linearity 
net:add(nn.SpatialMaxPooling(2,2,2,2))		-- A max-pooling operation that looks at 2x2 windows and finds the max.
net:add(nn.View(16*27*12))			-- reshapes from a 3D tensor into 1D tensor 
net:add(nn.Linear(16*27*12, 2000))			-- fully connected layer (matrix multiplication between input and weights)
net:add(nn.ReLU())				-- non-linearity 
net:add(nn.Linear(2000, outputs))			-- the number of outputs of the network
net:add(nn.Sigmoid())			-- converts the output to a log-probability. Useful for classification problems

criterion = nn.BCECriterion()
trainer = nn.StochasticGradient(net, criterion)
trainer.learningRate = 0.001
trainer.maxIteration = 50
trainer:train(dataset)

torch.save('network.t7', net)
