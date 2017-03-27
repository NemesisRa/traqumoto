require 'torch'
require 'nn'
require 'trepl'

N = 15 + 64
l = 70
L = 140

train_data = torch.load('train_data.t7','ascii')

net = nn.Sequential();  -- make a multi-layer perceptron
inputs = 3*l*L; outputs = 1; HL = 50000; -- parameters

net:add(nn.Linear(inputs, HL))
net:add(nn.Tanh())
net:add(nn.Linear(HL, outputs))

criterion = nn.ClassNLLCriterion()
trainer = nn.StochasticGradient(net, criterion)
trainer.learningRate = 0.001
trainer.maxIteration = 5 -- just do 5 epochs of training.
trainer:train(train_data)

predicted = net:forward(train_data.data[1])
print(predicted:exp())
