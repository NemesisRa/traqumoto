require 'torch'
require 'image'
require 'nn'
require 'trepl'

N = 15 + 64
l = 70
L = 140

classes = {'Moto', 'Pas_Moto'}
n_classes = 2

imgset = torch.Tensor(N,3,L,l):zero()
labelset = torch.Tensor(N):zero()

for i = 1,N do
	if i <= 15 then
		imgname = string.format('BDD/Motos/%02d.PNG', i)
		Img = image.load(imgname,3)
		labelset[i] = 1
	else
		imgname = string.format('BDD/Pas_Motos/%02d.PNG', i-15)
		Img = image.load(imgname,3)
		labelset[i] = 2
	end
	r_image = image.scale(Img, l, L)
	imgset[i] = torch.Tensor(3,l,L):copy(r_image)
	Img = nil
end

dataset = {}
for i=1,imgset:size(1) do
  local input = imgset[i]
  local target = labelset[i]
  dataset[i] = {input, target}
end

function dataset:size()
    return imgset:size(1)
end

torch.save('train_data.t7', dataset, 'ascii')
--dataset = torch.load('train_data.t7','ascii')

net = nn.Sequential();  -- make a multi-layer perceptron
inputs = 3; outputs = 2; -- parameters

net = nn.Sequential()
net:add(nn.SpatialConvolution(inputs, 6, 5, 10)) -- 3 input image channels, 6 output channels, 5x10 convolution kernel
net:add(nn.ReLU())				-- non-linearity 
net:add(nn.SpatialMaxPooling(2,2,2,2))		-- A max-pooling operation that looks at 2x2 windows and finds the max.
net:add(nn.SpatialConvolution(6, 16, 5, 10))
net:add(nn.ReLU())				-- non-linearity 
net:add(nn.SpatialMaxPooling(2,2,2,2))
net:add(nn.SpatialConvolution(16, 32, 5, 10))	
net:add(nn.ReLU())				-- non-linearity 
net:add(nn.SpatialMaxPooling(2,2,2,2))
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
trainer.maxIteration = 200
trainer:train(dataset)

predicted = net:forward(dataset[1][1])
print(predicted:exp())
predicted = net:forward(dataset[2][1])
print(predicted:exp())
predicted = net:forward(dataset[N][1])
print(predicted:exp())

Img = image.load('BDD/Images_a_tester/Motos01.PNG',3)

width = Img[1]:size()[1]
length = Img[1]:size()[2]

for i=1,width-L do
	print(string.format('[%03d\%]Prédiction de l\'image', i/width-L))
	for j=1,length-l do
		predicted = net:forward(Img[{{},{i,i+L},{j,j+l}}])
		predicted:exp()
		if predicted[1]>0.90 then
			image.drawRect(Img, i, j, i+L, j+l, {lineWidth = 3, color = {0, 255, 0}})
		end
	end
end

image.display(Img)
