require 'torch'
require 'image'
require 'nn'
require 'trepl'

l = 60
L = 120

net = torch.load('network.t7','ascii')

Img = image.load('BDD/test01.PNG',1)
r_image = image.scale(Img, l, L)

predicted = net:forward(r_image:view(1, l, L))
print(predicted:exp())

image.display(r_image)
