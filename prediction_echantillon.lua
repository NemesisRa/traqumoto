require 'torch'
require 'image'
require 'nn'
require 'trepl'

N = 15 + 64
l = 70
L = 140

net = torch.load('network.t7','ascii')

Img = image.load('BDD/test01.PNG',3)
r_image = image.scale(Img, l, L)

predicted = net:forward(r_image)
print(predicted:exp())

image.display(Img)
