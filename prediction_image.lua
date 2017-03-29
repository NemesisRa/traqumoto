require 'torch'
require 'image'
require 'nn'
require 'trepl'

N = 15 + 64
l = 70
L = 140

net = torch.load('network.t7','ascii')

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
