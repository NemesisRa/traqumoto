require 'torch'
require 'image'
require 'nn'
require 'trepl'

N = 15 + 64
l = 70
L = 140

net = torch.load('network.t7','ascii')

Img = image.load('BDD/Images_a_tester/Motos04.PNG',3)

width = Img[1]:size()[1]
length = Img[1]:size()[2]

for i=1,width-L,5 do
	print(string.format('[%2.0f', i/(width-L)*100)..'%] Prédiction de l\'image')
	for j=1,length-l,5 do
		predicted = net:forward(Img[{{},{i,i+L},{j,j+l}}])
		predicted:exp()
		if predicted[1]>0.99 then
			Img = image.drawRect(Img, j, i, j+l, i+L, {lineWidth = 3, color = {0, 255, 0}})
		end
	end
end

image.display(Img)
