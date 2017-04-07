require 'torch'
require 'image'
require 'nn'
require 'trepl'
require 'math'

n1 = 15
n2 = 64
N = n1 + n2
l = 70
L = 140

net = torch.load('network.t7','ascii')

imgname = 'BDD/Images_a_tester/Motos04.PNG'
Imgcoul = image.load(imgname,3)
Img = image.load(imgname,1)

width = Img:size()[1]
length = Img:size()[2]

Img = Img:view(1, width, length)

for i=1,width-L,5 do
	print(string.format('[%2.0f', i/(width-L)*100)..'%] Prédiction de l\'image')
	for j=1,length-l,5 do
		predicted = net:forward(Img[{{},{i,i+L},{j,j+l}}])
		predicted:exp()
		if predicted[1]>0.999 then
			Imgcoul = image.drawRect(Imgcoul, j, i, j+l, i+L, {lineWidth = 2, color = {0,255,0}})
		end
	end
end

image.display(Imgcoul)
image.display(Img)
