require 'torch'
require 'image'
require 'nn'
require 'trepl'
require 'math'

nt = 1
n1 = 100
n2 = 390
N = n1 + n2
l = 70
L = 140

net = torch.load('network.t7')

imgname = 'BDD/Image_à_tester/vidtest.png'
Imgcoul = image.load(imgname,3)
Img = image.load(imgname,1,'byte')

width = Img:size()[1]
length = Img:size()[2]

for i=1,width-L,5 do
	print(string.format('[%2.0f', i/(width-L)*100)..'%] Prédiction de l\'image')
	for j=1,length-l,5 do
		sub = torch.Tensor(1,L,l):copy(Img:sub(i,i+L-1,j,j+l-1))
		predicted = net:forward(sub:view(1,L,l))
		predicted:exp()
		if predicted[1]>0.999 then
			Imgcoul = image.drawRect(Imgcoul, j, i, j+l-1, i+L-1, {lineWidth = 2, color = {0,255,0}})
		end
	end
end

image.display(Imgcoul)
image.display(Img)
