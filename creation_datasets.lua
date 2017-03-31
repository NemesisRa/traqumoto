require 'torch'
require 'image'
require 'nn'
require 'trepl'

n1 = 15
n2 = 64
N = n1 + n2
l = 70
L = 140

classes = {'Moto', 'Pas_Moto'}
n_classes = 2

imgset = torch.Tensor(N,3,L,l):zero()
labelset = torch.Tensor(N):zero()

for i = 1,N do
	if i <= n1 then
		imgname = string.format('BDD/Motos/%02d.PNG', i)
		Img = image.load(imgname,3)
		labelset[i] = 1
	else
		imgname = string.format('BDD/Pas_Motos/%02d.PNG', i-n1)
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
