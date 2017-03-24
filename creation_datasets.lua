require 'torch'
require 'image'

N = 15 + 64
l = 70
L = 140

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
		labelset[i] = 0
	end
	r_image = image.scale(Img, l, L)
	imgset[i] = torch.Tensor(3,l,L):copy(r_image)
	Img = nil
end

train_data = {data = imgset, label=labelset}

torch.save('train_data.t7', train_data, 'ascii')
