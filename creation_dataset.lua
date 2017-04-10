require 'torch'
require 'image'
require 'nn'
require 'trepl'

nt = 1
n1 = 100
n2 = 300
N = n1 + n2
l = 70
L = 140

classes = {'Moto', 'Pas_Moto'}
n_classes = 2

imgset = torch.Tensor(N*nt,1,L,l):zero()
labelset = torch.Tensor(N*nt):zero()

k=1
for i = 1,N do
	if i <= n1 then
		if i<100 then
			imgname = string.format('BDD/Motos/%02d.png', i)
		else
			imgname = string.format('BDD/Motos/%03d.png', i)
		end
		for j=k,k+nt-1 do
			labelset[j] = 1
		end
	else
		if i-n1<100 then
			imgname = string.format('BDD/Pas_Motos/%02d.png', i-n1)
		else
			imgname = string.format('BDD/Pas_Motos/%03d.png', i-n1)
		end
		for j=k,k+nt-1 do
			labelset[j] = 2
		end
	end
	Img = image.load(imgname,1,'byte')
	r_image = image.scale(Img, l, L)
	imgset[k] = torch.Tensor(1,L,l):copy(r_image)
	k = k+1
	--[[for t=5,15,5 do
		Imgloc = image.rotate(r_image, t*math.pi/180)
		imgset[k] = torch.Tensor(1,L,l):copy(Imgloc)
		k = k+1
		Imgloc = image.rotate(r_image, t*-1*math.pi/180)
		imgset[k] = torch.Tensor(1,L,l):copy(Imgloc)
		k = k+1
	end

	Imgloc = r_image:apply(function(x)
	  x = math.max(x - 25,0)
	  return x
	end)
	imgset[k] = torch.Tensor(1,L,l):copy(Imgloc)
	k = k+1
	Imgloc = r_image:apply(function(x)
	  x = math.max(x - 50,0)
	  return x
	end)
	imgset[k] = torch.Tensor(1,L,l):copy(Imgloc)
	k = k+1
	Imgloc = r_image:apply(function(x)
	  x = math.min(x + 50,255)
	  return x
	end)
	imgset[k] = torch.Tensor(1,L,l):copy(Imgloc)
	k = k+1
	Imgloc = r_image:apply(function(x)
	  x = math.min(x + 100,255)
	  return x
	end)
	imgset[k] = torch.Tensor(1,L,l):copy(Imgloc)
	k = k+1


	r_image = image.hflip(r_image)
	imgset[k] = torch.Tensor(1,L,l):copy(r_image)
	k = k+1
	for t=5,15,5 do
		Imgloc = image.rotate(r_image, t*math.pi/180)
		imgset[k] = torch.Tensor(1,L,l):copy(Imgloc)
		k = k+1
		Imgloc = image.rotate(r_image, t*-1*math.pi/180)
		imgset[k] = torch.Tensor(1,L,l):copy(Imgloc)
		k = k+1
	end

	Imgloc = r_image:apply(function(x)
	  x = math.max(x - 25,0)
	  return x
	end)
	imgset[k] = torch.Tensor(1,L,l):copy(Imgloc)
	k = k+1
	Imgloc = r_image:apply(function(x)
	  x = math.max(x - 50,0)
	  return x
	end)
	imgset[k] = torch.Tensor(1,L,l):copy(Imgloc)
	k = k+1
	Imgloc = r_image:apply(function(x)
	  x = math.min(x + 50,255)
	  return x
	end)
	imgset[k] = torch.Tensor(1,L,l):copy(Imgloc)
	k = k+1
	Imgloc = r_image:apply(function(x)
	  x = math.min(x + 100,255)
	  return x
	end)
	imgset[k] = torch.Tensor(1,L,l):copy(Imgloc)
	k = k+1]]

	Img = nil
end

dataset = {}
for i=1,N*nt do
  local input = imgset[i]
  local target = labelset[i]
  dataset[i] = {input, target}
end

function dataset:size()
    return N*nt
end

torch.save('dataset.t7', dataset)
