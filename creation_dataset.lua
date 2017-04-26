require 'torch'
require 'image'
require 'nn'
require 'trepl'
cv = require 'cv'
require 'cv.features2d'
require 'cv.imgcodecs'
require 'cv.highgui'
require 'cv.imgproc'

nt = 10
n1 = 212
n2 = 600
N = n1 + n2
l = 60
L = 120

classes = {'Moto', 'Pas_Moto'}
n_classes = 2

imgset = torch.Tensor(N*nt,1,L,l):zero()
labelset = torch.Tensor(N*nt,1):zero()

k=1
for i = 1,N do
	if i <= n1 then
		if i<100 then
			imgname = string.format('BDD/Motos/%02d.png', i)	-- images de 01 à 99
		else
			imgname = string.format('BDD/Motos/%03d.png', i)	-- images de 100 à 999
		end
		for j=k,k+nt-1 do
			labelset[j] = 1
		end
	else
		if i-n1<100 then
			imgname = string.format('BDD/Pas_Motos/%02d.png', i-n1)		--images de 01 à 99
		else
			imgname = string.format('BDD/Pas_Motos/%03d.png', i-n1)		-- images de 100 à 999
		end
		for j=k,k+nt-1 do
			labelset[j] = 0
		end
	end
	Img = cv.imread{imgname,cv.IMREAD_GRAYSCALE}
	Imgr = cv.resize{Img,{l,L}}

	Imgs1 = torch.ByteTensor(Imgr:size()[1],Imgr:size()[2]):copy(Imgr)
	Imgs2 = torch.ByteTensor(Imgr:size()[1],Imgr:size()[2]):copy(Imgr)
	for i=6,Imgr:size()[1] do
		Imgs1[i-5]=Imgr[i]:copy(Imgr[i])
		Imgs2[i]=Imgr[i-5]:copy(Imgr[i-5])
	end
	Imgs3 = torch.ByteTensor(Imgr:size()[1],Imgr:size()[2]):copy(Imgr)
	Imgs4 = torch.ByteTensor(Imgr:size()[1],Imgr:size()[2]):copy(Imgr)
	for j=3,Imgr:size()[2] do
		for i=1,Imgr:size()[1] do
			Imgs3[i][j-2]=Imgr[i][j]
			Imgs4[i][j]=Imgr[i][j-2]
		end
	end
	imgset[k] = torch.Tensor(1,L,l):copy(Imgr)
	k = k+1
	imgset[k] = torch.Tensor(1,L,l):copy(Imgs1)
	k = k+1
	imgset[k] = torch.Tensor(1,L,l):copy(Imgs2)
	k = k+1
	imgset[k] = torch.Tensor(1,L,l):copy(Imgs3)
	k = k+1
	imgset[k] = torch.Tensor(1,L,l):copy(Imgs4)
	k = k+1

	Imgf = torch.ByteTensor(Imgr:size()[1],Imgr:size()[2]):copy(Imgr)
	cv.flip{Imgr,Imgf,1}

	Imgs1 = torch.ByteTensor(Imgf:size()[1],Imgf:size()[2]):copy(Imgf)
	Imgs2 = torch.ByteTensor(Imgf:size()[1],Imgf:size()[2]):copy(Imgf)
	for i=6,Imgf:size()[1] do
		Imgs1[i-5]=Imgf[i]:copy(Imgf[i])
		Imgs2[i]=Imgf[i-5]:copy(Imgf[i-5])
	end
	Imgs3 = torch.ByteTensor(Imgf:size()[1],Imgf:size()[2]):copy(Imgf)
	Imgs4 = torch.ByteTensor(Imgf:size()[1],Imgf:size()[2]):copy(Imgf)
	for j=3,Imgf:size()[2] do
		for i=1,Imgf:size()[1] do
			Imgs3[i][j-2]=Imgf[i][j]
			Imgs4[i][j]=Imgf[i][j-2]
		end
	end
	imgset[k] = torch.Tensor(1,L,l):copy(Imgf)
	k = k+1
	imgset[k] = torch.Tensor(1,L,l):copy(Imgs1)
	k = k+1
	imgset[k] = torch.Tensor(1,L,l):copy(Imgs2)
	k = k+1
	imgset[k] = torch.Tensor(1,L,l):copy(Imgs3)
	k = k+1
	imgset[k] = torch.Tensor(1,L,l):copy(Imgs4)
	k = k+1

	Img = nil
end

mean = imgset:mean()
stdv = imgset:std()

imgset = imgset:apply(function(x)
		x=x*(42/stdv)-mean+127
		x = math.max(math.min(x,255),0)
		return x
	end)

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
