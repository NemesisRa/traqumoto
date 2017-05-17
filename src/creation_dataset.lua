-- Programme de formatage et prétraitement des données
-- Groupe de PI n°4 | 27/04/2017

require 'torch'
cv = require 'cv'
require 'cv.imgcodecs'
require 'cv.imgproc'

local nt = 10		-- Nombre de transformation faite pour chaque image
local n1 = 212		-- Nombre d'images de motos
local n2 = 600		-- Nombre d'images de pas motos
local N = n1 + n2	-- Nombre total d'images
local l = 60		-- largeur normalisée des images en entrée du réseau de neurone
local L = 120		-- hauteur normalisée des images en entrée du réseau de neurone

local imgset = torch.Tensor(N*nt,1,L,l):zero()
local labelset = torch.Tensor(N*nt,1):zero()

local k=1
for i = 1,N do
	if i <= n1 then
		if i<100 then
			imgname = string.format('../BDD/Motos/%02d.png', i)	-- images de 01 à 99
		else
			imgname = string.format('../BDD/Motos/%03d.png', i)	-- images de 100 à 999
		end
		for j=k,k+nt-1 do
			labelset[j] = 1		-- label de moto
		end
	else
		if i-n1<100 then
			imgname = string.format('../BDD/Pas_Motos/%02d.png', i-n1)		--images de 01 à 99
		else
			imgname = string.format('../BDD/Pas_Motos/%03d.png', i-n1)		-- images de 100 à 999
		end
		for j=k,k+nt-1 do
			labelset[j] = 0		-- label de pas moto
		end
	end

	local Img = cv.imread{imgname,cv.IMREAD_GRAYSCALE}
	local Imgr = cv.resize{Img,{l,L}}

	local Imgs1 = torch.ByteTensor(Imgr:size()[1],Imgr:size()[2]):copy(Imgr)
	local Imgs2 = torch.ByteTensor(Imgr:size()[1],Imgr:size()[2]):copy(Imgr)
	for i=6,Imgr:size()[1] do
		Imgs1[i-5]=Imgr[i]:copy(Imgr[i])
		Imgs2[i]=Imgr[i-5]:copy(Imgr[i-5])
	end
	local Imgs3 = torch.ByteTensor(Imgr:size()[1],Imgr:size()[2]):copy(Imgr)
	local Imgs4 = torch.ByteTensor(Imgr:size()[1],Imgr:size()[2]):copy(Imgr)
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

	local Imgf = torch.ByteTensor(Imgr:size()[1],Imgr:size()[2]):copy(Imgr)
	cv.flip{Imgr,Imgf,1}

	local Imgfs1 = torch.ByteTensor(Imgf:size()[1],Imgf:size()[2]):copy(Imgf)
	local Imgfs2 = torch.ByteTensor(Imgf:size()[1],Imgf:size()[2]):copy(Imgf)
	for i=6,Imgf:size()[1] do
		Imgfs1[i-5]=Imgf[i]:copy(Imgf[i])
		Imgfs2[i]=Imgf[i-5]:copy(Imgf[i-5])
	end
	local Imgfs3 = torch.ByteTensor(Imgf:size()[1],Imgf:size()[2]):copy(Imgf)
	local Imgfs4 = torch.ByteTensor(Imgf:size()[1],Imgf:size()[2]):copy(Imgf)
	for j=3,Imgf:size()[2] do
		for i=1,Imgf:size()[1] do
			Imgfs3[i][j-2]=Imgf[i][j]
			Imgfs4[i][j]=Imgf[i][j-2]
		end
	end
	imgset[k] = torch.Tensor(1,L,l):copy(Imgf)
	k = k+1
	imgset[k] = torch.Tensor(1,L,l):copy(Imgfs1)
	k = k+1
	imgset[k] = torch.Tensor(1,L,l):copy(Imgfs2)
	k = k+1
	imgset[k] = torch.Tensor(1,L,l):copy(Imgfs3)
	k = k+1
	imgset[k] = torch.Tensor(1,L,l):copy(Imgfs4)
	k = k+1

	Img = nil
end

local mean = imgset:mean()
local stdv = imgset:std()
imgset = imgset:apply(function(x)
		x=x*(42/stdv)-mean+127
		x = math.max(math.min(x,255),0)
		return x
	end)

local dataset = {}
for i=1,N*nt do
  local input = imgset[i]
  local target = labelset[i]
  dataset[i] = {input, target}
end

torch.save('dataset.t7', dataset)	-- Enregistre la dataset
