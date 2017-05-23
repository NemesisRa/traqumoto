-- Programme de formatage et prétraitement des données
-- Groupe de PI n°4 | 27/04/2017

require 'torch'
cv = require 'cv'
require 'cv.imgcodecs'
require 'cv.imgproc'

local n1 = 480		-- Nombre d'images de motos
local n2 = 720		-- Nombre d'images de pas motos
local N = n1 + n2	-- Nombre total d'images
local n1app = 380
local n2app = 700
local Napp = n1app + n2app
local n1test = n1 - n1app
local n2test = n2 - n2app
local Ntest = n1test + n2test

local l = 60		-- largeur normalisée des images en entrée du réseau de neurone
local L = 120		-- hauteur normalisée des images en entrée du réseau de neurone

local imgsetMoto = torch.Tensor(n1,1,L,l):zero()
local imgsetPasMoto = torch.Tensor(n2,1,L,l):zero()

for i=1,N do
	if i <= n1 then
		if i<100 then
			imgname = string.format('../BDD/Motos/%02d.png', i)	-- images de 01 à 99
		else
			imgname = string.format('../BDD/Motos/%03d.png', i)	-- images de 100 à 999
		end
		local Img = cv.imread{imgname,cv.IMREAD_GRAYSCALE}
		local Imgr = cv.resize{Img,{l,L}}
		imgsetMoto[i] = torch.Tensor(1,L,l):copy(Imgr)
	else
		if i-n1<100 then
			imgname = string.format('../BDD/Pas_Motos/%02d.png', i-n1)		--images de 01 à 99
		else
			imgname = string.format('../BDD/Pas_Motos/%03d.png', i-n1)		-- images de 100 à 999
		end
		local Img = cv.imread{imgname,cv.IMREAD_GRAYSCALE}
		local Imgr = cv.resize{Img,{l,L}}
		imgsetPasMoto[i-n1] = torch.Tensor(1,L,l):copy(Imgr)
	end
end

local dataMoto = {}
for i=1,n1 do
  local input = imgsetMoto[i]
  local target = 1
  dataMoto[i] = {input, target}
end

local dataPasMoto = {}
for i=1,n2 do
  local input = imgsetPasMoto[i]
  local target = 0
  dataPasMoto[i] = {input, target}
end

local rand1 = torch.randperm(n1)
local rand2 = torch.randperm(n2)

local dataMotoRand = {}
for i=1,n1 do
  dataMotoRand[i] = dataMoto[rand1[i]]
end

local dataPasMotoRand = {}
for i=1,n2 do
  dataPasMotoRand[i] = dataPasMoto[rand2[i]]
end

local nt = 10		-- Nombre de transformation faite pour chaque image

local imgsetApp = torch.Tensor(Napp*nt,1,L,l):zero()
local labelsetApp = torch.Tensor(Napp*nt,1):zero()

local k=1
for i = 1,Napp do
	if i <= n1app then
		for j=k,k+nt-1 do
			labelsetApp[j] = 1		-- label de moto
		end
		Imgr = torch.Tensor(1,L,l):copy(dataMotoRand[i][1])
	else
		for j=k,k+nt-1 do
			labelsetApp[j] = 0		-- label de pas moto
		end
		Imgr = torch.Tensor(1,L,l):copy(dataPasMotoRand[i-n1app][1])
	end

	local Imgs1 = torch.ByteTensor(1,L,l):copy(Imgr)
	local Imgs2 = torch.ByteTensor(1,L,l):copy(Imgr)
	for i=6,Imgr:size()[1] do
		Imgs1[i-5]=Imgr[i]:copy(Imgr[i])
		Imgs2[i]=Imgr[i-5]:copy(Imgr[i-5])
	end
	local Imgs3 = torch.ByteTensor(1,L,l):copy(Imgr)
	local Imgs4 = torch.ByteTensor(1,L,l):copy(Imgr)
	for j=3,Imgr:size()[2] do
		for i=1,Imgr:size()[1] do
			Imgs3[i][j-2]=Imgr[i][j]
			Imgs4[i][j]=Imgr[i][j-2]
		end
	end
	imgsetApp[k] = torch.Tensor(1,L,l):copy(Imgr)
	k = k+1
	imgsetApp[k] = torch.Tensor(1,L,l):copy(Imgs1)
	k = k+1
	imgsetApp[k] = torch.Tensor(1,L,l):copy(Imgs2)
	k = k+1
	imgsetApp[k] = torch.Tensor(1,L,l):copy(Imgs3)
	k = k+1
	imgsetApp[k] = torch.Tensor(1,L,l):copy(Imgs4)
	k = k+1

	local Imgf = torch.ByteTensor(1,L,l):copy(Imgr)
	cv.flip{Imgr,Imgf,1}

	local Imgfs1 = torch.ByteTensor(1,L,l):copy(Imgf)
	local Imgfs2 = torch.ByteTensor(1,L,l):copy(Imgf)
	for i=6,Imgf:size()[1] do
		Imgfs1[i-5]=Imgf[i]:copy(Imgf[i])
		Imgfs2[i]=Imgf[i-5]:copy(Imgf[i-5])
	end
	local Imgfs3 = torch.ByteTensor(1,L,l):copy(Imgf)
	local Imgfs4 = torch.ByteTensor(1,L,l):copy(Imgf)
	for j=3,Imgf:size()[2] do
		for i=1,Imgf:size()[1] do
			Imgfs3[i][j-2]=Imgf[i][j]
			Imgfs4[i][j]=Imgf[i][j-2]
		end
	end
	imgsetApp[k] = torch.Tensor(1,L,l):copy(Imgf)
	k = k+1
	imgsetApp[k] = torch.Tensor(1,L,l):copy(Imgfs1)
	k = k+1
	imgsetApp[k] = torch.Tensor(1,L,l):copy(Imgfs2)
	k = k+1
	imgsetApp[k] = torch.Tensor(1,L,l):copy(Imgfs3)
	k = k+1
	imgsetApp[k] = torch.Tensor(1,L,l):copy(Imgfs4)
	k = k+1

	Img = nil
end

local mean = imgsetApp:mean()
local stdv = imgsetApp:std()
imgsetApp = imgsetApp:apply(function(x)
		x=x*(42/stdv)-mean+127
		x = math.max(math.min(x,255),0)
		return x
	end)

local datasetApp = {}
for i=1,Napp*nt do
  local input = imgsetApp[i]
  local target = labelsetApp[i]
  datasetApp[i] = {input, target}
end

local datasetTest = {}
for i=1,Ntest*nt do
	if i<=n1test then
		datasetTest[i] = dataMotoRand[n1app+i]
	else
		datasetTest[i] = dataPasMotoRand[n2app+i-n1test]
	end
end

torch.save('datasetApp.t7', datasetApp)		-- Enregistre la dataset
torch.save('datasetTest.t7', datasetTest)
