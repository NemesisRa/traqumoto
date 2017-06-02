-- Programme de création du réseau de neuronne ainsi que l'entrainement
-- Groupe de PI n°4 | 23/05/2017

require 'torch'
require 'nn'
cv = require 'cv'
require 'cv.imgcodecs'
require 'cv.imgproc'

local n1 = 1300		-- Nombre d'images de motos
local n2 = 1800		-- Nombre d'images de pas motos
local N = n1 + n2	-- Nombre total d'images
local n1app = 1200
local n2app = 1700
local Napp = n1app + n2app
local n1test = n1 - n1app
local n2test = n2 - n2app
local Ntest = n1test + n2test

local nbiterations = 1

local nt = 10
local l = 60		-- largeur normalisée des images en entrée du réseau de neurone
local L = 120		-- hauteur normalisée des images en entrée du réseau de neurone

function creation_dataset()
	local imgsetMoto = torch.Tensor(n1,1,L,l):zero()
	local imgsetPasMoto = torch.Tensor(n2,1,L,l):zero()

	for i=1,N do
		if i <= n1 then
			if i<100 then
				imgname = string.format('../BDD/Motos/%02d.png', i)	-- images de 01 à 99
			else
				if i<1000 then
					imgname = string.format('../BDD/Motos/%03d.png', i)	-- images de 100 à 999
				else
					imgname = string.format('../BDD/Motos/%04d.png', i)	-- images de 1000 à 9999
				end
			end
			local Img = cv.imread{imgname,cv.IMREAD_GRAYSCALE}
			local Imgr = cv.resize{Img,{l,L}}
			imgsetMoto[i] = torch.Tensor(1,L,l):copy(Imgr)
		else
			if i-n1<100 then
				imgname = string.format('../BDD/Pas_Motos/%02d.png', i-n1)		--images de 01 à 99
			else
				if i-n1<1000 then
					imgname = string.format('../BDD/Pas_Motos/%03d.png', i-n1)		-- images de 100 à 999
				else
					imgname = string.format('../BDD/Pas_Motos/%04d.png', i-n1)		-- images de 1000 à 9999
				end
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

	torch.save('datasetApp.t7', datasetApp)		-- Enregistre les datasets
	torch.save('datasetTest.t7', datasetTest)
	return datasetApp,datasetTest
end

function entrainement(dataset)

	function dataset:size()
		return Napp*nt
	end

	local inputs = 1
	local couche1 = 4
	local couche2 = 16
	local couche3 = 2000
	local outputs = 1

	local tailleConvolution = 5
	local tailleMaxPooling = 2

	local net = nn.Sequential()
	net:add(nn.SpatialConvolution(inputs,couche1,tailleConvolution,tailleConvolution))			-- Convulution
	net:add(nn.ReLU())											-- Application du ReLU 
	net:add(nn.SpatialMaxPooling(tailleMaxPooling,tailleMaxPooling,tailleMaxPooling,tailleMaxPooling))	-- Max Pooling pour réduire les images
	net:add(nn.SpatialConvolution(couche1,couche2,tailleConvolution,tailleConvolution))			-- 6 input image channels, 16 output channels, 5x5 convolution kernel
	net:add(nn.ReLU())											-- Application du ReLU 
	net:add(nn.SpatialMaxPooling(tailleMaxPooling,tailleMaxPooling,tailleMaxPooling,tailleMaxPooling))	-- Max Pooling pour réduire les images
	net:add(nn.View(couche2*27*12))										-- redimmensionnement en un seul tableau 
	net:add(nn.Linear(couche2*27*12,couche3))								-- Liens entre la deuxième et troisième couche
	net:add(nn.ReLU())											-- Application du ReLU
	net:add(nn.Linear(couche3,outputs))									-- Liens entre la  troisième couche et la couche de sortie
	net:add(nn.Sigmoid())											-- Sigmoid pour que les résultats soient entre 0 et 1

	criterion = nn.BCECriterion()				-- Choix du critère d'entrainement, BCE adapté à deux classes
	trainer = nn.StochasticGradient(net, criterion)
	trainer.learningRate = 0.001
	trainer.maxIteration = nbiterations
	trainer:train(dataset)
	return net
end

function testNetwork(net,datasetTest,seuil)
	local cptVP=0
	local cptFN=0
	for i = 1,Ntest do
		local predicted = net:forward(datasetTest[i][1])
		if datasetTest[i][2]==1 then
			if predicted[1] >= seuil then
				cptVP = cptVP + 1
			end
		else
			if predicted[1] < seuil then
				cptFN = cptFN + 1
			end
		end
	end
	print(string.format('[Résultat] %.1f%% de Vrai-Positifs pour le seuil de ', cptVP/n1test*100) .. seuil)
	print(string.format('[Résultat] %.1f%% de Faux-Négatifs pour le seuil de ', cptFN/n2test*100) .. seuil)
	return cptVP/n1test*100, cptFN/n2test*100
end

nbTests = 10

print("[Main] " .. nbTests .. " tests vont être fait sur ces paramètres")

local tpstab = torch.Tensor(nbTests):zero()
local VP1 = torch.Tensor(nbTests):zero()
local FN1 = torch.Tensor(nbTests):zero()
local VP2 = torch.Tensor(nbTests):zero()
local FN2 = torch.Tensor(nbTests):zero()
local VP3 = torch.Tensor(nbTests):zero()
local FN3 = torch.Tensor(nbTests):zero()
local meilleurNet = {}
for i=1,nbTests do
	-- Main
	print("[Test" .. i .. "] Prétraitement de la base de donnée")
	datasetApp,datasetTest = creation_dataset()
	print("[Test" .. i .. "] Entrainement du réseau")
	local tps = os.time()
	net = entrainement(datasetApp)
	tps = (os.time() - tps)
	tpstab[i] = tps
	print("[Test" .. i .. "] Temps d'entrainement : " .. math.floor(tps/86400) .. "d " .. math.floor(tps/3600)%86400 .. "h " .. math.floor(tps/60)%60 .. "m " .. tps%60 .. "s")
	print("[Test" .. i .. "] Test du réseau")
	r1,r2 = testNetwork(net,datasetTest,1)
	if i==1 then
		meilleurNet[1] = net
		meilleurNet[2] = r1
		meilleurNet[3] = r2
	else
		if meilleurNet[2]>r1 and meilleurNet[3]>r2 then
			meilleurNet[1] = net
			meilleurNet[2] = r1
			meilleurNet[3] = r2
		end
	end
	VP1[i] = r1
	FN1[i] = r2
	r1,r2 = testNetwork(net,datasetTest,0.9999)
	VP2[i] = r1
	FN2[i] = r2
	r1,r2 = testNetwork(net,datasetTest,0.9)
	VP3[i] = r1
	FN3[i] = r2
end

torch.save('network.t7', meilleurNet[1])		-- Sauvagarde du meilleur réseau de neurone en fichier t7
print("[Main] Meilleur réseau sauvegardé")

tpsmoy = tpstab:mean()
tpsstd = tpstab:std()
moy1VP = VP1:mean()
std1VP = VP1:std()
moy1FN = FN1:mean()
std1FN = FN1:std()
moy2VP = VP2:mean()
std2VP = VP2:std()
moy2FN = FN2:mean()
std2FN = FN2:std()
moy3VP = VP3:mean()
std3VP = VP3:std()
moy3FN = FN3:mean()
std3FN = FN3:std()
print("[Main] Rappel des Paramètres :")
print("[Main] Nombre d'images total : " .. n1 .. " motos et " .. n2 .. " pas motos. Soit " .. N .. " images.")
print("[Main] Nombre d'images pour l'apprentissage : " .. n1app .. " motos et " .. n2app .. " pas motos. Soit " .. Napp .. " images.")
print("[Main] Nombre d'images pour les tests : " .. n1test .. " motos et " .. n2test .. " pas motos. Soit " .. Ntest .. " images.")
print("[Main] Nombre d'itérations : " .. nbiterations)
print("[Résultats] Temps d'entrainement moyen : " .. math.floor(tpsmoy/86400) .. "d " .. math.floor(tpsmoy/3600)%86400 .. "h " .. math.floor(tpsmoy/60)%60 .. "m " .. string.format('%.1f', (tpsmoy%60)) .. "s")
print("[Résultats] Ecart-type du temps d'entrainement : " .. math.floor(tpsstd/86400) .. "d " .. math.floor(tpsstd/3600)%86400 .. "h " .. math.floor(tpsstd/60)%60 .. "m " .. string.format('%.1f', (tpsstd%60)) .. "s")
print(string.format('[Résultats] Moyenne de %.1f%% de Vrai-Positifs pour le seuil de 1, écart-type de %.1f%%', moy1VP, std1VP))
print(string.format('[Résultats] Moyenne de %.1f%% de Faux-Negatifs pour le seuil de 1, écart-type de %.1f%%', moy1FN, std1FN))
print(string.format('[Résultats] Moyenne de %.1f%% de Vrai-Positifs pour le seuil de 0.9999, écart-type de %.1f%%', moy2VP, std2VP))
print(string.format('[Résultats] Moyenne de %.1f%% de Faux-Negatifs pour le seuil de 0.9999, écart-type de %.1f%%', moy2FN, std2FN))
print(string.format('[Résultats] Moyenne de %.1f%% de Vrai-Positifs pour le seuil de 0.9, écart-type de %.1f%%', moy3VP, std3VP))
print(string.format('[Résultats] Moyenne de %.1f%% de Faux-Negatifs pour le seuil de 0.9, écart-type de %.1f%%', moy3FN, std3FN))
