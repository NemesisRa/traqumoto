-- Programme de test du réseau de neurone
-- Groupe de PI n°4 | 26/04/2017

require 'torch'			-- Utilisation du module torch
require 'nn'			-- Utilisation du mosule neural network
cv = require 'cv'		-- Utilisation d'OpenCV
require 'cv.imgproc'		-- Utilisation du module imgproc d'OpenCV
require 'cv.imgcodecs'		-- Utilisation du module imgcodecs d'OpenCV

filedir = uigetdir()

local l = 60		-- largeur normalisée des images en entrée du réseau de neurone
local L = 120		-- hauteur normalisée des images en entrée du réseau de neurone
local Ntest = 100	-- Nombre d'échantillons de tests
local seuil = 0.999999	-- seuil pour comparer au résultat de la prédiction

local net = torch.load('network.t7')	-- Chargement du fichier du réseau

print('Détails ? (y or n)')		-- Demande des détails
local key = io.read()			-- Lecture de la réponse

if key == 'y' or key == 'n' then
	local cpt=0
	for i = 1,Ntest do
		if i <100 then
			imgname = string.format('../BDDTest/Motos/%02d.png', i)
		else
			imgname = string.format('../BDDTest/Motos/%03d.png', i)
		end
		local Img = cv.imread{imgname,cv.IMREAD_GRAYSCALE}
		local Imgr = cv.resize{Img,{l,L}}
		local Imgpred = torch.Tensor(1,L,l):copy(Imgr)
		local predicted = net:forward(Imgpred)
		if predicted[1] >= seuil then
			cpt = cpt + 1
		end
		if key == 'y' then
			print('n°' .. i)
			print('pred = ' .. predicted[1])
			print('cpt = ' .. cpt)
			print('\n')
		end
	end
	print('[Résultat] ' .. cpt/Ntest*100 .. '% de Vrai-Positifs pour le seuil de ' .. seuil)
else
	print('Mauvaise entrée')
end
