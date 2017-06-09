-- Programme de test du réseau de neurone
-- Groupe de PI n°4 | 26/04/2017

require 'torch'			-- Utilisation du module torch
require 'nn'			-- Utilisation du module neural network
cv = require 'cv'		-- Utilisation d'OpenCV
require 'cv.imgproc'		-- Utilisation du module imgproc d'OpenCV
require 'cv.imgcodecs'		-- Utilisation du module imgcodecs d'OpenCV

local l = 60		-- largeur normalisée des images en entrée du réseau de neurone
local L = 120		-- hauteur normalisée des images en entrée du réseau de neurone
local n1test = 280
local n2test = 320
local Ntest = n1test + n2test	-- Nombre d'échantillons de tests
local seuil = 1			-- seuil pour comparer au résultat de la prédiction (moto=1, pasmoto=0)

local net = torch.load('network.t7')	-- Chargement du fichier du réseau
local datasetTest = torch.load('datasetTest.t7') -- Chargement du fichier de la base de données

print('Détails ? (y or n)')		-- Demande des détails
local key = io.read()			-- Lecture de la réponse

if key == 'y' or key == 'n' then
	local cptVP = 0
	local cptFN = 0
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
		if key == 'y' then
			print('n°' .. i)
			print('pred = ' .. predicted[1])
			print('cptVP = ' .. cptVP)
			print('cptFN = ' .. cptFN)
			print('\n')
		end
	end
	print('[Résultat] ' .. cptVP/Ntest*100 .. '% de Vrai-Positifs pour le seuil de ' .. seuil)
	print('[Résultat] ' .. cptFN/Ntest*100 .. '% de Faux-Negatifs pour le seuil de ' .. seuil)
else
	print('Mauvaise entrée')
end
