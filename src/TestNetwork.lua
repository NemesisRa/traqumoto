-- Programme de test du réseau de neurone
-- Groupe de PI n°4 | 26/04/2017

require 'torch'			-- Utilisation du module torch
require 'nn'			-- Utilisation du mosule neural network
cv = require 'cv'		-- Utilisation d'OpenCV
require 'cv.imgproc'		-- Utilisation du module imgproc d'OpenCV
require 'cv.imgcodecs'		-- Utilisation du module imgcodecs d'OpenCV

local l = 60		-- largeur normalisée des images en entrée du réseau de neurone
local L = 120		-- hauteur normalisée des images en entrée du réseau de neurone
local Ntest = 120	-- Nombre d'échantillons de tests
local seuil = 0.9999	-- seuil pour comparer au résultat de la prédiction

local net = torch.load('network.t7')	-- Chargement du fichier du réseau
local datasetTest = torch.load('datasetTest.t7')

print('Détails ? (y or n)')		-- Demande des détails
local key = io.read()			-- Lecture de la réponse

if key == 'y' or key == 'n' then
	local cpt=0
	for i = 1,Ntest do
		local predicted = net:forward(datasetTest[i][1])
		if datasetTest[i][2]==1 then
			if predicted[1] >= seuil then
				cpt = cpt + 1
			end
		else
			if predicted[1] < seuil then
				cpt = cpt + 1
			end
		end
		if key == 'y' then
			print('n°' .. i)
			print('pred = ' .. predicted[1])
			print('cpt = ' .. cpt)
			print('\n')
		end
	end
	print('[Résultat] ' .. cpt/Ntest*100 .. '% de bonne prédictions pour le seuil de ' .. seuil)
else
	print('Mauvaise entrée')
end
