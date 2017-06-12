-- Programme de test du réseau de neurone
-- Groupe de PI n°4 | 26/04/2017

--[[ Ce programme charge le réseau de neurones puis le teste avec la base de données de tests.
Il renvoie les pourcentages de réussite des échantillons de motos et de pas motos ]]

require 'torch'			-- Utilisation du module torch
require 'nn'			-- Utilisation du module neural network
cv = require 'cv'		-- Utilisation d'OpenCV
require 'cv.imgproc'		-- Utilisation du module imgproc d'OpenCV
require 'cv.imgcodecs'		-- Utilisation du module imgcodecs d'OpenCV

local l = 60			-- largeur normalisée des images en entrée du réseau de neurone
local L = 120			-- hauteur normalisée des images en entrée du réseau de neurone
local n1test = 280		-- Nombre d'images de motos pour le test
local n2test = 320		-- Nombre d'images de pas motos pour le test
local Ntest = n1test + n2test	-- Nombre d'échantillons de tests
local seuil = 1			-- seuil pour comparer au résultat de la prédiction (moto=1, pasmoto=0)

local net = torch.load('network.t7')			-- Chargement du réseau de neurones
local datasetTest = torch.load('datasetTest.t7') 	-- Chargement de la base de données

print('Détails ? (y or n)')		-- Demande de détails
local key = io.read()			-- Lecture de la réponse

if key == 'y' or key == 'n' then	-- si touche y ou n appuyée
	local cptVP = 0 -- compteur Vrai Postif (signifie moto vue comme moto par le reseau)
	local cptFN = 0 -- compteur Faux Negatif (signifie pas moto vue comme pas moto par le reseau)
	for i = 1,Ntest do
		local predicted = net:forward(datasetTest[i][1])	-- prediction de l'echantillon
		if datasetTest[i][2]==1 then		-- si l'image est une moto
			if predicted[1] >= seuil then	-- si la prediction du reseau est >= au seuil
				cptVP = cptVP + 1	-- bonne prediction donc on incrémente le compteur
			end
		else					-- sinon l'image est une pas moto
			if predicted[1] < seuil then	-- et si la prediction est < au seuil 
				cptFN = cptFN + 1	-- bonne prediction donc on incrémente le compteur
			end
		end
		if key == 'y' then				-- si touche y appuyée, afficher
			print('n°' .. i)			-- n° image
			print('pred = ' .. predicted[1])	-- prediction du reseau de neurones
			print('cptVP = ' .. cptVP)		-- nombre vrai positif
			print('cptFN = ' .. cptFN)		-- nombre faux negatif
			print('\n')
		end
	end
	print('[Résultat] ' .. cptVP/n1test*100 .. '% de Vrai-Positifs pour le seuil de ' .. seuil)
	print('[Résultat] ' .. cptFN/n2test*100 .. '% de Faux-Negatifs pour le seuil de ' .. seuil)
else
	print('Mauvaise entrée')
end
