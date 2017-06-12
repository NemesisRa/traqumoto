-- Programme de prédiction d'une vidéo à l'aide de masques
-- Groupe de PI n°4 | 27/04/2017

require 'torch'		-- Utilisation du module torch
require 'nn'		-- Utilisation du module neural network
require 'math'		-- Utilisation du module math
cv = require 'cv'	-- Utilisation d'OpenCV
require 'cv.features2d'	-- Utilisation du module features2d d'OpenCV
require 'cv.highgui'	-- Utilisation du module highgui d'OpenCV
require 'cv.videoio'	-- Utilisation du module videoio d'OpenCV
require 'cv.imgproc'	-- Utilisation du module imgproc d'OpenCV
require 'cv.video'	-- Utilisation du module video d'OpenCV


net = torch.load('src/network.t7')	-- Charge le réseau de neurones network.t7
vidname = getFile()			-- Appel de la fonction getFile de traqumoto.cpp
vidname = vidname:gsub("\n", "")	-- retire le caractère \n à la fin du chemin de la vidéo
print(vidname)				-- Affiche le chemin

dofile("src/prediction.lua")		-- Execute le programme de prediction

write('Resultat.csv',data,';')		-- Ecris le résultat dans un fichier excel
