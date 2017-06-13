-- Programme de prédiction d'une vidéo à l'aide de masques
-- Groupe de PI n°4 | 27/04/2017

--[[ Ce fichier permet d'executer le promgramme Traqu'moto par commandes via le terminal (>> th predictionVideo.lua)
On charge le réseau de neurones, choisit le chemin de la vidéo, execute le programme de prédiction puis enregistre les résultats dans un fichier excel. ]]

require 'torch'		-- Utilisation du module torch
require 'nn'		-- Utilisation du module neural network
require 'math'		-- Utilisation du module math
cv = require 'cv'	-- Utilisation d'OpenCV
require 'cv.features2d'	-- Utilisation du module features2d d'OpenCV
require 'cv.highgui'	-- Utilisation du module highgui d'OpenCV
require 'cv.videoio'	-- Utilisation du module videoio d'OpenCV
require 'cv.imgproc'	-- Utilisation du module imgproc d'OpenCV
require 'cv.video'	-- Utilisation du module video d'OpenCV

net = torch.load('network.t7')			-- Charge le réseau de neurones network.t7
vidname = '/home/pi2017/Bureau/Video/demo.avi'	-- Chemin de la vidéo en entrée

dofile("prediction.lua")	-- Execute le fichier prediction.lua

write('../Resultat.csv',data,';')	-- Ecris le résultat dans un fichier excel
