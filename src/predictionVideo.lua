-- Programme de prédiction d'une vidéo à l'aide de masques
-- Groupe de PI n°4 | 27/04/2017

require 'torch'
require 'nn'
require 'math'
cv = require 'cv'
require 'cv.features2d'
require 'cv.highgui'
require 'cv.videoio'
require 'cv.imgproc'
require 'cv.video'

net = torch.load('network.t7')
vidname = '/home/pi2017/Bureau/Video/test.avi'

dofile("prediction.lua")

write('../Resultat.csv',data,';')
