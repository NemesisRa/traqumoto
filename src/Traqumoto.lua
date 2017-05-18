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

net = torch.load('src/network.t7')
vidname = 'Video/test2.mp4'

dofile("prediction.lua")
