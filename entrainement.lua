-- Programme d'entrainement du réseau de neurones  | Version GPU
-- Groupe de PI n°4 | 27/04/2017

require 'torch'
require 'nn'
require 'cutorch'
require 'cunn'

local nt = 10
local n1 = 212
local n2 = 600
local N = n1 + n2
local l = 60
local L = 120

local dataset = torch.load('dataset.t7')

function dataset:size()
    return N*nt
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

net = net:cuda()

criterion = nn.BCECriterion():cuda()			-- Choix du critère d'entrainement, BCE adapté à deux classes
trainer = nn.StochasticGradient(net, criterion)
trainer.learningRate = 0.001
trainer.maxIteration = 100
trainer:train(dataset)

torch.save('network.t7', net)				-- Sauvagarde du réseau de neurone en fichier t7
