require 'torch'
require 'image'
require 'nn'
require 'trepl'
require 'math'

n1 = 15
n2 = 64
N = n1 + n2
l = 70
L = 140

imgname = 'BDD/Motos/35.png'
Img = image.load(imgname,1,'byte')
image.display(Img)
print(Img[1][1])
Imgd = Img:apply(function(x)
  --x = math.max(x - 25,0)
  x = math.min(x + 100,255)
  return x
end)
print(Imgd[1][1])
image.display(Imgd)
--[[for i=5,15,5 do
	Imgloc = image.rotate(Img, i*math.pi/180)
	image.display(Imgloc)
	Imgloc = image.rotate(Img, i*-1*math.pi/180)
	image.display(Imgloc)
end

Img = image.hflip(Img)
image.display(Img)
for i=5,15,5 do
	Imgloc = image.rotate(Img, i*math.pi/180)
	image.display(Imgloc)
	Imgloc = image.rotate(Img, i*-1*math.pi/180)
	image.display(Imgloc)
end]]
