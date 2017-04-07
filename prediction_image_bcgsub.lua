require 'torch'
require 'image'
require 'nn'
require 'trepl'
require 'math'
cv = require 'cv'
require 'cv.features2d'
require 'cv.imgcodecs'
require 'cv.highgui'
require 'cv.objdetect'
require 'cv.videoio'
require 'cv.imgproc'

nt = 1
n1 = 100
n2 = 130
N = n1 + n2
l = 70
L = 140

net = torch.load('network.t7')

imgname = 'BDD/Image_à_tester/Motos04.PNG'
Imgcoul = cv.imread{imgname}
Img = cv.imread{imgname,cv.IMREAD_GRAYSCALE}

width = Img:size()[1]
length = Img:size()[2]

for i=1,width-L,5 do
	print(string.format('[%2.0f', i/(width-L)*100)..'%] Prédiction de l\'image')
	for j=1,length-l,5 do
		sub = torch.Tensor(1,L,l):copy(Img:sub(i,i+L-1,j,j+l-1))
		predicted = net:forward(sub:view(1,L,l))
		predicted:exp()
		if predicted[1]>0.999 then
			cv.rectangle{Imgcoul, pt1={j, i}, pt2={j+l-1, i+L-1}, color = {0,255,0}}

		end
	end
end

cv.namedWindow{'win1'}
cv.setWindowTitle{'win1', 'N&B'}
cv.imshow{'win1', Img}

cv.namedWindow{'win2'}
cv.setWindowTitle{'win2', 'Couleur'}
cv.imshow{'win2', Imgcoul}
cv.waitKey{0}
cv.destroyAllWindows{}
