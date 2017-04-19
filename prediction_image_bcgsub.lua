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
require 'cv.video'

nt = 1
n1 = 140
n2 = 500
N = n1 + n2
l = 60
L = 120

net = torch.load('network.t7')

vidname = 'Video/test2.mp4'
vid = cv.VideoCapture{vidname}
if not vid:isOpened() then
    print("Failed to open the video")
    os.exit(-1)
end

--[[imgname = 'BDD/Image_à_tester/Motos04.PNG'
Imgcoul = cv.imread{imgname}
Img = cv.imread{imgname,cv.IMREAD_GRAYSCALE}

width = Img:size()[1]
length = Img:size()[2] ]]

pMOG2 = cv.BackgroundSubtractorMOG2{}
local _, frame = vid:read{}
for i=1,30 do
	if not(vid:read{frame}) then
		break
	end
	fgMaskMOG2 = pMOG2:apply{frame}
end

width = frame:size()[1]
length = frame:size()[2]
Img = torch.Tensor(width,length):zero()
Img = cv.cvtColor{frame, nil, cv.COLOR_BGR2GRAY}

cv.namedWindow{'win1'}
cv.setWindowTitle{'win1', 'N&B'}
cv.imshow{'win1', Img}

cv.namedWindow{'win2'}
cv.setWindowTitle{'win2', 'Mask'}
cv.imshow{'win2', fgMaskMOG2}

cv.namedWindow{'win3'}
cv.setWindowTitle{'win3', 'Couleur'}
cv.imshow{'win3', frame}

key=0
while key~=27 and key~=113 do
	if not(vid:read{frame}) then
		break
	end
	fgMaskMOG2 = pMOG2:apply{frame}

	Img = cv.cvtColor{frame, nil, cv.COLOR_BGR2GRAY}

	for i=L/2+1,width-L/2,5 do
		print(string.format('[%2.0f', i/(width-L/2)*100)..'%] Prédiction de l\'image')
		for j=l/2+1,length-l/2,5 do
			if fgMaskMOG2[i][j]>250 then
				m=0
				for k=-5,5 do
					for l=-4,4 do
						m=m+fgMaskMOG2[i+k][j+l]
					end
				end
				m=m/99
				if m>200 then
					--cv.rectangle{frame, pt1={j-l/2, i-L/2}, pt2={j+l/2-1, i+L/2-1}, color = {255,0,0}}
					sub = torch.Tensor(1,L,l):copy(Img:sub(i-L/2,i+L/2-1,j-l/2,j+l/2-1))
					predicted = net:forward(sub:view(1,L,l))
					--predicted = predicted:exp()
					if predicted[1]==1 then
						cv.rectangle{frame, pt1={j-l/2, i-L/2}, pt2={j+l/2-1, i+L/2-1}, color = {0,255,0}}
					end
				end
			end
		end
	end

	cv.imshow{'win1', Img}
	cv.imshow{'win2', fgMaskMOG2}
	cv.imshow{'win3', frame}
	key=cv.waitKey{1}
	
	for i=1,2 do
		if not(vid:read{frame}) then
			break
		end
		fgMaskMOG2 = pMOG2:apply{frame}
	end
end

cv.waitKey{0}
cv.destroyAllWindows{}

--[[width = frame:size()[1]
length = frame:size()[2]
Img = torch.Tensor(width,length):zero()
Img = cv.cvtColor{frame, nil, cv.COLOR_BGR2GRAY}

for i=L/2+1,width-L/2,5 do
	print(string.format('[%2.0f', i/(width-L/2)*100)..'%] Prédiction de l\'image')
	for j=l/2+1,length-l/2,5 do
		if fgMaskMOG2[i][j] > 150 then
			--cv.rectangle{frame, pt1={j-l/2, i-L/2}, pt2={j+l/2-1, i+L/2-1}, color = {255,0,0}}
			sub = torch.Tensor(1,L,l):copy(Img:sub(i-L/2,i+L/2-1,j-l/2,j+l/2-1))
			predicted = net:forward(sub:view(1,L,l))
			predicted = predicted:exp()
			if predicted[1]>0.9 then
				cv.rectangle{frame, pt1={j-l/2, i-L/2}, pt2={j+l/2-1, i+L/2-1}, color = {0,255,0}}
			end
		end
	end
end

cv.namedWindow{'win1'}
cv.setWindowTitle{'win1', 'N&B'}
cv.imshow{'win1', Img}

cv.namedWindow{'win2'}
cv.setWindowTitle{'win2', 'Mask'}
cv.imshow{'win2', fgMaskMOG2}

cv.namedWindow{'win3'}
cv.setWindowTitle{'win3', 'Couleur'}
cv.imshow{'win3', frame}
cv.waitKey{0}
cv.destroyAllWindows{}]]
