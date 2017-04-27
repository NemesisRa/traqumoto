-- Programme de prédiction d'une vidéo à l'aide de masques | Version GPU
-- Groupe de PI n°4 | 27/04/2017

require 'torch'
require 'nn'
require 'cutorch'
require'cunn'
cv = require 'cv'
require 'cv.features2d'
require 'cv.highgui'
require 'cv.videoio'
require 'cv.imgproc'
require 'cv.video'

local nt = 10
local n1 = 212
local n2 = 600
local N = n1 + n2
local l = 60
local L = 120

local net = torch.load('network.t7')
local vidname = 'Video/test2.mp4'
local vid = cv.VideoCapture{vidname}

if not vid:isOpened() then
    print("Failed to open the video")
    os.exit(-1)
end


local pMOG2 = cv.BackgroundSubtractorMOG2{}
local frame = vid:read{}
for i=1,30 do
	if not(vid:read{frame}) then
		break
	end
	fgMaskMOG2 = pMOG2:apply{frame}
end


local width = frame:size()[1]
local length = frame:size()[2]
local Img = torch.Tensor(width,length):zero()
local Img = cv.cvtColor{frame, nil, cv.COLOR_BGR2GRAY}

cv.namedWindow{'win1'}
cv.setWindowTitle{'win1', 'N&B'}

cv.namedWindow{'win2'}
cv.setWindowTitle{'win2', 'BcgSub'}

cv.namedWindow{'win3'}
cv.setWindowTitle{'win3', 'Mask'}

cv.namedWindow{'win4'}
cv.setWindowTitle{'win4', 'Blob'}

cv.namedWindow{'win5'}
cv.setWindowTitle{'win5', 'Detection'}

local pause = false
local key = 0
local key_SPACE = 32
local key_ESCAPE = 27
local key_Q = 113
 
while true do
	if key == key_SPACE then
		key = 0
		pause = not pause
		print("en pause")
		key=cv.waitKey{0}
	elseif key == key_ESCAPE or key == key_Q then --escape or q
		print("quitter")
		break
	end
	
	if not pause then

		if not(vid:read{frame}) then
			print("Fin de la video")
			key=cv.waitKey{0}
			break
		end

		local Img = cv.cvtColor{frame, nil, cv.COLOR_BGR2GRAY}
		local Imgcuda = Img:cuda()

		local fgMaskMOG2 = pMOG2:apply{frame}
		local Mask = torch.ByteTensor(fgMaskMOG2:size()[1],fgMaskMOG2:size()[2]):copy(fgMaskMOG2)

		cv.threshold{Mask, Mask, 100, 255, cv.THRESH_BINARY}

		local erodeElement = cv.getStructuringElement{ cv.MORPH_RECT, cv.Size{4,4}}
		cv.erode{Mask, Mask, erodeElement}
		local dilateElement = cv.getStructuringElement{ cv.MORPH_RECT, cv.Size{6,6}}
		cv.dilate{Mask, Mask, dilateElement}				

		Mask = Mask:apply(function(x)
				x=255-x
				return x
			end)
		
		local params = cv.SimpleBlobDetector_Params{}
		-- Change thresholds
		params.minThreshold = 0
		params.maxThreshold = 255
		-- Filter by Area.
		params.filterByArea = true
		params.minArea = 500
		params.maxArea = 10000000000000
		-- Filter by Circularity
		params.filterByCircularity = false
		-- Filter by Convexity
		params.filterByConvexity = false
		-- Filter by Inertia
		params.filterByInertia = false
		local detector = cv.SimpleBlobDetector{params}
		local keypoints = detector:detect{Mask}
		local ImgBlob = cv.drawKeypoints{Mask, keypoints}

		for k=1,keypoints.size do
			local x = keypoints.data[k].pt.x
			local y = keypoints.data[k].pt.y
			if y+L/2-1<width and y-L/2>0 and x-l/2>0 and x+l/2-1<length then
				cv.rectangle{Img, pt1={x-l/2, y-L/2}, pt2={x+l/2-1, y+L/2-1}, color = {255,255,255}}
				local sub = torch.CudaTensor(1,L,l):copy(Imgcuda:sub(y-L/2,y+L/2-1,x-l/2,x+l/2-1))
				local predicted = net:forward(sub:view(1,L,l))
				if predicted[1]>0.9 then
					cv.rectangle{frame, pt1={x-l/2, y-L/2}, pt2={x+l/2-1, y+L/2-1}, color = {0,255,0}}
				end
			end
		end

		cv.imshow{'win1', Img}			
		cv.imshow{'win2', fgMaskMOG2}
		cv.imshow{'win3', Mask}
		cv.imshow{'win4', ImgBlob}
		cv.imshow{'win5', frame}
		
		key=cv.waitKey{1} --en ms
	end
end

cv.destroyAllWindows{}
