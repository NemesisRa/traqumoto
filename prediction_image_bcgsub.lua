require 'torch'
require 'image'
require 'nn'
require 'trepl'
require 'math'
require 'cutorch'
require'cunn'
cv = require 'cv'
require 'cv.features2d'
require 'cv.imgcodecs'
require 'cv.highgui'
require 'cv.objdetect'
require 'cv.videoio'
require 'cv.imgproc'
require 'cv.video'

nt = 10
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

cv.namedWindow{'win2'}
cv.setWindowTitle{'win2', 'BcgSub'}

cv.namedWindow{'win3'}
cv.setWindowTitle{'win3', 'Mask'}

cv.namedWindow{'win4'}
cv.setWindowTitle{'win4', 'Blob'}

cv.namedWindow{'win4'}
cv.setWindowTitle{'win5', 'Detection'}

local pause = false
local key = 0
 
while true do
	
	if key == 32 then --space
		key = 0
		pause = not pause
		print("en pause")
		key=cv.waitKey{0}
	elseif key == 27 or key == 113 then --escape or q
		print("quitter")
		break
	end
	
	if not pause then

		if not(vid:read{frame}) then
			print("Fin de la video")
			key=cv.waitKey{0}
			break
		end

		Img = cv.cvtColor{frame, nil, cv.COLOR_BGR2GRAY}
		Imgcuda = Img:cuda()

		fgMaskMOG2 = pMOG2:apply{frame}
		Mask = 	torch.ByteTensor(fgMaskMOG2:size()[1],fgMaskMOG2:size()[2]):copy(fgMaskMOG2)

		cv.threshold{Mask, Mask, 100, 255, cv.THRESH_BINARY}

		erodeElement = cv.getStructuringElement{ cv.MORPH_RECT, cv.Size{4,4}}
		cv.erode{Mask, Mask, erodeElement}
		dilateElement = cv.getStructuringElement{ cv.MORPH_RECT, cv.Size{6,6}}
		cv.dilate{Mask, Mask, dilateElement}				

		Mask=Mask:apply(function(x)
				x=255-x
				return x
			end)
		
		params = cv.SimpleBlobDetector_Params{}
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
		detector = cv.SimpleBlobDetector{params}
		keypoints = detector:detect{Mask}
		ImgBlob = cv.drawKeypoints{Mask, keypoints}

		for k=1,keypoints.size do
			x = keypoints.data[k].pt.x
			y = keypoints.data[k].pt.y
			if y+L/2-1<width and y-L/2>0 and x-l/2>0 and x+l/2-1<length then
				cv.rectangle{Img, pt1={x-l/2, y-L/2}, pt2={x+l/2-1, y+L/2-1}, color = {255,255,255}}
				sub = torch.CudaTensor(1,L,l):copy(Imgcuda:sub(y-L/2,y+L/2-1,x-l/2,x+l/2-1))
				predicted = net:forward(sub:view(1,L,l))
				if predicted[1]==1 then
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
