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

function write(path, data, sep)
    sep = sep or ';'
    local file = assert(io.open(path, "w"))
    for i=1,#data do
        for j=1,#data[i] do
            if j>1 then file:write(sep) end
            file:write(data[i][j])
        end
        file:write('\n')
    end
    file:close()
end

local nt = 10
local n1 = 212
local n2 = 600
local N = n1 + n2
local l = 60
local L = 120

local vid = cv.VideoCapture{vidname}

if not vid:isOpened() then
    print("Failed to open the video")
    os.exit(-1)
end

local fps = vid:get{cv.CAP_PROP_FPS}
local cptframe = 0
local oldtps = 0
local tps = 0
data ={{'Temps','Nombre de Motos'}}

local pMOG2 = cv.BackgroundSubtractorMOG2{}
local _, frame = vid:read{}
for i=1,30 do
	if not(vid:read{frame}) then
		break
	end
	cptframe = cptframe + 1
	fgMaskMOG2 = pMOG2:apply{frame}
end


local width = frame:size()[1]
local length = frame:size()[2]
local Img = torch.Tensor(width,length):zero()
Img = cv.cvtColor{frame, nil, cv.COLOR_BGR2GRAY}
local Imgpred = torch.Tensor(width,length):zero()
Imgpred = cv.cvtColor{frame, nil, cv.COLOR_BGR2GRAY}

local CoordTrack = {}
local NTrack = 0
local VTrack = 5
local TVTab = 5
local VTab = {}
for i=1,TVTab do VTab[i] = VTrack end

local cpt = 0
local cptglb = 0

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
		cptframe = cptframe + 1

		local Img = cv.cvtColor{frame, nil, cv.COLOR_BGR2GRAY}
		local Imgpred = cv.cvtColor{frame, nil, cv.COLOR_BGR2GRAY}

		local fgMaskMOG2 = pMOG2:apply{frame}
		local Mask = torch.ByteTensor(fgMaskMOG2:size()[1],fgMaskMOG2:size()[2]):copy(fgMaskMOG2)

		cv.threshold{Mask, Mask, 100, 255, cv.THRESH_BINARY}

		local erodeElement = cv.getStructuringElement{ cv.MORPH_RECT, cv.Size{4,4}}
		cv.erode{Mask, Mask, erodeElement}
		local dilateElement = cv.getStructuringElement{ cv.MORPH_RECT, cv.Size{5,5}}
		cv.dilate{Mask, Mask, dilateElement}				

		Mask = Mask:apply(function(x)
				x=255-x
				return x
			end)
		
		local params = cv.SimpleBlobDetector_Params{}
		-- Change thresholds
		params.minThreshold = 0
		params.maxThreshold = 255
		-- Filter by Area
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

		local CoordPredicted = torch.Tensor(keypoints.size,2):zero()
		local NPredicted = 0

		for k=1,keypoints.size do
			local x = keypoints.data[k].pt.x
			local y = keypoints.data[k].pt.y
			if y+L/2-1<width and y-L/2>1 and x-l/2>1 and x+l/2-1<length then
				local sub = torch.Tensor(1,L,l):copy(Imgpred:sub(y-L/2,y+L/2-1,x-l/2,x+l/2-1))
				local predicted = net:forward(sub:view(1,L,l))
				if predicted[1]==1 then
					NPredicted = NPredicted + 1
					CoordPredicted[NPredicted][1] = x
					CoordPredicted[NPredicted][2] = y
					cv.rectangle{frame, pt1={x-l/2, y-L/2}, pt2={x+l/2-1, y+L/2-1}, color = {0,255,0}}
					cv.rectangle{frame, pt1={x-2, y-2}, pt2={x+2, y+2}, color = {0,255,0}}
				end
				cv.rectangle{Img, pt1={x-l/2, y-L/2}, pt2={x+l/2-1, y+L/2-1}, color = {255,255,255}}
			end
		end

		local Coordchange = {}
		for i=1,NTrack do Coordchange[i] = false end

		for i=1,NPredicted do
			local new = true
			for j=1,NTrack do
				if math.abs(CoordTrack[j][1]-CoordPredicted[i][1])<30 and math.abs(CoordTrack[j][2]-CoordPredicted[i][2])<30 then
					new = false
					if math.abs(CoordTrack[j][2]-CoordPredicted[i][2])<15 then
						table.remove(VTab,1)
						table.insert(VTab,math.abs(CoordTrack[j][2]-CoordPredicted[i][2]))
						m=0
						for a=1,TVTab do
							m=m+VTab[a]
						end
						VTrack=m/TVTab
					end
					CoordTrack[j][1] = CoordPredicted[i][1]
					CoordTrack[j][2] = CoordPredicted[i][2]
					CoordTrack[j][3] = CoordTrack[j][3] + 1
					Coordchange[j] = true
				end
			end
			if new then
				NTrack = NTrack + 1
				table.insert(CoordTrack,{CoordPredicted[i][1],CoordPredicted[i][2],1})
			end
		end

		for j=1,NTrack do
			if not(Coordchange[i]) then
				if CoordTrack[j][1]-l/2>1 and CoordTrack[j][2]-L/2>1 and CoordTrack[j][1]+l/2-1<length and CoordTrack[j][2]+L/2-1<width then
					local sub = torch.Tensor(1,L,l):copy(Imgpred:sub(CoordTrack[j][2]-L/2,CoordTrack[j][2]+L/2-1,CoordTrack[j][1]-l/2,CoordTrack[j][1]+l/2-1))
					local predicted = net:forward(sub:view(1,L,l))
					if predicted[1]==1 then
						cv.rectangle{frame, pt1={CoordTrack[j][1]-2, CoordTrack[j][2]-2}, pt2={CoordTrack[j][1]+2, CoordTrack[j][2]+2}, color = {0,255,0}}
						CoordTrack[j][3] = CoordTrack[j][3] + 1
					end
				end
			end
		end

		local j = 1
		while j<=NTrack and j>0 do
			local k = 1
			while k<=NTrack and k>0 and j>0 do
				if k~=j then
					if CoordTrack[j][2] == CoordTrack[k][2] then
						table.remove(CoordTrack,k)
						NTrack = NTrack-1
						j = j-1
						k = k-1
					end
				end
				k = k+1
			end
			j = j+1
		end

		j=1
		while j<=NTrack and j>0 do
			if CoordTrack[j][1]-l/2>1 and CoordTrack[j][2]-L/2>1 and CoordTrack[j][1]+l/2-1<length and CoordTrack[j][2]+L/2-1<width then
				cv.rectangle{frame, pt1={CoordTrack[j][1]-l/2, CoordTrack[j][2]-L/2}, pt2={CoordTrack[j][1]+l/2-1, CoordTrack[j][2]+L/2-1}, color = {255,0,0}}
				CoordTrack[j][2] = CoordTrack[j][2]+VTrack
			else
				if CoordTrack[j][3] > 5 then
					cpt = cpt+1
					cptglb = cptglb+1
				end
				table.remove(CoordTrack,j)
				NTrack = NTrack-1
				j = j-1
			end
			j = j+1
		end

		if cptframe>=fps then
			cptframe = 0
			tps = tps + 1
			if tps%(6*60)==0 and tps~=0 then
				table.insert(data,{math.floor(oldtps/60) .. ':00' .. '-' .. math.floor(tps/60) .. ':00',cpt})
				oldtps = tps
				cpt = 0
			end
		end

		cv.rectangle{frame, pt1={0, 0}, pt2={60, 20}, color = {255,255,255},thickness=-1}
		cv.putText{frame,string.format('%02d',math.floor(tps/60)) .. ':' .. string.format('%02d',tps%60),{3,16},cv.FONT_HERSHEY_SIMPLEX,0.6,{0,0,0},2}
		cv.rectangle{frame, pt1={length-290, 0}, pt2={length, 20}, color = {255,255,255},thickness=-1}
		if cptglb>1 then
			cv.putText{frame,string.format('Traqumoto compte %d motos',cptglb),{length-290,16},cv.FONT_HERSHEY_SIMPLEX,0.6,{0,0,0},2}
		else
			cv.putText{frame,string.format('Traqumoto compte %d moto',cptglb),{length-290,16},cv.FONT_HERSHEY_SIMPLEX,0.6,{0,0,0},2}
		end

		cv.imshow{'win1', Img}			
		cv.imshow{'win2', fgMaskMOG2}
		cv.imshow{'win3', Mask}
		cv.imshow{'win4', ImgBlob}
		cv.imshow{'win5', frame}
		
		key=cv.waitKey{1} --en ms
	end
end

table.insert(data,{math.floor(oldtps/60) .. ':00' .. '-' .. math.floor(tps/60) .. ':' .. string.format('%02d',tps%60),cpt})

cv.destroyAllWindows{}
