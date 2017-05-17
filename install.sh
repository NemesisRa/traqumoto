# KEEP UBUNTU OR DEBIAN UP TO DATE

printf "\n[Install] sudo apt-get -y update\n"
sudo apt-get -y update
printf "\n[Install] sudo apt-get -y upgrade\n"
sudo apt-get -y upgrade
printf "\n[Install] sudo apt-get -y dist-upgrade\n"
sudo apt-get -y dist-upgrade
printf "\n[Install] sudo apt-get -y autoremove\n"
sudo apt-get -y autoremove


# INSTALL THE DEPENDENCIES

# Build tools:
printf "\n[Install] sudo apt-get install -y build-essential cmake\n"
sudo apt-get install -y build-essential cmake

# GUI (if you want to use GTK instead of Qt, replace 'qt5-default' with 'libgtkglext1-dev' and remove '-DWITH_QT=ON' option in CMake):
printf "\n[Install] sudo apt-get install -y qt5-default libvtk6-dev\n"
sudo apt-get install -y qt5-default libvtk6-dev

# Media I/O:
printf "\n[Install] sudo apt-get install -y zlib1g-dev libjpeg-dev libwebp-dev libpng-dev libtiff5-dev libjasper-dev libopenexr-dev libgdal-dev\n"
sudo apt-get install -y zlib1g-dev libjpeg-dev libwebp-dev libpng-dev libtiff5-dev libjasper-dev libopenexr-dev libgdal-dev

# Video I/O:
printf "\n[Install] sudo apt-get install -y libdc1394-22-dev libavcodec-dev libavformat-dev libswscale-dev libtheora-dev libvorbis-dev libxvidcore-dev libx264-dev yasm libopencore-amrnb-dev libopencore-amrwb-dev libv4l-dev libxine2-dev\n"
sudo apt-get install -y libdc1394-22-dev libavcodec-dev libavformat-dev libswscale-dev libtheora-dev libvorbis-dev libxvidcore-dev libx264-dev yasm libopencore-amrnb-dev libopencore-amrwb-dev libv4l-dev libxine2-dev

# Parallelism and linear algebra libraries:
printf "\n[Install] sudo apt-get install -y libtbb-dev libeigen3-dev\n"
sudo apt-get install -y libtbb-dev libeigen3-dev

# Python:
printf "\n[Install] sudo apt-get install -y python-dev python-tk python-numpy python3-dev python3-tk python3-numpy\n"
sudo apt-get install -y python-dev python-tk python-numpy python3-dev python3-tk python3-numpy

# Java:
printf "\n[Install] sudo apt-get install -y ant default-jdk\n"
sudo apt-get install -y ant default-jdk

# Git:
printf "\n[Install] sudo apt-get install -y git\n"
sudo apt-get install -y git

# Documentation:
printf "\n[Install] sudo apt-get install -y doxygen\n"
sudo apt-get install -y doxygen


# INSTALL Applications
printf "\n[Install] mkdir Applications\n"
mkdir Applications
printf "\n[Install] cd Applications\n"
cd Applications

# INSTALL Torch

printf "\n[Install] git clone https://github.com/torch/distro.git ./torch\n"
git clone https://github.com/torch/distro.git ./torch
printf "\n[Install] cd torch\n"
cd torch
printf "\n[Install] bash install-deps\n"
bash install-deps
printf "\n[Install] ./install.sh\n"
./install.sh
printf "\n[Install] cd ..\n"
cd ..

# INSTALL OpenCV 3.1.0

printf "\n[Install] sudo apt-get install -y unzip wget\n"
sudo apt-get install -y unzip wget
printf "\n[Install] wget https://github.com/opencv/opencv/archive/3.1.0.zip\n"
wget https://github.com/opencv/opencv/archive/3.1.0.zip
printf "\n[Install] unzip 3.1.0.zip\n"
unzip 3.1.0.zip
printf "\n[Install] rm 3.1.0.zip\n"
rm 3.1.0.zip
printf "\n[Install] mv opencv-3.1.0 OpenCV\n"
mv opencv-3.1.0 OpenCV
printf "\n[Install] cd OpenCV\n"
cd OpenCV
printf "\n[Install] mkdir build\n"
mkdir build
printf "\n[Install] cd build\n"
cd build
printf "\n[Install] cmake -DWITH_QT=ON -DWITH_OPENGL=ON -DFORCE_VTK=ON -DWITH_TBB=ON -DWITH_GDAL=ON -DWITH_XINE=ON -DBUILD_EXAMPLES=ON ..\n"
cmake -DWITH_QT=ON -DWITH_OPENGL=ON -DFORCE_VTK=ON -DWITH_TBB=ON -DWITH_GDAL=ON -DWITH_XINE=ON -DBUILD_EXAMPLES=ON ..
printf "\n[Install] make -j4\n"
make -j4
printf "\n[Install] sudo make install\n"
sudo make install
printf "\n[Install] cd ..\n"
cd ..

# INSTALL Luarocks
printf "\n[Install] sudo apt-get install luarocks\n"
sudo apt-get install luarocks

# LINK Torch & OpenCV

printf "\n[Install] git clone https://github.com/VisionLabs/torch-opencv.git\n"
git clone https://github.com/VisionLabs/torch-opencv.git
printf "\n[Install] cd torch-opencv\n"
cd torch-opencv
printf "\n[Install] luarocks make cv-scm-1.rockspec\n"
luarocks make cv-scm-1.rockspec
printf "\n[Install] source .bashrc\n"
source .bashrc
printf "\n[Install] cd ..\n"
cd ..

# INSTALL Luastatic
printf "\n[Install] git clone https://github.com/ers35/luastatic\n"
git clone https://github.com/ers35/luastatic

printf "\n[Install] cd ..\n"
cd ..


# Création de l'exécutable
printf "\n[Install] th Applications/luastatic/luastatic.lua src/Traqumoto.lua Applications/torch/build/exe/luajit-rocks/luajit-2.1/libluajit-static.a -IApplications/torch/exe/luajit-rocks/luajit-2.1/src/\n"
th Applications/luastatic/luastatic.lua src/Traqumoto.lua Applications/torch/build/exe/luajit-rocks/luajit-2.1/libluajit-static.a -IApplications/torch/exe/luajit-rocks/luajit-2.1/src/