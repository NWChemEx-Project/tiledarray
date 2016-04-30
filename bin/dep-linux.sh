#! /bin/sh

# Exit on error
set -ev

# Add PPA for a newer version GCC
sudo add-apt-repository ppa:ubuntu-toolchain-r/test -y
# Add PPA for newer cmake (3.2.3)
sudo add-apt-repository ppa:george-edison55/precise-backports -y
# Add repo with newer clang
sudo add-apt-repository 'deb http://llvm.org/apt/precise/ llvm-toolchain-precise-3.7 main'
sudo wget -O - http://llvm.org/apt/llvm-snapshot.gpg.key|sudo apt-key add -

# Update package list
sudo apt-get update -qq

# Install updated gcc compilers
sudo apt-get install -qq -y gcc-$GCC_VERSION g++-$GCC_VERSION
if [ "$CXX" = "g++" ]; then
    export CC=/usr/bin/gcc-$GCC_VERSION
    export CXX=/usr/bin/g++-$GCC_VERSION
fi

# Print compiler information
$CC --version
$CXX --version

# Install very recent CMAKE
#curl -O https://cmake.org/files/v3.4/cmake-3.4.1.tar.gz
#tar -xzf cmake-3.4.1.tar.gz
#cd ./cmake-3.4.1
#./configure --prefix=/usr/local
#make -j2
#sudo make install

# Install CMake 3
sudo apt-get -y -qq --no-install-suggests --no-install-recommends --force-yes install cmake cmake-data
cmake --version

# Install the rest
sudo apt-get install -qq -y libblas-dev liblapack-dev mpich2 libtbb-dev libeigen3-dev libboost1.48-dev clang-3.7
