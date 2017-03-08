apt-get install libboost-log-dev libboost-test-dev
CUDA_REPO_PKG=cuda-repo-ubuntu1404_7.5-18_amd64.deb
wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1404/x86_64/$CUDA_REPO_PKG
dpkg -i $CUDA_REPO_PKG
rm $CUDA_REPO_PKG

apt-get -y update

# install packages
CUDA_PKG_VERSION="7-5"
CUDA_VERSION="7.5"
apt-get install -y --no-install-recommends \
cuda-core-$CUDA_PKG_VERSION \
cuda-cudart-dev-$CUDA_PKG_VERSION \
cuda-cublas-dev-$CUDA_PKG_VERSION \
cuda-curand-dev-$CUDA_PKG_VERSION
# manually create CUDA symlink
ln -s /usr/local/cuda-$CUDA_VERSION /usr/local/cuda