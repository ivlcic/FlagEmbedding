python3.12 -m venv venv
source venv/bin/activate
pip install wheel
pip install -e .\[finetune\]

wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
dpkg -i cuda-keyring_1.1-1_all.deb
apt-get update
apt-get install cuda-toolkit-12-4 vim tree

git clone https://github.com/ivlcic/FlagEmbedding.git fe
cd fe
python -m venv venv
source venv/bin/activate
pip install wheel
pip install packaging
pip install torch torchvision torchaudio
pip install -e .\[finetune\]
