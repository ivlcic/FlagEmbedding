python3.12 -m venv venv
source venv/bin/activate
pip install wheel
pip install -e .\[finetune\]

wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
dpkg -i cuda-keyring_1.1-1_all.deb
apt-get update
apt-get install cuda-toolkit-12-4 vim tree

apt install vim tree nvtop
git clone https://github.com/ivlcic/FlagEmbedding.git fe
cd fe
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install wheel
pip install packaging
pip install torch torchvision torchaudio
pip install -e .\[finetune\]


scp -P 17352 ~/projects/emma/data/mulabel/split/bge_m3_mulabel_sl_p1_s0_hn.jsonl root@149.7.4.9:/root/data/mulabel/
scp -P 47168 data/mulabel/split/bge_m3_mulabel_sl_p1_s0_hn.jsonl root@185.65.93.212:/root/data/mulabel/
