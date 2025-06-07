pip3 install torch==2.6.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install git+https://github.com/huggingface/diffusers
pip3 install ftfy imageio numpy imageio-ffmpeg GPUtil transformers accelerate
pip3 install torchao
cd SageAttention
python setup.py install
cd ..
cd ParaAttention
pip3 install packaging 'setuptools>=64' 'setuptools_scm>=8' wheel
pip3 install ninja
pip3 install -e '.[dev]' --no-build-isolation --no-use-pep517