# upscale
upscale_fake
conda create --name hat python=3.8 -y
conda activate hat
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge
cd HAT/
pip install -r requirements.txt
python setup.py develop
