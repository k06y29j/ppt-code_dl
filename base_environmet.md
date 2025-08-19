这是在Linux环境下

1.创建conda下的隔离环境
conda create -n learn python=3.8

激活：source activate learn

2.基于支持的CUDA安装torch
官网：k06y29j/project1
这里选择采用：
conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=12.1 -c pytorch -c nvidia

3.安装相应包
pip install -r requirement.txt
