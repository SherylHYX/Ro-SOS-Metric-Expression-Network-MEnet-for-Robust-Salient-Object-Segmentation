# %cd '/mnt/pub_workspace_2T/hong_data/detect/MEnet/drive'
import os
# from subprocess import check_output

if not os.path.exists('../../MEnet'):
    os.system('git clone https://github.com/scutDACIM/MEnet.git')

# %cd '/mnt/pub_workspace_2T/hong_data/detect/MEnet/drive/MEnet'
os.system('pip install -e ..')

# !cat /etc/os-release
# !apt install caffe-cuda
# !apt install caffe-cpu
