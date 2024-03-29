# Ro-SOS-Metric-Expression-Network-MEnet-for-Robust-Salient-Object-Segmentation
codes for the paper "Ro-SOS: Metric Expression Network (MEnet) for Robust Salient Object Segmentation"

## MEnet 

This repository contains a Dockerfile and scripts to build and run our Metric Expression Network for Salient Object Segmentation in Docker containers. We also provide some example data to test the networks. 


If you use this project or parts of it in your research, please cite the original paper of MEnet:

    @misc{zeng2018rosos,
    title={Ro-SOS: Metric Expression Network (MEnet) for Robust Salient Object Segmentation},
    author={Delu Zeng and Yixuan He and Li Liu and Zhihong Chen and Jiabin Huang and Jie Chen and John Paisley},
    year={2018},
    eprint={1805.05638},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
    }


See the [paper website](https://arxiv.org/pdf/1805.05638.pdf) for more details.

## 1 Requirements
caffe:gpu,
pytorch v1.3.0 (for robustness experiment)


## 2 Train
First, you can use your own datasets to train the MEnet, and the train_data_dir is './0408data/Images_train',
train_labels_dir is './0408data/GT_train'.

After that, you can start training with:
        $python ./tools/training.py 

Models will be saved in './model/snapshot/' every 10000 iters, you can get the final model and solverstate which named 'attention_iter_110000.caffemdel' and 'attention_iter_110000.solverstate' in the same path.

## 3 Test
First, you can change the path 'GV.test_dir' in testing.py where you want to put your test images,
After that, you can start testing with:
        $python ./tools/testing.py 

Test results will be saved in this dir.

## 4 Robustness experiment
### put the checkpoint in pytorch format from:
    https://drive.google.com/open?id=1-6pHXUR73eTTTYss_uqmgie-wfAkEPtx
### to
'ckpt/pytorch_model.ckpt'
### and fulfill the datasets used in the robustness experiment by official releases in corresponding subfolders of
'hong/datasets/'
### then run the robustness experiment of MEnet with pytorch v1.3.0 with which MEnet is re-implemented
$python hong/nb42_MEnet_main_test.py


