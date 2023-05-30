# Insightface-wav2lip

## Environment

python 3.8.16

torch 2.0.1 (cpu)

You can run  ```pip install -r requirements.txt``` to install the environment

### Weights

1. You can download the checkpoints of InsightFace2D106，pytorch_resnet50，wav2lip，YolovV5Face form link (https://pan.baidu.com/s/1Uv8s4ESpbama6oYaLXmpzQ?pwd=92mb) . And then you should put them in the directory  ```root/modelhub```
2. You can download the checkpoint of DAGAN from https://github.com/harlanhong/CVPR2022-DaGAN or https://pan.baidu.com/s/1mE92MhHm8r24Z22qtrVNNw?pwd=ghj3 
3. . And then you should put ```depth.pth```, ```encoder.pth```, ``` SPADE_DaGAN_vox_adv_256.pth.tar``` to the directory ```root/weights```

## Implementation detail

1. You should place the driving video in the ```root/tools/chsy```
2. You should place the texts in the ```root/text.txt```
3. You can start your own inference by just run ```python main.py```
4. You can evaluate the result by just run ```python evaluate.py```
5. Inference results are place in the directory ```root/result```
6. The generated audios are place in the ```root/outputs```



```
@INPROCEEDINGS{9879781,
  author={Hong, Fa-Ting and Zhang, Longhao and Shen, Li and Xu, Dan},
  booktitle={2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)}, 
  title={Depth-Aware Generative Adversarial Network for Talking Head Video Generation}, 
  year={2022},
  volume={},
  number={},
  pages={3387-3396},
  doi={10.1109/CVPR52688.2022.00339}}
```

```
@ARTICLE{9449988,
  author={Deng, Jiankang and Guo, Jia and Yang, Jing and Xue, Niannan and Kotsia, Irene and Zafeiriou, Stefanos},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={ArcFace: Additive Angular Margin Loss for Deep Face Recognition}, 
  year={2022},
  volume={44},
  number={10},
  pages={5962-5979},
  doi={10.1109/TPAMI.2021.3087709}}
```



