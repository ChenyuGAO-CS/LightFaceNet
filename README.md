### LightFaceNet_TF

Tensorflow implementation for MobileFaceNet, SqueezeFaceNet and ShuffleFaceNet.

## dependencies

- tensorflow >= r1.2
- opencv-python 3.x
- python 3.x
- scipy
- sklearn
- numpy
- mxnet
- pickle

## Prepare dataset

1. choose one of The following links to download dataset which is provide by insightface. (Special Recommend MS1M)
* [Refined-MS1M@BaiduDrive](https://pan.baidu.com/s/1nxmSCch), [Refined-MS1M@GoogleDrive](https://drive.google.com/file/d/1XRdCt3xOw7B3saw0xUSzLRub_HI4Jbk3/view)
* [VGGFace2@BaiduDrive](https://pan.baidu.com/s/1c3KeLzy), [VGGFace2@GoogleDrive](https://drive.google.com/open?id=1KORwx_DWyIScAjD6vbo4CSRu048APoum)
2. move dataset to ${MobileFaceNet_TF_ROOT}/datasets.
3. run ${MobileFaceNet_TF_ROOT}/utils/data_process.py.

## training

1. refined super parameters by yourself special project.
2. run script
'''${MobileFaceNet_TF_ROOT}/train_nets.py'''
3. have a snapshot result at ${MobileFaceNet_TF_ROOT}/output.

## testing

1. run script
'''${MobileFaceNet_TF_ROOT}/inference.py'''
2. the best model of SqueezeFaceNet_TF at ${MobileFaceNet_TF_ROOT}/output/ckpt_best/squeezefacenet_best_ckpt.
3. [download the model](https://pan.baidu.com/s/1lYOX0DNt7c7jB5KSi99uVA)
key of the download link：lpze

## MobileFaceNet_TF performance

|  size  | LFW(%) | Val@1e-3(%) | inference@MSM8976(ms) |
| ------ | ------ | ----------- | --------------------- |
|  5.7M  | 99.25+ |    96.8+    |          260-         |

## SqueezeFaceNet_TF performance

 | Size:112x112 |   Accuracy  |                 FPS                |
 | ------------ | ----------- | ---------------------------------- |
 |     LFW      |   97.2      |            BatchSize = 1           |
 |    Cfp_FF    |   96.2      |         Model: squeezenet          |
 |    Cfp_FP    |   78.5      |       (i7-5930K @ 3.50~3.70GHz)    |
 |   Agedb_30   |   83.7      |                83.3                |


## References

1. [facenet](https://github.com/davidsandberg/facenet)
2. [InsightFace mxnet](https://github.com/deepinsight/insightface)
3. [InsightFace_TF](https://github.com/auroua/InsightFace_TF)
4. [MobileFaceNets: Efficient CNNs for Accurate Real-Time Face Verification on Mobile Devices](https://arxiv.org/abs/1804.07573)
5. [CosFace: Large Margin Cosine Loss for Deep Face Recognition](https://arxiv.org/abs/1801.09414)
6. [InsightFace : Additive Angular Margin Loss for Deep Face Recognition](https://arxiv.org/abs/1801.07698)
7. [SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size](http://arxiv.org/abs/1602.07360)
