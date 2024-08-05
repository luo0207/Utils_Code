# Utils_Code
The collection of the useful function such as plot figures and so on.
## Introduction
* 使用之前需要运行对应的bash文件安装所需要的库
### plot_figures
#### plotly_camera_pose.py: 
用于可视化相机的位姿序列 ： 输入为txt文件的文件名称，将其相机位姿按照时序可视化出来，这里txt文件每一行是12个数字，代表[R | T] flatten之后的结果。

### Transformer
#### relative_position.py
Transformer中的相对位置编码，Self-Attention with Relative Position Representations (paper)[https://arxiv.org/pdf/1803.02155]
