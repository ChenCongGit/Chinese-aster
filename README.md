# Chinese-aster
该项目将自然场景文本识别网络ASTER([here](https://ieeexplore.ieee.org/abstract/document/8395027/))应用于中英文钢铁订单标签信息识别

## 项目介绍
钢铁订单标签识别项目主要是对下图这种常见的钢铁领域订单标签信息进行识别，如公司名、钢卷号、日期等。整个项目的主要流程包括钢铁订单标签矫正，标签行文本检测，行文本识别，文本信息分类。
![](/相关图片/钢铁订单.jpg "钢铁订单标签")
![](/相关图片/钢铁订单标签识别流程.jpg "标签识别项目主要流程")

我们共有1500张钢铁订单标签图像，其中450张用作检测数据集，通过检测网络得到大约18000张行文本图像，我们挑选其中的15000张能够辨识的作为ASTER识别网络数据集。另外的1050张订单标签图像用作最终整个OCR系统的测试。

## Chinese-aster介绍
ASTER(ASTER: An Attentional Scene Text Recognizer with Flexible Rectification)是2018年提出的识别精度较好的行文本识别网络，在多个公共数据集上都拥有非常出色的精度。该项目将ASTER网络应用于钢铁标签中英文行文本识别，通过在300k合成订单标签数据集上预训练，在15k真实钢铁订单标签图像微调，在15k数据集上测试准确率达到94.40%(考虑大小写差异)，94.61%(忽略大小写)，在训练数据不足的情况下，实现了较高准确率。下图是一些行文本图像:
![](/相关图片/行文本图像300k合成数据集.jpg "行文本图像(300k合成数据集)")
![](/相关图片/行文本图像15k真实数据集.jpg "行文本图像(15k真实数据集)")

该项目订单中的行文本相比于公共数据集文本具有以下特点:
 - 更加接近于印刷体文本，行文本排列比较整齐，相对而言更容易识别
 - 中英文符号文本混合，因此具有相当大的类别(字典包括17个常见可打印符号，10个数字，52个大小写英文字母，1062个钢铁订单中出现过的的中文字符)
 - 训练数据集过小，我们需要人工制做标签，有很大的难度，可用的样本只有15000张，相对于1141个分类类别，显然非常容易过拟合
 - 行文本大小、宽高比等变化范围大，宽高比范围从1到22的范围，这增加了识别的难度

对于以上问题，我们对原始模型及代码作出以下的一些改进:
 - 根据订单文本特点合成了大约300k行文本图像，总体上与真实订单还是很接近的，用其进行预训练，克服类别多容易过拟合的问题
 - 对图像进行预处理，所有宽高比小于8的行文本图像通过numpy的pad函数中心填充到宽高比等于8，填充的像素值取图像的像素均值与像素值极差的加权和(不超过255);宽高比大于8的行文本图像直接resize到宽高比等于8,保证图像不会过份变。
 - 修改ASTER模型中的相关程序，在制作tfrecord的时候将文本行按字符切分，将单个字符转化为utf-8编码，解决无法识别中文字符的问题
 - 添加随机旋转、随机调整亮度对比度等数据增强，有效的提升了识别准确率，减小过拟合
 - 由于行文本排列基本都是规则文本，并且避免模型过大，省去ASTER中的STN矫正网络部分

## 模型训练过程
我们首先在300k合成数据集上预训练大约13000步，得到预训练模型，然后在此基础上在15k真实数据集上微调大约6000步，得到最终的训练模型。

model | 300ktrain(0-3000) | 300ktest | 15ktrain(0-3000) | 15ktest
:-: | :-: | :-: | :-: | :-:
13180 | 92.93% | 91.89% | - | 67.60% |
19029 | - | - | 98.47% | 94.40% |

预训练和微调batch size大小都设为128，预训练学习率保持为1e-1，微调阶段学习率13k-15k为1e-1，15k-18k为0.5e-1，18k-20k为1e-2。

## 代码程序
### Prerequisites
ASTER was developed and tested with **TensorFlow r1.4**. Higher versions may not work.

ASTER requires [Protocol Buffers](https://github.com/google/protobuf) (version>=2.6). Besides, in Ubuntu 16.04:
```
sudo apt install cmake libcupti-dev
pip3 install --user protobuf tqdm numpy editdistance
```

### Installation
  1. Go to `c_ops/` and run `build.sh` to build the custom operators
  2. Execute `protoc Chinese_aster/protos/*.proto --python_out=.` to build the protobuf files
  3. Add `/path/to/Chinese_aster` to `PYTHONPATH`, or set this variable for every run

## 推理和测试
运行demo.py文件，它将对文件夹中的所有图片进行识别，并在命令行中返回识别结果，并计算识别准确率。
```
python Chinese_aster/demo.py \
  --exp_dir Chinese_aster/experiments/demo/ \
  --input_image_dir Chinese_aster/ocr_dataset/new_ocr_test/ \
  --check_num 19029 \
  --padding True \
  --pad_threshold 8
```

## 训练
首先通过tools/create_ic15_tfrecord.py和create_synth_chinese_tfrecord.py将训练数据集转化为tfrecord格式。
```
python Chinese_aster/tools/create_ic15_tfrecord.py \
  --data_dir Chinese_aster/ocr_dataset/new_ocr_train/ \
  --output_path Chinese_aster/ocr_dataset/ \
  --padding True \
  --pad_threshold 8
```
运行create_synth_chinese_tfrecord.py将会把300k预训练数据集随机划分为训练集与测试集，将训练集写为tfrecord，测试集保存为ICDAR2015格式。
```
python Chinese_aster/tools/create_synth_chinese_tfrecord.py \
  --data_dir Chinese_aster/ocr_dataset/new_ocr_train/ \
  --output_path Chinese_aster/ocr_dataset/ \
  --test_data_dir Chinese_aster/ocr_dataset/synth_test_ocr/ \
  --padding True \
  --pad_threshold 8
```
运行Chinese_aster/train.py文件进行训练，相关训练参数，在Chinese_aster/experiments/demo/config/ocr_train.prototxt中设置。
```
python Chinese_aster/train.py \
  --exp_dir Chinese_aster/experiments/demo \
  --num_clones 2
```

## Citation

If you find this project helpful for your research, please cite the following papers:

```
@article{bshi2018aster,
  author  = {Baoguang Shi and
               Mingkun Yang and
               Xinggang Wang and
               Pengyuan Lyu and
               Cong Yao and
               Xiang Bai},
  title   = {ASTER: An Attentional Scene Text Recognizer with Flexible Rectification},
  journal = {IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  volume  = {}, 
  number  = {}, 
  pages   = {1-1},
  year    = {2018}, 
}

@inproceedings{ShiWLYB16,
  author    = {Baoguang Shi and
               Xinggang Wang and
               Pengyuan Lyu and
               Cong Yao and
               Xiang Bai},
  title     = {Robust Scene Text Recognition with Automatic Rectification},
  booktitle = {2016 {IEEE} Conference on Computer Vision and Pattern Recognition,
               {CVPR} 2016, Las Vegas, NV, USA, June 27-30, 2016},
  pages     = {4168--4176},
  year      = {2016}
}
```

IMPORTANT NOTICE: Although this software is licensed under MIT, our intention is to make it free for academic research purposes. If you are going to use it in a product, we suggest you [contact us](xbai@hust.edu.cn) regarding possible patent issues.
