# Baidu face_det_v3 Download Tool
Attributes ground truth from [Baidu API](
https://ai.baidu.com/ai-doc/FACE/yk37c1u4t).

## Data Format
| Index | Data Field | Data Type | Description                                                  |
| ----- | ---------- | --------- | ------------------------------------------------------------ |
| 0+1   | class_id   | int       | 物体的类别。                                                 |
| 1+4   | location   | float[4]  | 边界框的xyxy归一化格式。                                     |
| 5+10  | landmarks  | float[10] | 人脸的xy归一化关键点。选择的5个关键点分别是：左眼(21)，右眼(38)，鼻尖(57)，左嘴角(58)，右嘴角(62)。 |
| 15+2  | eye_status | float[2]  | 睁眼程度，1表示睁开眼睛，0表示闭眼。                         |
| 17+7  | occlusions | float[7]  | 左眼、右眼、鼻子、嘴巴、左脸颊、右脸颊、下巴的遮挡程度。     |
| 24+3  | quality    | float[3]  | 图像质量的信息，包括模糊度（1=更模糊）、光照（1=好）、完整度（1=完整）。 |
| 27+1  | age        | float     | 年龄的信息，映射到区间 [0.25, 0.75]，其中0.25对应0岁，0.75对应100岁。 |
| 28+1  | gender     | float     | 性别的信息，1表示男性，0表示女性。                           |
| 29+1  | glasses    | float     | 戴眼镜的概率，1表示戴眼镜，0表示未戴眼镜。                   |
| 30+1  | mask       | float     | 戴口罩的概率，1表示戴口罩，0表示未戴口罩。                   |


## 生成步骤
```shell
cd baiduface;
python download.py  # 按api下载
python convert.py   # 转换label格式
python split.py     # 切分训练/测试集
```