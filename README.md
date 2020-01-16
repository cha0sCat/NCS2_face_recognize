# NCS2_Face_Recognize
## 简介
本程序用于识别摄像头实时回传图像中的人脸，并与
已经记录的人脸做比对  
如果比对成功则触发HOOK，通知开发者  

程序支持保存未知和已知人脸图像，并在检测后实时将新的人脸信息录入数据集

因此就算刚开始运行时没有任何人脸数据，程序也可以自动记录识别的所有人脸信息，方便后续人工设定每张人脸对应的人名

## 依赖
1. 安装OPENVINO套件的系统
2. NCS1/NCS2 神经计算棒
3. 摄像头套件

## 程序启动
`python3 main.py`

## 程序文件说明
### main.py
这是程序启动入口，包括了程序的主要运行逻辑

### facedect.py
这是用于检测图片中人脸位置的神经网络  
可以直接通过 `network = FaceDetect()`进行初始化

### facerecognize.py
#### detectFace函数：  
将facedect输出的人脸部分图片进行特征识别，返回一个512点的维list  

#### 特别说明
FaceRecognize类使用的是OPENVINO套件  
但是因为会和FaceDetect(使用cv2套件)抢NCS设备   
所以如果你想同时使用FaceDetect和FaceRecognize  
那么就使用 cv2FaceRecognize类

## utils.py
一大堆小工具  
包括：
 - 图像处理
 - 函数运行计时
 - 人脸与已知人脸比对
 - 读取已知人脸数据集
 - 随机字符生成
 - 保存图像
 
 faceMatch 计算非常耗时，所以可以使用电脑进行识别，返回识别结果  
 服务端代码详见server
 
## traning.py
在程序跑了一段时间以后，images文件夹里面会多出各种人的人脸图片  
这时候你就可以对每个人的文件夹进行命名，然后运行学习程序，  
这样程序就会把这些人的人脸数据记录到已知人脸里了
