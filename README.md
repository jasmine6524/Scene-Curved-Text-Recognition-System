## An Assistance System for the Visually Impaired with Curved Text Detection and Recognition

### PART I: PGNet

#### Download

下载PGNet_models(下载地址: 链接: https://pan.baidu.com/s/1rApX0GnLr_qX8t92uDVyIg 提取码: y3sy)

#### PGNet_Without_OpenVino

基于paddlepaddle实现的PGNet模型，没有使用OpenVino进行推理加速

##### Directories

- imgs: 存放待识别的包含弯曲文字的jpg图片
- models: PGNet的推理模型(将云盘中models/PGNet/paddle_model中的模型放在该文件夹下)
- results: 存放识别结果，包括三个文件(results.txt、results.jpg、results.mp3)

##### Files

- infer_without_openvino.py: PGNet推理代码。运行该文件会使用PGNet识别imgs文件夹的所有图片
- server.py: 基于flask将PGNet部署在服务器的相关代码。运行该文件，设备(个人计算机、云服务器、工作站等)会作为服务器提供弯曲文字识别服务，等待客户端(个人计算机、移动设备等)请求服务。
- w2v.py: 文字转语音代码。

#### PGNet_OpenVino_Onnx

基于onnx格式的PGNet模型，使用OpenVino进行推理加速

##### Directories

- imgs: 存放待识别的包含弯曲文字的jpg图片
- onnx_model: onnx格式的PGNet推理模型(将云盘中models/PGNet/onnx_model中的模型放在该文件夹下)
- results: 存放识别结果，包括三个文件(results.txt、results.jpg、results.mp3)

##### Files

- infer_with_openvino_onnx.py: onnx格式的PGNet模型的推理代码。运行该文件会使用PGNet识别imgs文件夹的所有图片。
- server.py: 基于flask将onnx格式的PGNe推理模型部署在服务器上的相关代码。运行该文件，设备(个人计算机、云服务器、工作站等)会作为服务器提供弯曲文字识别服务，等待客户端(个人计算机、移动设备等)请求服务。
- w2v.py: 文字转语音代码。

#### PGNet_OpenVino_Xml

基于xml格式的PGNet模型，使用OpenVino进行推理加速

##### Directories

- imgs: 存放待识别的包含弯曲文字的jpg图片
- xml_model: xml格式的PGNet推理模型(将云盘中models/PGNet/xml_model中的模型放在该文件夹下)
- results: 存放识别结果，包括三个文件(results.txt、results.jpg、results.mp3)

##### Files

- infer_with_openvino_xml.py: xml格式的PGNet模型的推理代码。运行该文件会使用PGNet识别imgs文件夹的所有图片。
- server.py: 基于flask将xml格式的PGNe推理模型部署在服务器上的相关代码。运行该文件，设备(个人计算机、云服务器、工作站等)会作为服务器提供弯曲文字识别服务，等待客户端(个人计算机、移动设备等)请求服务。
- w2v.py: 文字转语音代码。

#### PGNet_Client

客户端(个人计算机)相关代码，用于请求部署在服务器上的PGNet的弯曲文字识别服务

##### Directory

- imgs: 存放客户端待识别的jpg图片

##### File

- client.py: 运行该代码，可向同局域网下的已部署了PGNet的服务器请求弯曲文字识别服务。

### PART III: Mobile Application

基于App inventor（https://appinventor.mit.edu/）开发的安卓应用程序，整合了多种服务，包括弯曲文字识别、目标检测、人脸识别、货币识别和语音识别等。

##### App Interface

<img src="./imgs/2.JPG" alt="IMG_1128" style="zoom:20%;" />



##### Download Links

下载链接: https://pan.baidu.com/s/1rApX0GnLr_qX8t92uDVyIg 提取码: y3sy

- aia file：可以使用App Inventor打开app.aia进行app的二次设计与开发（models/App/app.aia）

- apk file：可以直接安装在安卓手机，结合上述部署PGNet的服务器代码进行配合使用