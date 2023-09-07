# YOLOv5-and-image-classification-in-the-NCNN
## first we need install package
Vulkan https://developer.nvidia.com/vulkan-driver \
ncnn  https://github.com/Tencent/ncnn/releases install ncnn-20230816-windows-vs2022.zip\
opencv https://opencv.org/releases/ opencv4.7.0\
## second open ncnn.sln
use Release x64 to run project\
navigate to "Configuration Properties" -> "C/C++" -> "General". Add the path to the "Include Directories" field \
"yourPath"\Vulkan\Include \
"yourPath"\ncnn-20230816-windows-vs2022\x64\include \
"yourPath"\opencv\opencv\build\include \
In the same Properties window, "Library" field \
"yourPath"\opencv\opencv\build\x64\vc16\lib \
"yourPath"\Vulkan\Lib \
"yourPath"\ncnn-20230816-windows-vs2022\x64\lib \
navigate to "Configuration Properties" -> "Linker" -> "Input"\
dxcompiler.lib\
GenericCodeGen.lib\
glslang-default-resource-limits.lib\
glslang.lib\
HLSL.lib\
MachineIndependent.lib\
OGLCompiler.lib\
OSDependent.lib\
shaderc.lib\
shaderc_combined.lib\
shaderc_shared.lib\
shaderc_util.lib\
spirv-cross-c-shared.lib\
spirv-cross-c.lib\
spirv-cross-core.lib\
spirv-cross-cpp.lib\
spirv-cross-glsl.lib\
spirv-cross-hlsl.lib\
spirv-cross-msl.lib\
spirv-cross-reflect.lib\
spirv-cross-util.lib\
SPIRV-Tools-diff.lib\
SPIRV-Tools-link.lib\
SPIRV-Tools-lint.lib\
SPIRV-Tools-opt.lib\
SPIRV-Tools-reduce.lib\
SPIRV-Tools-shared.lib\
SPIRV-Tools.lib\
SPIRV.lib\
SPVRemapper.lib\
vulkan-1.lib\
ncnn.lib\
opencv_world470.lib\
opencv_world470d.lib\
`Different download versions may result in a chance of missing or adding libraries. Please check it yourself` \
`The project comes with 5 pre-trained models `\
`shape, which is an object detection model for detecting the positions of basic shapes in images `\
`resShape, which is an image classification model for classifying basic shapes `\
`carcard, which is an object detection model for detecting the positions of license plates `\
`main, which is an object detection model with 11 classes `\
`and color, which is an image classification model for classifying colors `
```testCode
//we can use ResNet or Yolov5 or Yolov8 to create model.
ResNet model;
//put modelPath to Init
model.Init("./model/color");
//we can use "utils::Dectet" to dectet image and video and file
//Dectet(string path, Model* model, vector<string> classes, bool saveFlag, string savePath, bool showFlag)
//saveFlag is set to true by default, which means the processed image or video will be saved. The default save location is the "output" folder in this project. showFlag is set to false by default, which means the processed image will not be displayed.
utils::Dectet("./images", &model, utils::colorClasses);
```
## If you want to join your own new model
specifically a YOLOv5 model, you need to make the following modifications in the .param file: 
1. In the line with three Permute operations, change the fourth parameter to "output", "output1", and "output2" respectively. 
2. Each Permute operation has a corresponding reshape operation. In the line with the reshape operation, change the fifth parameter from 0=? to 0=-1. 
For classification models, modify the .param file as follows:
1. In the first line with the Input operation, change the three parameters to 1 1 images.
2. In the second line, change the first three parameters to 1 1 images. 
3. In the line with the InnerProduct operation, change the fourth parameter to "output". 
By making these modifications, you will be able to run the model within the current framework. It is recommended to use YOLOv5-5.6.2 for converting to ONNX and NCNN. The YOLOv5s, YOLOv5m, and YOLOv5s6 models may not require any additional operations and can be used directly.

# yolov8->ncnn First
we need to find the class c2f in the file "block.py" located at "anaconda3\envs\yolov8\Lib\site-packages\ultralytics\nn\modules". Replace the code in the forward function with the following:

```  
def forward(self, x):
    """Forward pass through C2f layer."""
    #y = list(self.cv1(x).chunk(2, 1))
    #y.extend(m(y[-1]) for m in self.m)
    #return self.cv2(torch.cat(y, 1))

    x = self.cv1(x)
    x = [x, x[:, self.c:, ...]]
    x.extend(m(x[-1]) for m in self.m)
    x.pop(1)
    return self.cv2(torch.cat(x, 1))
```
Then, find the class Dectet in the file "head.py" located in the same directory. Replace the forward function with the following:

```
def forward(self, x):
    shape = x[0].shape  # BCHW
    for i in range(self.nl):
        x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
    if self.training:
        return x
    elif self.dynamic or self.shape != shape:
        self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
        self.shape = shape

    # x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
    # if self.export and self.format in ('saved_model', 'pb', 'tflite', 'edgetpu', 'tfjs'):  # avoid TF FlexSplitV ops
    #     box = x_cat[:, :self.reg_max * 4]
    #     cls = x_cat[:, self.reg_max * 4:]
    # else:
    #     box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)
    # dbox = dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides
    # y = torch.cat((dbox, cls.sigmoid()), 1)
    # return y if self.export else (y, x)

    pred = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2).permute(0, 2, 1)
    return pred
```

Then use the yolov8 built-in 

```
modelPath = 'your path'
model = YOLO(modelPath)
model.export(format='ncnn', half=True, imgsz=640)
```

 to generate the param file. Add 1 to each of the two arrays in the second line and modify these two numbers directly!!! 

Add Permute at the end, 1111 1 1 out0 output 0=1 in the first line, change input in0 to images, and in the second line, change in0 to images. 

This will successfully run the yolov8 model in this project.

### onnx -> ncnn online conversion website https://convertmodel.com/

# 在NCNN中使用YOLOv5和图像分类
## 首先需要安装一些软件包
Vulkan https://developer.nvidia.com/vulkan-driver \
ncnn  https://github.com/Tencent/ncnn/releases 安装 ncnn-20230816-windows-vs2022.zip\
opencv https://opencv.org/releases/ opencv4.7.0\
## 第二步打开 ncnn.sln
使用 release x64 运行项目\
导航到 "Configuration Properties" -> "C/C++" -> "General"。在 "Include Directories" 字段中添加以下路径 \
"你的路径"\Vulkan\Include \
"你的路径"\ncnn-20230816-windows-vs2022\x64\include \
"你的路径"\opencv\opencv\build\include \
在同一个属性窗口中，找到 "Library" 字段 \
"你的路径"\opencv\opencv\build\x64\vc16\lib \
"你的路径"\Vulkan\Lib \
"你的路径"\ncnn-20230816-windows-vs2022\x64\lib \
导航到 "Configuration Properties" -> "Linker" -> "Input"\
dxcompiler.lib\
GenericCodeGen.lib\
glslang-default-resource-limits.lib\
glslang.lib\
HLSL.lib\
MachineIndependent.lib\
OGLCompiler.lib\
OSDependent.lib\
shaderc.lib\
shaderc_combined.lib\
shaderc_shared.lib\
shaderc_util.lib\
spirv-cross-c-shared.lib\
spirv-cross-c.lib\
spirv-cross-core.lib\
spirv-cross-cpp.lib\
spirv-cross-glsl.lib\
spirv-cross-hlsl.lib\
spirv-cross-msl.lib\
spirv-cross-reflect.lib\
spirv-cross-util.lib\
SPIRV-Tools-diff.lib\
SPIRV-Tools-link.lib\
SPIRV-Tools-lint.lib\
SPIRV-Tools-opt.lib\
SPIRV-Tools-reduce.lib\
SPIRV-Tools-shared.lib\
SPIRV-Tools.lib\
SPIRV.lib\
SPVRemapper.lib\
vulkan-1.lib\
ncnn.lib\
opencv_world470.lib\
opencv_world470d.lib\
`不同的下载版本可能导致缺少或添加库的机会。请自行检查` \
`该项目附带了5个预训练模型`\
`shape，这是一个用于检测图像中基本形状位置的目标检测模型`\
`resShape，这是一个用于对基本形状进行分类的图像分类模型`\
`carcard，这是一个用于检测车牌位置的目标检测模型`\
`main，这是一个具有11个类别的目标检测模型`\
`以及color，这是一个用于对颜色进行分类的图像分类模型`
```testCode
//我们可以使用ResNet或Yolov5或Yolov8创建模型。
ResNet model;
//将模型路径放入Init函数中
model.Init("./model/color");
//我们可以使用"utils::Dectet"函数来检测图像、视频和文件
//Dectet(string path, Model* model, vector<string> classes, bool saveFlag, string savePath, bool showFlag)
//saveFlag默认为true，表示处理后的图像或视频将被保存。默认保存位置是项目中的"output"文件夹。showFlag默认为false，表示不显示处理后的图像。
utils::Dectet("./images", &model, utils::colorClasses);
```
## 如果你想使用自己的新模型
特别是YOLOv5模型，你需要在.param文件中进行以下修改：
1. 在包含三个Permute操作的行中，将第四个参数分别改为"output"、"output1"和"output2"。
2. 每个Permute操作都有一个对应的reshape操作。在包含reshape操作的行中，将第五个参数从0=?改为0=-1。
对于分类模型，修改.param文件如下：
1. 在第一行的Input操作中，将三个参数改为1 1 images。
2. 在第二行中，将前三个参数改为1 1 images。
3. 在包含InnerProduct操作的行中，将第四个参数改为"output"。
通过进行这些修改，您将能够在当前框架中运行模型。建议使用YOLOv5-5.6.2进行转换为ONNX和NCNN。YOLOv5s、YOLOv5m和YOLOv5s6模型可能不需要任何额外的操作，可以直接使用。

# yolov8->ncnn 
首先我们需要找到anaconda3\envs\yolov8\Lib\site-packages\ultralytics\nn\modules\block.py中的 class c2f 将其中forward的代码替换为

```    
def forward(self, x):
        """Forward pass through C2f layer."""
        #y = list(self.cv1(x).chunk(2, 1))
        #y.extend(m(y[-1]) for m in self.m)
        #return self.cv2(torch.cat(y, 1))

        x = self.cv1(x)
        x = [x, x[:, self.c:, ...]]
        x.extend(m(x[-1]) for m in self.m)
        x.pop(1)
        return self.cv2(torch.cat(x, 1))

``` 
然后找到同级目录下head.py 中的 class Dectet 将其中的forward 替换为

 ```   
 def forward(self, x):
        shape = x[0].shape  # BCHW
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        if self.training:
            return x
        elif self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        # x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        # if self.export and self.format in ('saved_model', 'pb', 'tflite', 'edgetpu', 'tfjs'):  # avoid TF FlexSplitV ops
        #     box = x_cat[:, :self.reg_max * 4]
        #     cls = x_cat[:, self.reg_max * 4:]
        # else:
        #     box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)
        # dbox = dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides
        # y = torch.cat((dbox, cls.sigmoid()), 1)
        # return y if self.export else (y, x)

        pred = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2).permute(0, 2, 1)
        return pred
```

然后使用yolov8 自带的

```
modelPath = 'your path'
model = YOLO(modelPath)
model.export(format='ncnn', half=True, imgsz=640)
``` 

将生成的param 第二行两个数组各加1 直接修改这两个数字！！！ 

在最后一行添加Permute  1111 1 1 out0 output 0=1 第一行input in0 改为images 第二行in0 改为images

即可成功在此项目运行yolov8 模型
### onnx->ncnn 在线转换网站 https://convertmodel.com/
