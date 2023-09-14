#include"model.h"
#include"utils.h"
#include <ncnn/net.h>

int main() {
	//we can use ResNet or Yolo to create model.
	Yolov5 model;
	//put modelPath to Init
	model.Init("D:\\ncnnModel\\main-f16");
	//we can use "utils::Dectet" to dectet image and video and file
	//Dectet(string path, Model* model, vector<string> classes, bool saveFlag, string savePath, bool showFlag)
	//saveFlag is set to true by default, which means the processed image or video will be saved. The default save location is the "output" folder in this project. showFlag is set to false by default, which means the processed image will not be displayed.
	double start_time = ncnn::get_current_time();
	utils::Dectet("D:\\pytorch\\data\\YoloTFTandQrcode\\val\\images", &model, utils::mainClasses);
	cout << "�ܺ�ʱ: " << ncnn::get_current_time() - start_time << "ms\n";
	
}