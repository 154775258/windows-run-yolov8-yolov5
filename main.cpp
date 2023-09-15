#include"yolo.h"
#include"utils.h"
#include "yolo-seg.h"

int main() {
	//we can use ResNet or Yolo to create model.
	Yolov8 model;
	//put modelPath to Init
	model.Init("./model/yolov8s");
	//we can use "utils::Dectet" to dectet image and video and file
	//Dectet(string path, Model* model, vector<string> classes, bool saveFlag, string savePath, bool showFlag)
	//saveFlag is set to true by default, which means the processed image or video will be saved. The default save location is the "output" folder in this project. showFlag is set to false by default, which means the processed image will not be displayed.
	double start_time = ncnn::get_current_time();
	utils::Dectet("./images", &model, utils::cocoClasses);
	cout << "总耗时: " << ncnn::get_current_time() - start_time << "ms\n";
	
	 
    Yolov5Seg SegModel;
	SegModel.Init("./model/yolov5s-seg");
	start_time = ncnn::get_current_time();
	utils::DectetSeg("./images", &SegModel, utils::cocoClasses);
	cout << "总耗时: " << ncnn::get_current_time() - start_time << "ms\n";
	

}