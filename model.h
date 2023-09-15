#pragma once
#include<opencv2/opencv.hpp>
#include<string>
#include<vector>
#include<unordered_map>
#include <ncnn/layer.h>
#include <ncnn/net.h>
#include <ncnn/benchmark.h>
#include<algorithm>

struct Object
{
    float x;
    float y;
    float w;
    float h;
    int label;
    float prob;
};

struct ObjectSeg
{
    cv::Rect_<float> rect;
    int label;
    float prob;
    std::vector<float> mask_feat;
    cv::Mat cv_mask;

};

using namespace std;

//建立自己的模型继承这个类重写Init Detect即可使用utils::Dectet
class Model {
public:
    virtual bool Init(std::string modelPath) {
        return 1;
    }
    virtual vector<Object> Dectet(cv::Mat bitmap, bool use_gpu) {
        return {};
    }
};

class SegModel {
public:
    virtual bool Init(std::string modelPath) {
        return 1;
    }
    virtual vector<ObjectSeg> Dectet(cv::Mat& bgr, bool use_gpu) {
        return {};
    }
    virtual cv::Mat draw_objects(const cv::Mat& bgr, const std::vector<ObjectSeg>& objects, vector<string> class_names)
    {
        cv::Mat image(480, 640, CV_8UC3, cv::Scalar(0, 0, 0));
        return image;
    }
};



