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



