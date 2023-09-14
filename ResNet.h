#pragma once
#include "model.h"
#include<opencv2/opencv.hpp>
#include<string>
#include<vector>
#include<unordered_map>
#include <ncnn/layer.h>
#include <ncnn/net.h>
#include <ncnn/benchmark.h>
#include<algorithm>

class ResNet : public Model {
public:


    bool Init(std::string modelPath) override {
        int ret1 = resNet50.load_param((modelPath + ".param").c_str());
        int ret2 = resNet50.load_model((modelPath + ".bin").c_str());
        std::cout << "ResNet Model loaded\n";
        if (ret1 && ret2)
            return true;
        else
            return false;
    }

    vector<Object> Dectet(cv::Mat image, bool use_gpu) override {

        if (use_gpu == true && ncnn::get_gpu_count() == 0)
        {
            use_gpu = false;
        }

        // ncnn from bitmap
        const int target_size = 224;

        cv::resize(image, image, { target_size, target_size });

        ncnn::Mat in = ncnn::Mat::from_pixels_resize(image.data, ncnn::Mat::PIXEL_BGR, image.cols, image.rows, 224, 224);

        float mean[3] = { 0.485 * 255.f, 0.456 * 255.f, 0.406 * 255.f };
        float std[3] = { 1.0 / 0.229 / 255, 1.0 / 0.224 / 255, 1.0 / 0.225 / 255 };
        in.substract_mean_normalize(mean, std);

        ncnn::Extractor ex = resNet50.create_extractor();
        //ex.set_num_threads(4);
        ex.set_vulkan_compute(use_gpu);
        double start_time = ncnn::get_current_time();
        ex.input("input", in);
        ncnn::Mat preds;
        ex.extract("output", preds);
        float max_prob = 0.0f;
        int max_index = 0;
        for (int i = 0; i < preds.w; i++) {
            float prob = preds[i];
            if (prob > max_prob) {
                max_prob = prob;
                max_index = i;
            }
            //std::cout << "¸ÅÂÊ:" << prob << '\n';
        }
        vector<Object> ans;
        double end_time = ncnn::get_current_time() - start_time;
        std::cout << end_time << "ms Dectet\n";
        if (max_prob < 2)ans.push_back({ 10,10,0,0,preds.w, max_prob });
        else ans.push_back({ 10, 10, 0, 0, max_index, max_prob });
        return ans;
    }

private:
    ncnn::Net resNet50;

};