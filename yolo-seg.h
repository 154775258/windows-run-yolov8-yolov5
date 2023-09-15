#pragma once
#include <ncnn/layer.h>
#include <ncnn/net.h>

#include<opencv2/opencv.hpp>

#include <float.h>
#include <stdio.h>
#include <vector>
#include "model.h"

#define MAX_STRIDE 64


class Yolov5Seg:public SegModel{
public:
    static void slice(const ncnn::Mat& in, ncnn::Mat& out, int start, int end, int axis)
    {
        ncnn::Option opt;
        opt.num_threads = 4;
        opt.use_fp16_storage = false;
        opt.use_packing_layout = false;

        ncnn::Layer* op = ncnn::create_layer("Crop");

        // set param
        ncnn::ParamDict pd;

        ncnn::Mat axes = ncnn::Mat(1);
        axes.fill(axis);
        ncnn::Mat ends = ncnn::Mat(1);
        ends.fill(end);
        ncnn::Mat starts = ncnn::Mat(1);
        starts.fill(start);
        pd.set(9, starts);// start
        pd.set(10, ends);// end
        pd.set(11, axes);//axes

        op->load_param(pd);

        op->create_pipeline(opt);

        // forward
        op->forward(in, out, opt);

        op->destroy_pipeline(opt);

        delete op;
    }
    static void interp(const ncnn::Mat& in, const float& scale, const int& out_w, const int& out_h, ncnn::Mat& out)
    {
        ncnn::Option opt;
        opt.num_threads = 4;
        opt.use_fp16_storage = false;
        opt.use_packing_layout = false;

        ncnn::Layer* op = ncnn::create_layer("Interp");

        // set param
        ncnn::ParamDict pd;
        pd.set(0, 2);// resize_type
        pd.set(1, scale);// height_scale
        pd.set(2, scale);// width_scale
        pd.set(3, out_h);// height
        pd.set(4, out_w);// width

        op->load_param(pd);

        op->create_pipeline(opt);

        // forward
        op->forward(in, out, opt);

        op->destroy_pipeline(opt);

        delete op;
    }
    static void reshape(const ncnn::Mat& in, ncnn::Mat& out, int c, int h, int w, int d)
    {
        ncnn::Option opt;
        opt.num_threads = 4;
        opt.use_fp16_storage = false;
        opt.use_packing_layout = false;

        ncnn::Layer* op = ncnn::create_layer("Reshape");

        // set param
        ncnn::ParamDict pd;

        pd.set(0, w);// start
        pd.set(1, h);// end
        if (d > 0)
            pd.set(11, d);//axes
        pd.set(2, c);//axes
        op->load_param(pd);

        op->create_pipeline(opt);

        // forward
        op->forward(in, out, opt);

        op->destroy_pipeline(opt);

        delete op;
    }
    static void sigmoid(ncnn::Mat& bottom)
    {
        ncnn::Option opt;
        opt.num_threads = 4;
        opt.use_fp16_storage = false;
        opt.use_packing_layout = false;

        ncnn::Layer* op = ncnn::create_layer("Sigmoid");

        op->create_pipeline(opt);

        // forward

        op->forward_inplace(bottom, opt);
        op->destroy_pipeline(opt);

        delete op;
    }

    static void matmul(const std::vector<ncnn::Mat>& bottom_blobs, ncnn::Mat& top_blob)
    {
        ncnn::Option opt;
        opt.num_threads = 2;
        opt.use_fp16_storage = false;
        opt.use_packing_layout = false;

        ncnn::Layer* op = ncnn::create_layer("MatMul");

        // set param
        ncnn::ParamDict pd;
        pd.set(0, 0);// axis

        op->load_param(pd);

        op->create_pipeline(opt);
        std::vector<ncnn::Mat> top_blobs(1);
        op->forward(bottom_blobs, top_blobs, opt);
        top_blob = top_blobs[0];

        op->destroy_pipeline(opt);

        delete op;
    }

    static inline float intersection_area(const ObjectSeg& a, const ObjectSeg& b)
    {
        cv::Rect_<float> inter = a.rect & b.rect;
        return inter.area();
    }

    static void qsort_descent_inplace(std::vector<ObjectSeg>& faceobjects, int left, int right)
    {
        int i = left;
        int j = right;
        float p = faceobjects[(left + right) / 2].prob;

        while (i <= j)
        {
            while (faceobjects[i].prob > p)
                i++;

            while (faceobjects[j].prob < p)
                j--;

            if (i <= j)
            {
                // swap
                std::swap(faceobjects[i], faceobjects[j]);

                i++;
                j--;
            }
        }

#pragma omp parallel sections
        {
#pragma omp section
            {
                if (left < j) qsort_descent_inplace(faceobjects, left, j);
            }
#pragma omp section
            {
                if (i < right) qsort_descent_inplace(faceobjects, i, right);
            }
        }
    }

    static void qsort_descent_inplace(std::vector<ObjectSeg>& faceobjects)
    {
        if (faceobjects.empty())
            return;

        qsort_descent_inplace(faceobjects, 0, faceobjects.size() - 1);
    }

    static void nms_sorted_bboxes(const std::vector<ObjectSeg>& faceobjects, std::vector<int>& picked, float nms_threshold)
    {
        picked.clear();

        const int n = faceobjects.size();

        std::vector<float> areas(n);
        for (int i = 0; i < n; i++)
        {
            areas[i] = faceobjects[i].rect.area();
        }

        for (int i = 0; i < n; i++)
        {
            const ObjectSeg& a = faceobjects[i];

            int keep = 1;
            for (int j = 0; j < (int)picked.size(); j++)
            {
                const ObjectSeg& b = faceobjects[picked[j]];

                // intersection over union
                float inter_area = intersection_area(a, b);
                float union_area = areas[i] + areas[picked[j]] - inter_area;
                // float IoU = inter_area / union_area
                if (inter_area / union_area > nms_threshold)
                    keep = 0;
            }

            if (keep)
                picked.push_back(i);
        }
    }

    static inline float sigmoid(float x)
    {
        return static_cast<float>(1.f / (1.f + exp(-x)));
    }

    static void generate_proposals(const ncnn::Mat& anchors, int stride, const ncnn::Mat& in_pad, const ncnn::Mat& feat_blob, float prob_threshold, std::vector<ObjectSeg>& objects)
    {
        const int num_grid = feat_blob.h;

        int num_grid_x;
        int num_grid_y;
        if (in_pad.w > in_pad.h)
        {
            num_grid_x = in_pad.w / stride;
            num_grid_y = num_grid / num_grid_x;
        }
        else
        {
            num_grid_y = in_pad.h / stride;
            num_grid_x = num_grid / num_grid_y;
        }

        const int num_class = feat_blob.w - 5 - 32;

        const int num_anchors = anchors.w / 2;

        for (int q = 0; q < num_anchors; q++)
        {
            const float anchor_w = anchors[q * 2];
            const float anchor_h = anchors[q * 2 + 1];

            const ncnn::Mat feat = feat_blob.channel(q);

            for (int i = 0; i < num_grid_y; i++)
            {
                for (int j = 0; j < num_grid_x; j++)
                {
                    const float* featptr = feat.row(i * num_grid_x + j);

                    // find class index with max class score
                    int class_index = 0;
                    float class_score = -FLT_MAX;
                    for (int k = 0; k < num_class; k++)
                    {
                        float score = featptr[5 + k];
                        if (score > class_score)
                        {
                            class_index = k;
                            class_score = score;
                        }
                    }

                    float box_score = featptr[4];

                    float confidence = sigmoid(box_score) * sigmoid(class_score);

                    if (confidence >= prob_threshold)
                    {
                        // yolov5/models/yolo.py Detect forward
                        // y = x[i].sigmoid()
                        // y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i].to(x[i].device)) * self.stride[i]  # xy
                        // y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh

                        float dx = sigmoid(featptr[0]);
                        float dy = sigmoid(featptr[1]);
                        float dw = sigmoid(featptr[2]);
                        float dh = sigmoid(featptr[3]);

                        float pb_cx = (dx * 2.f - 0.5f + j) * stride;
                        float pb_cy = (dy * 2.f - 0.5f + i) * stride;

                        float pb_w = pow(dw * 2.f, 2) * anchor_w;
                        float pb_h = pow(dh * 2.f, 2) * anchor_h;

                        float x0 = pb_cx - pb_w * 0.5f;
                        float y0 = pb_cy - pb_h * 0.5f;
                        float x1 = pb_cx + pb_w * 0.5f;
                        float y1 = pb_cy + pb_h * 0.5f;

                        ObjectSeg obj;
                        obj.rect.x = x0;
                        obj.rect.y = y0;
                        obj.rect.width = x1 - x0;
                        obj.rect.height = y1 - y0;
                        obj.label = class_index;
                        obj.prob = confidence;
                        obj.mask_feat.resize(32);
                        std::copy(featptr + 5 + num_class, featptr + 5 + num_class + 32, obj.mask_feat.begin());

                        objects.push_back(obj);
                    }
                }
            }
        }
    }
    static void decode_mask(const ncnn::Mat& mask_feat, const int& img_w, const int& img_h,
        const ncnn::Mat& mask_proto, const ncnn::Mat& in_pad, const int& wpad, const int& hpad,
        ncnn::Mat& mask_pred_result)
    {
        ncnn::Mat masks;
        matmul(std::vector<ncnn::Mat>{mask_feat, mask_proto}, masks);
        sigmoid(masks);
        reshape(masks, masks, masks.h, in_pad.h / 4, in_pad.w / 4, 0);
        interp(masks, 4.0, 0, 0, masks);
        //ncnn::Mat mask_pred_result;
        slice(masks, mask_pred_result, wpad / 2, in_pad.w - wpad / 2, 2);
        slice(mask_pred_result, mask_pred_result, hpad / 2, in_pad.h - hpad / 2, 1);
        interp(mask_pred_result, 1.0, img_w, img_h, mask_pred_result);

    }

    bool Init(string modelName) override {
        ncnn::Option opt;
        if (ncnn::get_gpu_count() != 0)
            opt.use_vulkan_compute = true;

        yolov5.opt = opt;


        // init param
        {
            int ret = yolov5.load_param((modelName + ".param").c_str());
            if (ret != 0)
            {
                std::cout << "YoloV5Ncnn load_param failed\n";
                return false;
            }
        }

        // init bin
        {
            int ret = yolov5.load_model((modelName + ".bin").c_str());
            if (ret != 0)
            {
                std::cout << "YoloV5Ncnn  load_model failed\n";
                return false;
            }
        }

        std::cout << "Yolov5 Model loaded\n";
        //std::cout << "Ä£ÐÍµØÖ·: " << &yolov5 << '\n';

        return true;
    }


    vector<ObjectSeg> Dectet(cv::Mat& bgr,bool use_gpu)override
    {
        if (use_gpu == true && ncnn::get_gpu_count() == 0)
        {
            use_gpu = false;
        }
        const int target_size = 640;
        const float prob_threshold = 0.25f;
        const float nms_threshold = 0.45f;
        vector<ObjectSeg> objects;
        int img_w = bgr.cols;
        int img_h = bgr.rows;

        // letterbox pad to multiple of MAX_STRIDE
        int w = img_w;
        int h = img_h;
        float scale = 1.f;
        if (w > h)
        {
            scale = (float)target_size / w;
            w = target_size;
            h = h * scale;
        }
        else
        {
            scale = (float)target_size / h;
            h = target_size;
            w = w * scale;
        }

        ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR2RGB, img_w, img_h, w, h);

        // pad to target_size rectangle
        // yolov5/utils/datasets.py letterbox
        int wpad = (w + MAX_STRIDE - 1) / MAX_STRIDE * MAX_STRIDE - w;
        int hpad = (h + MAX_STRIDE - 1) / MAX_STRIDE * MAX_STRIDE - h;
        ncnn::Mat in_pad;
        ncnn::copy_make_border(in, in_pad, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, ncnn::BORDER_CONSTANT, 114.f);

        const float norm_vals[3] = { 1 / 255.f, 1 / 255.f, 1 / 255.f };
        in_pad.substract_mean_normalize(0, norm_vals);

        ncnn::Extractor ex = yolov5.create_extractor();
        double start_time = ncnn::get_current_time();
        ex.set_vulkan_compute(use_gpu);
        ex.input("images", in_pad);

        std::vector<ObjectSeg> proposals;
        // stride 8
        {
            ncnn::Mat out;
            ex.extract("output", out);

            ncnn::Mat anchors(6);
            anchors[0] = 10.f;
            anchors[1] = 13.f;
            anchors[2] = 16.f;
            anchors[3] = 30.f;
            anchors[4] = 33.f;
            anchors[5] = 23.f;

            std::vector<ObjectSeg> objects8;
            generate_proposals(anchors, 8, in_pad, out, prob_threshold, objects8);

            proposals.insert(proposals.end(), objects8.begin(), objects8.end());
        }

        // stride 16
        {
            ncnn::Mat out;
            ex.extract("output1", out);


            ncnn::Mat anchors(6);
            anchors[0] = 30.f;
            anchors[1] = 61.f;
            anchors[2] = 62.f;
            anchors[3] = 45.f;
            anchors[4] = 59.f;
            anchors[5] = 119.f;

            std::vector<ObjectSeg> objects16;
            generate_proposals(anchors, 16, in_pad, out, prob_threshold, objects16);

            proposals.insert(proposals.end(), objects16.begin(), objects16.end());
        }

        // stride 32
        {
            ncnn::Mat out;
            ex.extract("output2", out);

            ncnn::Mat anchors(6);
            anchors[0] = 116.f;
            anchors[1] = 90.f;
            anchors[2] = 156.f;
            anchors[3] = 198.f;
            anchors[4] = 373.f;
            anchors[5] = 326.f;

            std::vector<ObjectSeg> objects32;
            generate_proposals(anchors, 32, in_pad, out, prob_threshold, objects32);

            proposals.insert(proposals.end(), objects32.begin(), objects32.end());
        }

        ncnn::Mat mask_proto;
        ex.extract("seg", mask_proto);

        // sort all proposals by score from highest to lowest
        qsort_descent_inplace(proposals);

        // apply nms with nms_threshold
        std::vector<int> picked;
        nms_sorted_bboxes(proposals, picked, nms_threshold);

        int count = picked.size();

        ncnn::Mat mask_feat = ncnn::Mat(32, count, sizeof(float));
        for (int i = 0; i < count; i++) {
            std::copy(proposals[picked[i]].mask_feat.begin(), proposals[picked[i]].mask_feat.end(), mask_feat.row(i));
        }

        ncnn::Mat mask_pred_result;
        decode_mask(mask_feat, img_w, img_h, mask_proto, in_pad, wpad, hpad, mask_pred_result);

        objects.resize(count);
        for (int i = 0; i < count; i++)
        {
            objects[i] = proposals[picked[i]];

            // adjust offset to original unpadded
            float x0 = (objects[i].rect.x - (wpad / 2)) / scale;
            float y0 = (objects[i].rect.y - (hpad / 2)) / scale;
            float x1 = (objects[i].rect.x + objects[i].rect.width - (wpad / 2)) / scale;
            float y1 = (objects[i].rect.y + objects[i].rect.height - (hpad / 2)) / scale;

            // clip
            x0 = max(min(x0, (float)(img_w - 1)), 0.f);
            y0 = max(min(y0, (float)(img_h - 1)), 0.f);
            x1 = max(min(x1, (float)(img_w - 1)), 0.f);
            y1 = max(min(y1, (float)(img_h - 1)), 0.f);

            objects[i].rect.x = x0;
            objects[i].rect.y = y0;
            objects[i].rect.width = x1 - x0;
            objects[i].rect.height = y1 - y0;

            objects[i].cv_mask = cv::Mat::zeros(img_h, img_w, CV_32FC1);
            cv::Mat mask = cv::Mat(img_h, img_w, CV_32FC1, (float*)mask_pred_result.channel(i));
            mask(objects[i].rect).copyTo(objects[i].cv_mask(objects[i].rect));
        }
        double elasped = ncnn::get_current_time() - start_time;
        std::cout << "YoloV5Ncnn " << elasped << "ms   detect\n";
        return objects;
    }

    cv::Mat draw_objects(const cv::Mat& bgr, const std::vector<ObjectSeg>& objects, vector<string> class_names)override
    {
        static const unsigned char colors[81][3] = {
                {56,  0,   255},
                {226, 255, 0},
                {0,   94,  255},
                {0,   37,  255},
                {0,   255, 94},
                {255, 226, 0},
                {0,   18,  255},
                {255, 151, 0},
                {170, 0,   255},
                {0,   255, 56},
                {255, 0,   75},
                {0,   75,  255},
                {0,   255, 169},
                {255, 0,   207},
                {75,  255, 0},
                {207, 0,   255},
                {37,  0,   255},
                {0,   207, 255},
                {94,  0,   255},
                {0,   255, 113},
                {255, 18,  0},
                {255, 0,   56},
                {18,  0,   255},
                {0,   255, 226},
                {170, 255, 0},
                {255, 0,   245},
                {151, 255, 0},
                {132, 255, 0},
                {75,  0,   255},
                {151, 0,   255},
                {0,   151, 255},
                {132, 0,   255},
                {0,   255, 245},
                {255, 132, 0},
                {226, 0,   255},
                {255, 37,  0},
                {207, 255, 0},
                {0,   255, 207},
                {94,  255, 0},
                {0,   226, 255},
                {56,  255, 0},
                {255, 94,  0},
                {255, 113, 0},
                {0,   132, 255},
                {255, 0,   132},
                {255, 170, 0},
                {255, 0,   188},
                {113, 255, 0},
                {245, 0,   255},
                {113, 0,   255},
                {255, 188, 0},
                {0,   113, 255},
                {255, 0,   0},
                {0,   56,  255},
                {255, 0,   113},
                {0,   255, 188},
                {255, 0,   94},
                {255, 0,   18},
                {18,  255, 0},
                {0,   255, 132},
                {0,   188, 255},
                {0,   245, 255},
                {0,   169, 255},
                {37,  255, 0},
                {255, 0,   151},
                {188, 0,   255},
                {0,   255, 37},
                {0,   255, 0},
                {255, 0,   170},
                {255, 0,   37},
                {255, 75,  0},
                {0,   0,   255},
                {255, 207, 0},
                {255, 0,   226},
                {255, 245, 0},
                {188, 255, 0},
                {0,   255, 18},
                {0,   255, 75},
                {0,   255, 151},
                {255, 56,  0},
                {245, 255, 0}
        };

        int color_index = 0;

        cv::Mat image = bgr.clone();

        for (int i = 0; i < objects.size(); i++) {
            const ObjectSeg& obj = objects[i];

            const unsigned char* color = colors[color_index % 80];
            color_index++;

            cv::Scalar cc(color[0], color[1], color[2]);

            for (int y = 0; y < image.rows; y++) {
                uchar* image_ptr = image.ptr(y);
                const float* mask_ptr = obj.cv_mask.ptr<float>(y);
                for (int x = 0; x < image.cols; x++) {
                    if (mask_ptr[x] >= 0.5)
                    {
                        image_ptr[0] = cv::saturate_cast<uchar>(image_ptr[0] * 0.5 + color[2] * 0.5);
                        image_ptr[1] = cv::saturate_cast<uchar>(image_ptr[1] * 0.5 + color[1] * 0.5);
                        image_ptr[2] = cv::saturate_cast<uchar>(image_ptr[2] * 0.5 + color[0] * 0.5);
                    }
                    image_ptr += 3;
                }
            }

            cv::rectangle(image, obj.rect, cc, 2);

            string text;
            if (class_names.size() > obj.label) {
                //cout << classes.size();
                text = class_names[obj.label] + ' ' + to_string(obj.prob * 100) + "%";
            }
            else
                text = to_string(obj.label) + ' ' + to_string(obj.prob * 100) + "%";
            int baseLine = 0;
            cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

            int x = obj.rect.x;
            int y = obj.rect.y - label_size.height - baseLine;
            if (y < 0)
                y = 0;
            if (x + label_size.width > image.cols)
                x = image.cols - label_size.width;

            cv::rectangle(image, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                cc, -1);

            cv::putText(image, text, cv::Point(x, y + label_size.height),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255));
        }
        return image.clone();
    }
    /*
    int main(int argc, char** argv)
    {
        if (argc != 2)
        {
            fprintf(stderr, "Usage: %s [imagepath]\n", argv[0]);
            return -1;
        }

        const char* imagepath = argv[1];

        cv::Mat m = cv::imread(imagepath, 1);
        if (m.empty())
        {
            fprintf(stderr, "cv::imread %s failed\n", imagepath);
            return -1;
        }

        std::vector<ObjectSeg> objects;

        detect_yolov5(m, objects);

        draw_objects(m, objects);

        return 0;
    }
    */
    private:
        ncnn::Net yolov5;

};
