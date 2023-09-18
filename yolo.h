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

static inline float intersection_area(const Object& a, const Object& b)
{
    if (a.x > b.x + b.w || a.x + a.w < b.x || a.y > b.y + b.h || a.y + a.h < b.y)
    {
        // no intersection
        return 0.f;
    }

    float inter_width = min(a.x + a.w, b.x + b.w) - max(a.x, b.x);
    float inter_height = min(a.y + a.h, b.y + b.h) - max(a.y, b.y);

    return inter_width * inter_height;
}

static void nms_sorted_bboxes(const std::vector<Object>& faceobjects, std::vector<int>& picked, float nms_threshold)
{
    picked.clear();

    const int n = faceobjects.size();

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++)
    {
        areas[i] = faceobjects[i].w * faceobjects[i].h;
    }

    for (int i = 0; i < n; i++)
    {
        const Object& a = faceobjects[i];

        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++)
        {
            const Object& b = faceobjects[picked[j]];

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


static float fast_exp(float x)
{
    union {
        uint32_t i;
        float f;
    } v{};
    v.i = (1 << 23) * (1.4426950409 * x + 126.93490512f);
    return v.f;
}

static inline float sigmoid(float x)
{
    return static_cast<float>(1.f / (1.f + fast_exp(-x)));
}

static void qsort_descent_inplace(std::vector<Object>& faceobjects, int left, int right)
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

static void qsort_descent_inplace(std::vector<Object>& faceobjects)
{
    if (faceobjects.empty())
        return;

    qsort_descent_inplace(faceobjects, 0, faceobjects.size() - 1);
}

class Yolov5 :public Model {
public:

    static void generate_proposals(const ncnn::Mat& anchors, int stride, const ncnn::Mat& in_pad, const ncnn::Mat& feat_blob, float prob_threshold, std::vector<Object>& objects)
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

        const int num_class = feat_blob.w - 5;

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

                        Object obj;
                        obj.x = x0;
                        obj.y = y0;
                        obj.w = x1 - x0;
                        obj.h = y1 - y0;
                        obj.label = class_index;
                        obj.prob = confidence;

                        objects.push_back(obj);
                    }
                }
            }
        }
    }

    bool Init(std::string modelName, int targetSize, double conf, double iou) override
    {
        ncnn::Option opt;
        opt.lightmode = true;
        opt.num_threads = 4;
        opt.blob_allocator = &g_blob_pool_allocator;
        opt.workspace_allocator = &g_workspace_pool_allocator;
        opt.use_packing_layout = true;
        prob_threshold = conf;
        nms_threshold = iou;
        std::string modelPath = modelName;
        target_size = targetSize;
        // use vulkan compute
        if (ncnn::get_gpu_count() != 0)
            opt.use_vulkan_compute = true;

        yolov5.opt = opt;


        // init param
        {
            int ret = yolov5.load_param((modelPath + ".param").c_str());
            if (ret != 0)
            {
                std::cout << "YoloV5Ncnn load_param failed\n";
                return false;
            }
        }

        // init bin
        {
            int ret = yolov5.load_model((modelPath + ".bin").c_str());
            if (ret != 0)
            {
                std::cout << "YoloV5Ncnn  load_model failed\n";
                return false;
            }
        }

        std::cout << "Yolov5 Model loaded\n";
        //std::cout << "模型地址: " << &yolov5 << '\n';

        return true;
    }

    // public native Obj[] Detect(Bitmap bitmap, boolean use_gpu);
    vector<Object> Dectet(cv::Mat bitmap, bool use_gpu) override
    {
        if (use_gpu == true && ncnn::get_gpu_count() == 0)
        {
            use_gpu = false;
        }

        const int width = bitmap.cols;
        const int height = bitmap.rows;

        // letterbox pad to multiple of 32
        int w = width;
        int h = height;
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

        ncnn::Mat in = ncnn::Mat::from_pixels_resize(bitmap.data, ncnn::Mat::PIXEL_BGR, bitmap.cols, bitmap.rows, w, h);

        // pad to target_size rectangle
        // yolov5/utils/datasets.py letterbox
        int wpad = (w + 31) / 32 * 32 - w;
        int hpad = (h + 31) / 32 * 32 - h;
        ncnn::Mat in_pad;
        ncnn::copy_make_border(in, in_pad, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, ncnn::BORDER_CONSTANT, 114.f);
        double start_time;
        // yolov5
        std::vector<Object> objects;
        {
            const float norm_vals[3] = { 1 / 255.f, 1 / 255.f, 1 / 255.f };
            in_pad.substract_mean_normalize(0, norm_vals);

            ncnn::Extractor ex = yolov5.create_extractor();
            //ex.set_num_threads(4);
            ex.set_vulkan_compute(use_gpu);
            start_time = ncnn::get_current_time();
            ex.input("images", in_pad);

            std::vector<Object> proposals;

            // anchor setting from yolov5/models/yolov5s.yaml

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

                std::vector<Object> objects8;
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

                std::vector<Object> objects16;
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

                std::vector<Object> objects32;
                generate_proposals(anchors, 32, in_pad, out, prob_threshold, objects32);

                proposals.insert(proposals.end(), objects32.begin(), objects32.end());
            }

            // sort all proposals by score from highest to lowest
            qsort_descent_inplace(proposals);

            // apply nms with nms_threshold
            std::vector<int> picked;
            nms_sorted_bboxes(proposals, picked, nms_threshold);

            int count = picked.size();

            objects.resize(count);
            for (int i = 0; i < count; i++)
            {
                objects[i] = proposals[picked[i]];

                // adjust offset to original unpadded
                float x0 = (objects[i].x - (wpad / 2)) / scale;
                float y0 = (objects[i].y - (hpad / 2)) / scale;
                float x1 = (objects[i].x + objects[i].w - (wpad / 2)) / scale;
                float y1 = (objects[i].y + objects[i].h - (hpad / 2)) / scale;

                // clip
                x0 = max(min(x0, (float)(width - 1)), 0.f);
                y0 = max(min(y0, (float)(height - 1)), 0.f);
                x1 = max(min(x1, (float)(width - 1)), 0.f);
                y1 = max(min(y1, (float)(height - 1)), 0.f);

                objects[i].x = x0;
                objects[i].y = y0;
                objects[i].w = x1 - x0;
                objects[i].h = y1 - y0;
            }
        }

        /*
        // objects to Obj[]
        static const char* class_names[] = {
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
            "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
            "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
            "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
            "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
            "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
            "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
            "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
            "hair drier", "toothbrush"
        };
         */

        double elasped = ncnn::get_current_time() - start_time;
        std::cout << "YoloV5Ncnn " << elasped << "ms   detect\n";

        return objects;
    }

private:
    ncnn::UnlockedPoolAllocator g_blob_pool_allocator;
    ncnn::PoolAllocator g_workspace_pool_allocator;
    ncnn::Net yolov5;
    int target_size = 640;
    float prob_threshold = 0.25f;
    float nms_threshold = 0.45f;

};

class Yolov8 : public Model {

public:

    struct GridAndStride
    {
        int grid0;
        int grid1;
        int stride;
    };

    Yolov8()
    {
        blob_pool_allocator.set_size_compare_ratio(0.f);
        workspace_pool_allocator.set_size_compare_ratio(0.f);
    }

    static void generate_grids_and_stride(const int target_w, const int target_h, std::vector<int>& strides, std::vector<GridAndStride>& grid_strides)
    {
        for (int i = 0; i < (int)strides.size(); i++)
        {
            int stride = strides[i];
            int num_grid_w = target_w / stride;
            int num_grid_h = target_h / stride;
            for (int g1 = 0; g1 < num_grid_h; g1++)
            {
                for (int g0 = 0; g0 < num_grid_w; g0++)
                {
                    GridAndStride gs;
                    gs.grid0 = g0;
                    gs.grid1 = g1;
                    gs.stride = stride;
                    grid_strides.push_back(gs);
                }
            }
        }
    }
    static void generate_proposals(std::vector<GridAndStride> grid_strides, const ncnn::Mat& pred, float prob_threshold, std::vector<Object>& objects)
    {
        const int num_points = grid_strides.size();
        const int num_class = pred.w - 64;//-64即为模型实际可识别类别数量
        const int reg_max_1 = 16;

        for (int i = 0; i < num_points; i++)
        {
            const float* scores = pred.row(i) + 4 * reg_max_1;

            // find label with max score
            int label = -1;
            float score = -FLT_MAX;
            for (int k = 0; k < num_class; k++)
            {
                float confidence = scores[k];
                if (confidence > score)
                {
                    label = k;
                    score = confidence;
                }
            }
            float box_prob = sigmoid(score);
            if (box_prob >= prob_threshold)
            {
                ncnn::Mat bbox_pred(reg_max_1, 4, (void*)pred.row(i));
                {
                    ncnn::Layer* softmax = ncnn::create_layer("Softmax");

                    ncnn::ParamDict pd;
                    pd.set(0, 1); // axis
                    pd.set(1, 1);
                    softmax->load_param(pd);

                    ncnn::Option opt;
                    opt.num_threads = 1;
                    opt.use_packing_layout = false;

                    softmax->create_pipeline(opt);

                    softmax->forward_inplace(bbox_pred, opt);

                    softmax->destroy_pipeline(opt);

                    delete softmax;
                }

                float pred_ltrb[4];
                for (int k = 0; k < 4; k++)
                {
                    float dis = 0.f;
                    const float* dis_after_sm = bbox_pred.row(k);
                    for (int l = 0; l < reg_max_1; l++)
                    {
                        dis += l * dis_after_sm[l];
                    }

                    pred_ltrb[k] = dis * grid_strides[i].stride;
                }

                float pb_cx = (grid_strides[i].grid0 + 0.5f) * grid_strides[i].stride;
                float pb_cy = (grid_strides[i].grid1 + 0.5f) * grid_strides[i].stride;

                float x0 = pb_cx - pred_ltrb[0];
                float y0 = pb_cy - pred_ltrb[1];
                float x1 = pb_cx + pred_ltrb[2];
                float y1 = pb_cy + pred_ltrb[3];

                Object obj;
                obj.x = x0;
                obj.y = y0;
                obj.w = x1 - x0;
                obj.h = y1 - y0;
                obj.label = label;
                obj.prob = box_prob;

                objects.push_back(obj);
            }
        }
    }


    bool Init(string modelName, int targetSize, double conf, double iou)override
    {

        ncnn::Option opt;
        opt.lightmode = true;
        opt.num_threads = 4;
        opt.blob_allocator = &blob_pool_allocator;
        opt.workspace_allocator = &workspace_pool_allocator;
        opt.use_packing_layout = true;
        prob_threshold = conf;
        nms_threshold = iou;
        std::string modelPath = modelName;

        // use vulkan compute
        if (ncnn::get_gpu_count() != 0)
            opt.use_vulkan_compute = true;

        yolo.opt = opt;

        int ret = yolo.load_param((modelPath + ".param").c_str());
        if (ret != 0)
        {
            std::cout << "YoloV8Ncnn load_param failed\n";
            return false;
        }



        ret = yolo.load_model((modelPath + ".bin").c_str());
        if (ret != 0)
        {
            std::cout << "YoloV8Ncnn  load_model failed\n";
            return false;
        }


        yolo.load_param((modelName + ".param").c_str());
        yolo.load_model((modelName + ".bin").c_str());

        target_size = targetSize;

        std::cout << "Yolov8 Model loaded\n";
        //std::cout << "模型地址: " << &yolo << '\n';

        return 1;
    }

    vector<Object> Dectet(cv::Mat rgb, bool use_gpu)override
    {
        vector<Object> objects;

        int width = rgb.cols;
        int height = rgb.rows;

        // pad to multiple of 32
        int w = width;
        int h = height;
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
        ncnn::Mat in = ncnn::Mat::from_pixels_resize(rgb.data, ncnn::Mat::PIXEL_RGB2BGR, width, height, w, h);

        // pad to target_size rectangle
        int wpad = (w + 31) / 32 * 32 - w;
        int hpad = (h + 31) / 32 * 32 - h;
        ncnn::Mat in_pad;
        ncnn::copy_make_border(in, in_pad, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, ncnn::BORDER_CONSTANT, 0.f);

        in_pad.substract_mean_normalize(0, norm_vals);

        ncnn::Extractor ex = yolo.create_extractor();
        //ex.set_num_threads(4);
        double start_time = ncnn::get_current_time();

        ex.input("images", in_pad);

        std::vector<Object> proposals;

        ncnn::Mat out;
        ex.extract("output", out);

        std::vector<int> strides = { 8, 16, 32 }; // might have stride=64
        std::vector<GridAndStride> grid_strides;
        generate_grids_and_stride(in_pad.w, in_pad.h, strides, grid_strides);
        generate_proposals(grid_strides, out, prob_threshold, proposals);

        // sort all proposals by score from highest to lowest
        qsort_descent_inplace(proposals);

        // apply nms with nms_threshold
        std::vector<int> picked;
        nms_sorted_bboxes(proposals, picked, nms_threshold);

        int count = picked.size();

        objects.resize(count);
        for (int i = 0; i < count; i++)
        {
            objects[i] = proposals[picked[i]];

            // adjust offset to original unpadded
            float x0 = (objects[i].x - (wpad / 2)) / scale;
            float y0 = (objects[i].y - (hpad / 2)) / scale;
            float x1 = (objects[i].x + objects[i].w - (wpad / 2)) / scale;
            float y1 = (objects[i].y + objects[i].h - (hpad / 2)) / scale;

            // clip
            x0 = max(min(x0, (float)(width - 1)), 0.f);
            y0 = max(min(y0, (float)(height - 1)), 0.f);
            x1 = max(min(x1, (float)(width - 1)), 0.f);
            y1 = max(min(y1, (float)(height - 1)), 0.f);

            objects[i].x = x0;
            objects[i].y = y0;
            objects[i].w = x1 - x0;
            objects[i].h = y1 - y0;
        }

        // sort objects by area
        struct
        {
            bool operator()(const Object& a, const Object& b) const
            {
                return a.w * a.h > b.w * b.h;
            }
        } objects_area_greater;
        std::sort(objects.begin(), objects.end(), objects_area_greater);
        double elasped = ncnn::get_current_time() - start_time;
        std::cout << "YoloV5Ncnn " << elasped << "ms   detect\n";
        return objects;
    }

private:
    ncnn::Net yolo;
    int target_size;
    float mean_vals[3] = { 103.53f, 116.28f, 123.675f };
    float norm_vals[3] = { 1 / 255.f, 1 / 255.f, 1 / 255.f };
    float prob_threshold = 0.25;
    float nms_threshold = 0.45;
    ncnn::UnlockedPoolAllocator blob_pool_allocator;
    ncnn::PoolAllocator workspace_pool_allocator;
};