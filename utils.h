#pragma once

#include<string>
#include<vector>
#include <fstream>
#include <sstream>
#include<unordered_map>
#include <filesystem>
#include <ctime>

#include<opencv2/opencv.hpp>
#include <ncnn/layer.h>
#include <ncnn/net.h>
#include <ncnn/benchmark.h>


#include "model.h"
#include"yolo.h"
#include "ResNet.h"
#include "yolo-seg.h"


namespace utils {
    cv::Mat getDectetImage(cv::Mat image, Model* model, vector<string>& classes);//获取检测后画好框的图片
    cv::Mat drawText2Image(cv::Mat image, std::string text);//在图片上添加文字
    cv::Mat drawYoloRect(cv::Mat image, std::vector<Object> boxs, std::vector<std::string>& classes);//在图片画框
    std::vector<cv::Mat> getAllYoloDectetBox(cv::Mat image, std::vector<Object> res);//根据检测框分割图片
    void imageTest();//处理图片文件的测试
    void videoTest();//处理视频文件的测试
    void putFps();//每秒处理多少张图片
    void createFile(string folderPath);//创建文件夹
    void saveImage2Classes(cv::Mat image, vector<Object> res, vector<string> classes);//保存画框标识类别后的图片
    vector<cv::Mat> readJpgFiles(const std::string& folderPath);
    bool Dectet(string path, Model* model, vector<string>& classes, bool saveFlag = true, string savePath = "./output", bool showFlag = false);//目标检测检测图片视频并选择是否保存到本地
    bool DectetSeg(string path, SegModel* model, vector<string>& classes, bool saveFlag = true, string savePath = "./output", bool showFlag = false);//图像分割检测图片视频并选择是否保存到本地
    vector<cv::String> getImagePath(string path);
    vector<cv::String> getVideoPath(string path);

    cv::Mat getDectetImage(cv::Mat image, Model* model, vector<string>& classes){
        auto obj = model->Dectet(image, true);
        return drawYoloRect(image, obj, classes).clone();

    }

    cv::Mat drawText2Image(cv::Mat image, std::string text) {
        cv::Rect rect(0, 0, 40, 20);
        std::vector<cv::Point> contour = { {0,0},{0,20}, {40,20}, {40,0} };
        cv::fillConvexPoly(image, contour, cv::Scalar(255, 255, 255));
        cv::putText(image, text, cv::Point(0, 0 + 10), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.5f, cv::Scalar(0, 0, 255), 1, 1);
        return image;
    }

    cv::Mat drawYoloRect(cv::Mat image, std::vector<Object> boxs, std::vector<std::string>& classes) {
        string text;
        for (auto& box : boxs) {
            if (classes.size() > box.label) {
                //cout << classes.size();
                text = classes[box.label] + ' ' + std::to_string(box.prob);
            }
            else
                text = to_string(box.label) + to_string(box.prob);
            int x = box.x, y = box.y, w = box.w, h = box.h;
            cv::rectangle(image, { x, y, w, h }, cv::Scalar(0, 0, 255));
            cv::putText(image, text, cv::Point(x, y), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.8f, cv::Scalar(0, 0, 255), 1, 1);
        }
        return image;
    }

    std::vector<cv::Mat> getAllYoloDectetBox(cv::Mat image, std::vector<Object> res) {
        std::vector<cv::Mat> ans;
        for (int i = 0; i < res.size(); ++i) {
            cv::Rect roi((int)res[i].x, (int)res[i].y, (int)res[i].w, (int)res[i].h);
            ans.push_back(image(roi).clone());
        }
        return ans;
    }

    std::vector<std::string> cocoClasses = {
                                            "person", "bicycle", "car", "motorcycle", "airplane", "bus",
                                            "train", "truck", "boat", "traffic light", "fire hydrant",
                                            "stop sign", "parking meter", "bench", "bird", "cat", "dog",
                                            "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
                                            "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
                                            "skis", "snowboard", "sports ball", "kite", "baseball bat",
                                            "baseball glove", "skateboard", "surfboard", "tennis racket",
                                            "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
                                            "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
                                            "hot dog", "pizza", "donut", "cake", "chair", "couch",
                                            "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
                                            "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
                                            "toaster", "sink", "refrigerator", "book", "clock", "vase",
                                            "scissors", "teddy bear", "hair drier", "toothbrush"
    };

    std::vector<std::string> colorClasses = {
    "red", "blue", "white", "yellow", "cyan", "purple", "green", "black"
    };

    std::vector<std::string> mainClasses = {
                "TFT",
                "bicycle",
                "car",
                "motorbike",
                "person",
                "qrcode",
                "trafficGreen",
                "trafficRed",
                "trafficSign",
                "trafficYellow",
                "truck"
    };

    std::vector<std::string> shapeClasses = {
         "ju",
         "ling",
         "san",
         "wu",
         "yuan",
         "other"
    };

    std::vector<std::string> yoloShapeClasses = {
        "shape"
    };

    void imageTest() {
        ResNet model;
        ResNet shapeModel;
        Yolov5 yoloModel;
        Yolov5 TFTModel;
        model.Init("./model/color");
        shapeModel.Init("./model/resShape");
        yoloModel.Init("./model/shape");
        TFTModel.Init("./model/main");

        auto image = cv::imread("./images/2.jpg");

        std::vector<Object> res = yoloModel.Dectet(image, true);

        //image = drawYoloRect(image, res, yoloShapeClasses);

        cv::imshow("image", image);
        cv::waitKey(0);
        cv::destroyWindow("image");
        std::vector<cv::Mat> allMat = getAllYoloDectetBox(image, res);
        for (int i = 0; i < allMat.size(); ++i) {
            cv::cvtColor(allMat[i], allMat[i], cv::COLOR_BGR2RGB);
            std::string ans = colorClasses[model.Dectet(allMat[i], true)[0].label];
            ans += "   " + shapeClasses[shapeModel.Dectet(allMat[i], true)[0].label];
            cout << ans << '\n';
            cv::cvtColor(allMat[i], allMat[i], cv::COLOR_RGB2BGR);
            cv::imshow("image", allMat[i]);
            cv::waitKey(3000);
            cv::destroyWindow("image");
        }
    }

    void outImageTest(string path, string modelPath, vector<string> classes = shapeClasses) {
        srand(time(0));
        Yolov5 yoloModel;
        yoloModel.Init(modelPath);
        vector<cv::Mat> images = readJpgFiles(path);
        for (int i = 0; i < images.size(); ++i) {
            auto tmp = yoloModel.Dectet(images[i], true);
            saveImage2Classes(images[i], tmp, classes);
            
        }

    }

    int dectetfps = 0;

    void putFps() {
        int cnt = 0;
        while (cnt < 5) {
            if (dectetfps)cnt = 0;
            else cnt++;
            std::cout << "fps:  " << dectetfps << '\n';
            dectetfps = 0;
            Sleep(1000);
        }
    }

    void videoTest() {
        cv::VideoCapture video("./video/1.mp4");
        Yolov5 yoloModel;
        yoloModel.Init("./model/main");
        if (!video.isOpened()) {
            std::cout << "视频未找到\n";
            return;
        }

        int width = video.get(cv::CAP_PROP_FRAME_WIDTH);
        int height = video.get(cv::CAP_PROP_FRAME_HEIGHT);
        double fps = video.get(cv::CAP_PROP_FPS);

        cv::VideoWriter outputVideo("./output/output.mp4", cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps, cv::Size(width, height));

        std::thread myThread(putFps);

        cv::Mat frame;
        while (video.read(frame)) {
            frame = drawYoloRect(frame, yoloModel.Dectet(frame, true), mainClasses);
            dectetfps++;
            outputVideo.write(frame);
        }
        video.release();
        outputVideo.release();

    }

    void createFile(string folderPath) {
        struct stat info;
        if (stat(folderPath.c_str(), &info) != 0) { // 检查文件夹是否存在
            std::filesystem::create_directory(folderPath);
        }
 
    }

    void saveImage2Classes(cv::Mat image, vector<Object> res, vector<string> classes) {
        for (int i = 0; i < res.size(); ++i) {
            cv::Rect roi((int)res[i].x, (int)res[i].y, (int)res[i].w, (int)res[i].h);
            auto tmpImage = image(roi);
            string path = "./output/" + classes[res[i].label] + "/" + to_string(rand()) + ".jpg";
            createFile("./output/" + classes[res[i].label]);
            cv::imwrite(path, tmpImage);
        }
    }

    vector<cv::Mat> readJpgFiles(const std::string& folderPath) {
        vector<cv::Mat> images;
        cv::String pattern = folderPath + "/*.jpg";
        std::vector<cv::String> fileNames;
        cv::glob(pattern, fileNames, true);

        for (const auto& fileName : fileNames) {
            cv::Mat image = cv::imread(fileName);
            if (!image.empty()) {
                images.push_back(image);
            }
        }
        return images;
    }

    vector<cv::String> getVideoPath(string path) {
        vector<string> videoPath = { ".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".mpeg", ".asf", ".3gp" };
       
        std::vector<cv::String> fileNames;
        for (int i = 0; i < videoPath.size(); ++i) {
            cv::String pattern = path + "/*" + videoPath[i];
            std::vector<cv::String> tmp;
            cv::glob(pattern, tmp, true);
            for (int j = 0; j < tmp.size(); ++j) {
                fileNames.push_back(tmp[j]);
            }
        }
        return fileNames;
    }

    vector<cv::String> getImagePath(string path) {
        vector<string> ImagePath = { ".jpg", ".png", ".bmp", ".tiff", ".gif", ".pbm", ".pgm", ".ppm", ".hdr", ".ico"};

        std::vector<cv::String> fileNames;
        for (int i = 0; i < ImagePath.size(); ++i) {
            cv::String pattern = path + "/*" + ImagePath[i];
            std::vector<cv::String> tmp;
            cv::glob(pattern, tmp, true);
            for (int j = 0; j < tmp.size(); ++j) {
                fileNames.push_back(tmp[j]);
            }
        }
        return fileNames;
    }

    vector<int> yolo2xyxy(int img_width, int img_height, vector<float> bbox) {
        // [xmin, ymin, width, height] ->[x, y, width, height](归一化)
        double W = img_width;
        double H = img_height;
        int classes = bbox[0];
        double x = bbox[1];
        double y = bbox[2];
        double w = bbox[3];
        double h = bbox[4];

        int x1 = int((x - w / 2) * W);  
        int y1 = int((y - h / 2) * H);
        int x2 = int((x + w / 2) * W);
        int y2 = int((y + h / 2) * H);

        return { classes, x1, y1, x2, y2 };
    }

    vector<vector<int>> getYoloRect(string path) {
        std::ifstream file(path); // 打开txt文件
        std::string line;
        std::vector<std::string> words;
        vector<vector<int>> ans;
        if (file.is_open()) {
            while (std::getline(file, line)) { // 逐行读取文件内容
                std::istringstream iss(line);
                std::string word;
                while (iss >> word) { // 按照空格分割每行的单词
                    words.push_back(word);
                }
                vector<float> yoloRect;
                for (int i = 0; i < words.size(); ++i) {
                    yoloRect.push_back(stof(words[i]));
                }
                words.clear();
                ans.push_back(yolo2xyxy(640, 640, yoloRect));
            }
            file.close(); // 关闭文件
        }
        else {
            std::cout << "无法打开文件" << std::endl;
            return {};
        }

        return ans;
    }

    bool checkFile(string path) {
        for (int i = 1; i < path.size(); ++i) {
            if (path[i] == '.')return false;
        }
        return true;
    }
    
    bool checkImage(string path) {
        vector<string> ImagePath = { ".jpg", ".png", ".bmp", ".tiff", ".gif", ".pbm", ".pgm", ".ppm", ".hdr", ".ico" };
        for (int i = 0; i < ImagePath.size(); ++i) {
            if (path.find(ImagePath[i]) != string::npos)return true;
        }
        return false;
    }

    bool checkVideo(string path) {
        vector<string> videoPath = { ".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".mpeg", ".asf", ".3gp" };
        for (int i = 0; i < videoPath.size(); ++i) {
            if (path.find(videoPath[i]) != string::npos)return true;
        }
        return false;
    }

    void dectetImage(string path, Model* model, vector<string>& classes, bool saveFlag, string savePath, bool showFlag) {
        cv::Mat image = cv::imread(path);
        std::vector<Object> res = model->Dectet(image, true);
        image = drawYoloRect(image, res, classes);
        if (saveFlag) {
            string saveName = savePath + '/' + to_string(rand()) + ".jpg";
            cv::imwrite(saveName, image);
        }
        if (showFlag) {
            cv::imshow("image", image);
            cv::waitKey(0);
        }

    }

    void dectetVideo(string path, Model* model, vector<string>& classes, bool saveFlag, string savePath, bool showFlag) {
        cv::VideoCapture video(path);
        if (!video.isOpened()) {
            std::cout << "视频未找到\n";
            return;
        }

        int width = video.get(cv::CAP_PROP_FRAME_WIDTH);
        int height = video.get(cv::CAP_PROP_FRAME_HEIGHT);
        double fps = video.get(cv::CAP_PROP_FPS);
        int total_frames = int(video.get(cv::CAP_PROP_FRAME_COUNT));
        string saveName = savePath + '/' + to_string(rand()) + ".mp4";
        vector<cv::Mat> images;
        dectetfps = 0;
        cv::Mat frame;
        while (video.read(frame)) {
            frame = drawYoloRect(frame, model->Dectet(frame, true), classes);//画框函数
            dectetfps++;
            if (saveFlag)
                images.push_back(frame.clone());
            cout << dectetfps << '/' << total_frames << endl;
            if (showFlag) {
                cv::imshow("image", frame);
                cv::waitKey(1);
            }
        }
        if(showFlag)cv::destroyWindow("image");
        if (saveFlag) {
            cv::VideoWriter outputVideo(saveName, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps, cv::Size(width, height));
            for(int i = 0;i < images.size();++i)
            outputVideo.write(images[i]);
            outputVideo.release();
        }
        video.release();

    }

    bool ModelPerformanceTesting(string imagePath, string labelPath, Model* model) {
        vector<cv::String> imagePathList;
        imagePathList = getImagePath(imagePath);
        double mAp = 0;
        int allLabelNum = 0;
        for (int i = 0; i < imagePathList.size(); ++i) {
            cv::Mat image = cv::imread(imagePathList[i], cv::IMREAD_COLOR);
            cv::resize(image, image, cv::Size(640, 640));
            vector<Object> res = model->Dectet(image, true);
            vector<vector<int>> ans;
            string labelName = imagePathList[i];
            while (labelName.back() != '.')labelName.pop_back();
            labelName += "txt";
            string tmp;
            while (labelName.back() != '\\') {
                tmp = labelName.back() + tmp;
                labelName.pop_back();
            }
            labelName.pop_back();
            labelName.pop_back();
            while (labelName.back() != '\\') {
                labelName.pop_back();
            }
            vector<vector<int>> yoloRect = getYoloRect(labelPath + "\\" + tmp);
            //cout << labelPath + "\\" + tmp << '\n';
            if (yoloRect.empty())continue;
            /*            
            for (int j = 0; j < yoloRect.size(); ++j) {
                cout << res[j].label << ' ' << (int)res[j].x << ' ' << (int)res[j].y << ' ' << (int)res[j].x + (int)res[j].w << ' ' << (int)res[j].y + (int)res[j].h << '\n';
            }*/

            
            allLabelNum += res.size();
            for (int j = 0; j < res.size(); ++j) {
                ans.push_back({ res[j].label,(int)res[j].x,(int)res[j].y, (int)res[j].x + (int)res[j].w,(int)res[j].y + (int)res[j].h });
            }
            
            for (int j = 0; j < ans.size(); ++j) {
                double maxIou = 0;
                for (int k = 0; k < yoloRect.size(); ++k) {
                    if (ans[j][0] == yoloRect[k][0]) {
                        int iouX1 = max(yoloRect[k][1], ans[j][1]);
                        int iouY1 = max(yoloRect[k][2], ans[j][2]);
                        int iouX2 = min(yoloRect[k][3], ans[j][3]);
                        int iouY2 = min(yoloRect[k][4], ans[j][4]);
                        double area1 = (iouY2 - iouY1) * (iouX2 - iouX1) * 1.0;
                        double area2 = (yoloRect[k][3] - yoloRect[k][1]) * (yoloRect[k][4] - yoloRect[k][2]) * 1.0;
                        if (area1 > area2)swap(area1, area2);
                        if (area1 / area2 < 0)continue;
                        maxIou = max(maxIou, area1 / area2);
                        
                    }
                }
                mAp += maxIou;
                //cout << maxIou << '\n';
            }
        }
        cout << "检测框贴合度: " << mAp / (allLabelNum * 1.0) << '\n';
    }

    bool Dectet(string path, Model* model, vector<string>& classes, bool saveFlag, string savePath, bool showFlag) {
        srand(time(nullptr));
        vector<cv::String> imagePath, videoPath;
        createFile(savePath);
        savePath += "/dectet1";
        int tmp = 1;
        struct stat info;
        
        while (stat(savePath.c_str(), &info) == 0) {
            //cout << savePath << '\n';
            while (savePath.back() >= '0' && savePath.back() <= '9')savePath.pop_back();
            savePath += to_string(++tmp);
        }
        createFile(savePath);
        if (checkFile(path)) {
            imagePath = getImagePath(path);
            videoPath = getVideoPath(path);
            cout << "Find " << imagePath.size() << "image\n";
            cout << "Find " << videoPath.size() << "video\n";
            for (int i = 0; i < imagePath.size(); ++i) {
                dectetImage(imagePath[i], model, classes, saveFlag, savePath, showFlag);
                cout << i + 1 << "/" << imagePath.size() << '\n';
            }
            for (int i = 0; i < videoPath.size(); ++i) {
                dectetVideo(videoPath[i], model, classes, saveFlag, savePath, showFlag);
                cout << i + 1 << "/" << videoPath.size() << '\n';
            }
        }
        else {
            if (checkImage(path)) {
                dectetImage(path, model, classes, saveFlag, savePath, showFlag);
            }
            else if(checkVideo(path))
                dectetVideo(path, model, classes, saveFlag, savePath, showFlag);
            else {
                cout << "Not file or image or video path\n";
            }
        }
    }

    void dectetSegImage(string path, SegModel* model, vector<string>& classes, bool saveFlag, string savePath, bool showFlag) {
        cv::Mat image = cv::imread(path, cv::IMREAD_COLOR);
        vector<ObjectSeg> objects = model->Dectet(image, true);
        image = model->draw_objects(image, objects, classes);
        if (saveFlag) {
            string saveName = savePath + '/' + to_string(rand()) + ".jpg";
            cv::imwrite(saveName, image);
        }
        if (showFlag) {
            cv::imshow("image", image);
            cv::waitKey(0);
        }

    }

    void dectetSegVideo(string path, SegModel* model, vector<string>& classes, bool saveFlag, string savePath, bool showFlag) {
        cv::VideoCapture video(path);
        if (!video.isOpened()) {
            std::cout << "视频未找到\n";
            return;
        }
        int width = video.get(cv::CAP_PROP_FRAME_WIDTH);
        int height = video.get(cv::CAP_PROP_FRAME_HEIGHT);
        double fps = video.get(cv::CAP_PROP_FPS);
        int total_frames = int(video.get(cv::CAP_PROP_FRAME_COUNT));
        string saveName = savePath + '/' + to_string(rand()) + ".mp4";
        vector<cv::Mat> images;
        dectetfps = 0;
        cv::Mat frame;
        while (video.read(frame)) {
            vector<ObjectSeg> objects = model->Dectet(frame, true);
            frame = model->draw_objects(frame, objects, classes);//画框函数
            dectetfps++;
            if (saveFlag)
                images.push_back(frame.clone());
            cout << dectetfps << '/' << total_frames << endl;
            if (showFlag) {
                cv::imshow("image", frame);
                cv::waitKey(1);
            }
        }
        if (showFlag)cv::destroyWindow("image");
        if (saveFlag) {
            cv::VideoWriter outputVideo(saveName, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps, cv::Size(width, height));
            for (int i = 0; i < images.size(); ++i)
                outputVideo.write(images[i]);
            outputVideo.release();
        }
        video.release();

    }

    bool DectetSeg(string path, SegModel* model, vector<string>& classes, bool saveFlag, string savePath, bool showFlag) {
        srand(time(nullptr));
        vector<cv::String> imagePath, videoPath;
        createFile(savePath);
        savePath += "/dectet1";
        int tmp = 1;
        struct stat info;

        while (stat(savePath.c_str(), &info) == 0) {
            //cout << savePath << '\n';
            while (savePath.back() >= '0' && savePath.back() <= '9')savePath.pop_back();
            savePath += to_string(++tmp);
        }
        createFile(savePath);
        if (checkFile(path)) {
            imagePath = getImagePath(path);
            videoPath = getVideoPath(path);
            cout << "Find " << imagePath.size() << "image\n";
            cout << "Find " << videoPath.size() << "video\n";
            for (int i = 0; i < imagePath.size(); ++i) {
                dectetSegImage(imagePath[i], model, classes, saveFlag, savePath, showFlag);
                cout << i + 1 << "/" << imagePath.size() << '\n';
            }
            for (int i = 0; i < videoPath.size(); ++i) {
                dectetSegVideo(videoPath[i], model, classes, saveFlag, savePath, showFlag);
                cout << i + 1 << "/" << videoPath.size() << '\n';
            }
        }
        else {
            if (checkImage(path)) {
                dectetSegImage(path, model, classes, saveFlag, savePath, showFlag);
            }
            else if (checkVideo(path))
                dectetSegVideo(path, model, classes, saveFlag, savePath, showFlag);
            else {
                cout << "Not file or image or video path\n";
            }
        }
    }

}
