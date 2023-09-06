#pragma once
#include<opencv2/opencv.hpp>
#include<string>
#include<vector>
#include<unordered_map>
#include <ncnn/layer.h>
#include <ncnn/net.h>
#include <ncnn/benchmark.h>
#include "model.h"
#include"windows.h"
#include <filesystem>
#include <ctime>

namespace utils {
    cv::Mat getDectetImage(cv::Mat image, Model* model, vector<string>& classes);
    cv::Mat drawText2Image(cv::Mat image, std::string text);
    cv::Mat drawYoloRect(cv::Mat image, std::vector<Object> boxs, std::vector<std::string>& classes);
    std::vector<cv::Mat> getAllYoloDectetBox(cv::Mat image, std::vector<Object> res);
    void imageTest();
    void videoTest();
    void putFps();
    void createFile(string folderPath);
    void saveImage2Classes(cv::Mat image, vector<Object> res, vector<string> classes);
    vector<cv::Mat> readJpgFiles(const std::string& folderPath);
    bool Dectet(string path, Model* model, vector<string>& classes, bool saveFlag = true, string savePath = "./output", bool showFlag = false);
    vector<cv::String> getImagePath(string path);
    vector<cv::String> getVideoPath(string path);

    cv::Mat getDectetImage(cv::Mat image, Model* model, vector<string>& classes){
        auto obj = model->Detect(image, true);
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
                text = classes[box.label] + std::to_string(box.prob);
            }
            else
                text = to_string(box.label) + to_string(box.prob);
            int x = box.x, y = box.y, w = box.w, h = box.h;
            cv::rectangle(image, { x, y, w, h }, cv::Scalar(0, 0, 0));
            cv::putText(image, text, cv::Point(x, y), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.8f, cv::Scalar(0, 0, 255), 1, 1);
        }
        return image;
    }

    std::vector<cv::Mat> getAllYoloDectetBox(cv::Mat image, std::vector<Object> res) {
        std::vector<cv::Mat> ans;
        for (int i = 0; i < res.size(); ++i) {
            cv::Rect roi((int)res[i].x, (int)res[i].y, (int)res[i].w, (int)res[i].h);
            ans.push_back(image(roi));
        }
        return ans;
    }

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

        cv::cvtColor(image, image, cv::COLOR_BGR2RGB);

        std::vector<Object> res = yoloModel.Detect(image, true);

        cv::cvtColor(image, image, cv::COLOR_RGB2BGR);

        //image = drawYoloRect(image, res, yoloShapeClasses);

        cv::imshow("image", image);
        cv::waitKey(0);
        cv::destroyWindow("image");
        std::vector<cv::Mat> allMat = getAllYoloDectetBox(image, res);
        for (int i = 0; i < allMat.size(); ++i) {
            cv::cvtColor(allMat[i], allMat[i], cv::COLOR_BGR2RGB);
            std::string ans = colorClasses[model.Detect(allMat[i], true)[0].label];
            ans += "   " + shapeClasses[shapeModel.Detect(allMat[i], true)[0].label];
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
            auto tmp = yoloModel.Detect(images[i], true);
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
            frame = drawYoloRect(frame, yoloModel.Detect(frame, true), mainClasses);
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
        cv::Mat image = cv::imread(path, cv::IMREAD_COLOR);
        cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
        std::vector<Object> res = model->Detect(image, true);
        cv::cvtColor(image, image, cv::COLOR_RGB2BGR);
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
            frame = drawYoloRect(frame, model->Detect(frame, true), classes);//画框函数
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

    bool Dectet(string path, Model* model, vector<string>& classes, bool saveFlag, string savePath, bool showFlag) {
        srand(time(nullptr));
        vector<cv::String> imagePath, videoPath;
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
        cout << "总测试时长" << timeSum << '\n';
    }

}
