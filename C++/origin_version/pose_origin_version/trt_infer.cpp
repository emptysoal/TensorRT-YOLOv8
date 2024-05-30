#include "public.h"
#include "utils.h"
#include "preprocess.h"
#include "postprocess.h"
#include "calibrator.h"
#include <NvOnnxParser.h>

using namespace nvinfer1;


const int kGpuId = 0;
const int kNumClass = 1;
const int kNumKpt = 17;  // 单个目标对应的关键点的个数
const int kKptDims = 3;  // 单个关键点的维度，2 for x,y or 3 for x,y,visible
const int kInputH = 640;
const int kInputW = 640;
const float kNmsThresh = 0.45f;
const float kConfThresh = 0.25f;
const int kMaxNumOutputBbox = 1000;  // assume the box outputs no more than kMaxNumOutputBbox boxes that conf >= kNmsThresh;
const int kNumBoxElement = 7 + kNumKpt * kKptDims;  // left, top, right, bottom, confidence, class, keepflag(whether drop when NMS), 51 keypoints

const std::string onnxFile = "./yolov8s-pose.onnx";
const std::string trtFile = "./yolov8s-pose.plan";
const std::string testDataDir = "../images";  // 用于推理

static Logger gLogger(ILogger::Severity::kERROR);

// for FP16 mode
const bool bFP16Mode = false;
// for INT8 mode
const bool bINT8Mode = false;
const std::string cacheFile = "./int8.cache";
const std::string calibrationDataPath = "../calibrator";  // 用于 int8 量化


struct Detection
{
    float bbox[4];  // x1, y1, x2, y2
    float conf;
    float classId;
    float kpts[kNumKpt * kKptDims];
};


cv::Rect get_rect(cv::Mat& img, float bbox[4]) {
    float l, r, t, b;
    float r_w = kInputW / (img.cols * 1.0);
    float r_h = kInputH / (img.rows * 1.0);

    if (r_h > r_w) {
        l = bbox[0];
        r = bbox[2];
        t = bbox[1] - (kInputH - r_w * img.rows) / 2;
        b = bbox[3] - (kInputH - r_w * img.rows) / 2;
        l = l / r_w;
        r = r / r_w;
        t = t / r_w;
        b = b / r_w;
    } else {
        l = bbox[0] - (kInputW - r_h * img.cols) / 2;
        r = bbox[2] - (kInputW - r_h * img.cols) / 2;
        t = bbox[1];
        b = bbox[3];
        l = l / r_h;
        r = r / r_h;
        t = t / r_h;
        b = b / r_h;
    }
    return cv::Rect(round(l), round(t), round(r - l), round(b - t));
}


std::vector<std::vector<float>> scale_kpt_coords(cv::Mat& img, float* pkpt){
    float r_w = kInputW / (img.cols * 1.0);
    float r_h = kInputH / (img.rows * 1.0);
    float r = std::min(r_w, r_h);
    float pad_h = (kInputH - r * img.rows) / 2;
    float pad_w = (kInputW - r * img.cols) / 2;

    std::vector<std::vector<float>> vScaledKpts;
    float x;
    float y;
    float conf;
    float* pSingleKpt;
    for (int i = 0; i < kNumKpt; i++){
        pSingleKpt = pkpt + i * kKptDims;
        x = pSingleKpt[0] - pad_w;
        y = pSingleKpt[1] - pad_h;
        x = x / r;
        y = y / r;
        conf = pSingleKpt[2];
        std::vector<float> scaledKpt {x, y, conf};
        vScaledKpts.push_back(scaledKpt);
    }

    return vScaledKpts;
}


ICudaEngine* getEngine(){
    ICudaEngine* engine = nullptr;

    if (access(trtFile.c_str(), F_OK) == 0){
        std::ifstream engineFile(trtFile, std::ios::binary);
        long int fsize = 0;

        engineFile.seekg(0, engineFile.end);
        fsize = engineFile.tellg();
        engineFile.seekg(0, engineFile.beg);
        std::vector<char> engineString(fsize);
        engineFile.read(engineString.data(), fsize);
        if (engineString.size() == 0) { std::cout << "Failed getting serialized engine!" << std::endl; return nullptr; }
        std::cout << "Succeeded getting serialized engine!" << std::endl;

        IRuntime* runtime = createInferRuntime(gLogger);
        engine = runtime->deserializeCudaEngine(engineString.data(), fsize);
        if (engine == nullptr) { std::cout << "Failed loading engine!" << std::endl; return nullptr; }
        std::cout << "Succeeded loading engine!" << std::endl;
    } else {
        IBuilder *            builder     = createInferBuilder(gLogger);
        INetworkDefinition *  network     = builder->createNetworkV2(1U << int(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));
        IOptimizationProfile* profile     = builder->createOptimizationProfile();
        IBuilderConfig *      config      = builder->createBuilderConfig();
        config->setMaxWorkspaceSize(1 << 30);
        IInt8Calibrator *     pCalibrator = nullptr;
        if (bFP16Mode){
            config->setFlag(BuilderFlag::kFP16);
        }
        if (bINT8Mode){
            config->setFlag(BuilderFlag::kINT8);
            int batchSize = 8;
            pCalibrator = new Int8EntropyCalibrator2(batchSize, kInputW, kInputH, calibrationDataPath.c_str(), cacheFile.c_str());
            config->setInt8Calibrator(pCalibrator);
        }

        nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, gLogger);
        if (!parser->parseFromFile(onnxFile.c_str(), int(gLogger.reportableSeverity))){
            std::cout << std::string("Failed parsing .onnx file!") << std::endl;
            for (int i = 0; i < parser->getNbErrors(); ++i){
                auto *error = parser->getError(i);
                std::cout << std::to_string(int(error->code())) << std::string(":") << std::string(error->desc()) << std::endl;
            }
            return nullptr;
        }
        std::cout << std::string("Succeeded parsing .onnx file!") << std::endl;

        ITensor* inputTensor = network->getInput(0);
        profile->setDimensions(inputTensor->getName(), OptProfileSelector::kMIN, Dims32 {4, {1, 3, kInputH, kInputW}});
        profile->setDimensions(inputTensor->getName(), OptProfileSelector::kOPT, Dims32 {4, {1, 3, kInputH, kInputW}});
        profile->setDimensions(inputTensor->getName(), OptProfileSelector::kMAX, Dims32 {4, {1, 3, kInputH, kInputW}});
        config->addOptimizationProfile(profile);

        IHostMemory *engineString = builder->buildSerializedNetwork(*network, *config);
        std::cout << "Succeeded building serialized engine!" << std::endl;

        IRuntime* runtime = createInferRuntime(gLogger);
        engine = runtime->deserializeCudaEngine(engineString->data(), engineString->size());
        if (engine == nullptr) { std::cout << "Failed building engine!" << std::endl; return nullptr; }
        std::cout << "Succeeded building engine!" << std::endl;

        if (bINT8Mode && pCalibrator != nullptr){
            delete pCalibrator;
        }

        std::ofstream engineFile(trtFile, std::ios::binary);
        engineFile.write(static_cast<char *>(engineString->data()), engineString->size());
        std::cout << "Succeeded saving .plan file!" << std::endl;
    }

    return engine;
}


int run(){
    ICudaEngine* engine = getEngine();

    IExecutionContext* context = engine->createExecutionContext();
    context->setBindingDimensions(0, Dims32 {4, {1, 3, kInputH, kInputW}});

    // get engine output info
    Dims32 outDims = context->getBindingDimensions(1);  // [1, 56, 8400], 56 = 4 + 1 + 51 = bbox + class + keypoints
    int OUTPUT_CANDIDATES = outDims.d[2];  // 8400
    int outputSize = 1;  // 56 * 8400
    for (int i = 0; i < outDims.nbDims; i++){
        outputSize *= outDims.d[i];
    }

    // prepare input data and output data ---------------------------
    static float inputData[3 * kInputH * kInputW];
    static float outputData[1 + kMaxNumOutputBbox * kNumBoxElement];
    // prepare input and output space on device
    std::vector<void*> vBufferD(2, nullptr);
    CHECK(cudaMalloc(&vBufferD[0], 3 * kInputH * kInputW * sizeof(float)));
    CHECK(cudaMalloc(&vBufferD[1], outputSize * sizeof(float)));

    float* transposeDevide = nullptr;
    CHECK(cudaMalloc(&transposeDevide, outputSize * sizeof(float)));

    float* decodeDevice = nullptr;
    CHECK(cudaMalloc(&decodeDevice, (1 + kMaxNumOutputBbox * kNumBoxElement) * sizeof(float)));

    // get image file names for inferencing
    std::vector<std::string> file_names;
    if (read_files_in_dir(testDataDir.c_str(), file_names) < 0) {
        std::cout << "read_files_in_dir failed." << std::endl;
        return -1;
    }

    // inference
    for (int i = 0; i < file_names.size(); i++){
        std::string testImagePath = testDataDir + "/" + file_names[i];
        cv::Mat img = cv::imread(testImagePath, cv::IMREAD_COLOR);
        if (img.empty()) continue;

        auto start = std::chrono::system_clock::now();

        preprocess(img, inputData, kInputH, kInputW);  // put image data on inputData

        CHECK(cudaMemcpy(vBufferD[0], (void *)inputData, 3 * kInputH * kInputW * sizeof(float), cudaMemcpyHostToDevice));
        context->executeV2(vBufferD.data());
        // transpose [56 8400] convert to [8400 56]
        transpose((float*)vBufferD[1], transposeDevide, OUTPUT_CANDIDATES, 4 + kNumClass + kNumKpt * kKptDims);
        // convert [8400 56] to [58001, ], 58001 = 1 + 1000 * (4bbox + cond + cls + keepflag + 51kpts)
        // CHECK(cudaMemset(decodeDevice, 0, sizeof(int)));
        int nk = kNumKpt * kKptDims;  // number of keypoints total, default 51
        decode(transposeDevide, decodeDevice, OUTPUT_CANDIDATES, kNumClass, nk, kConfThresh, kMaxNumOutputBbox, kNumBoxElement);
        // cuda nms
        nms(decodeDevice, kNmsThresh, kMaxNumOutputBbox, kNumBoxElement);
        CHECK(cudaMemcpy(outputData, decodeDevice, (1 + kMaxNumOutputBbox * kNumBoxElement) * sizeof(float), cudaMemcpyDeviceToHost));

        std::vector<Detection> vDetections;
        int count = std::min((int)outputData[0], kMaxNumOutputBbox);
        for (int i = 0; i < count; i++){
            int pos = 1 + i * kNumBoxElement;
            int keepFlag = (int)outputData[pos + 6];
            if (keepFlag == 1){
                Detection det;
                memcpy(det.bbox, &outputData[pos], 4 * sizeof(float));
                det.conf = outputData[pos + 4];
                det.classId = outputData[pos + 5];
                memcpy(det.kpts, &outputData[pos + 7], kNumKpt * kKptDims * sizeof(float));
                vDetections.push_back(det);
            }
        }

        for (size_t i = 0; i < vDetections.size(); i++){
            cv::Rect r = get_rect(img, vDetections[i].bbox);
            cv::rectangle(img, r, cv::Scalar(255, 0, 255), 2);
            cv::putText(img, std::to_string((int)vDetections[i].classId), cv::Point(r.x, r.y - 1), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(255, 255, 255), 2);

            std::vector<std::vector<float>> vScaledKpts = scale_kpt_coords(img, vDetections[i].kpts);
            for (int j = 0; j < vScaledKpts.size(); j++){
                int x = (int)vScaledKpts[j][0];
                int y = (int)vScaledKpts[j][1];
                float conf = vScaledKpts[j][2];
                if (x < 0 || x > img.cols || y < 0 || y > img.rows) continue;
                if (conf < 0.5) continue;
                cv::circle(img, cv::Point(x, y), 5, cv::Scalar(0, 255, 0), -1);
            }
        }

        auto end = std::chrono::system_clock::now();
        int cost = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        std::cout << file_names[i] << " cost: " << cost << " ms"  << std::endl;

        cv::imwrite("_" + file_names[i], img);
    }

    // free device memory
    for (int i = 0; i < 2; ++i)
    {
        CHECK(cudaFree(vBufferD[i]));
    }

    delete context;
    delete engine;

    CHECK(cudaFree(transposeDevide));
    CHECK(cudaFree(decodeDevice));

    return 0;
}


int main(){
    CHECK(cudaSetDevice(kGpuId));
    run();
    return 0;
}
