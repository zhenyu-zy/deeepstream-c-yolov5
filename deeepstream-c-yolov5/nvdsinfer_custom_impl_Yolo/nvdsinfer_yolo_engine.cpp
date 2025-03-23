#include <algorithm>

#include "nvdsinfer_custom_impl.h"
#include "nvdsinfer_context.h"

#include "yolo.h"

#define USE_CUDA_ENGINE_GET_API 1  // 选择是否使用 CUDA 引擎 API

/**
 * @brief 解析并填充 YOLO 网络的相关信息。
 * @param networkInfo  存储解析出的网络信息。
 * @param initParams   传入的初始化参数，包含模型路径等信息。
 * @return 若解析成功返回 true，否则返回 false。
 */
static bool getYoloNetworkInfo(NetworkInfo& networkInfo, const NvDsInferContextInitParams* initParams) {
    // 获取 ONNX、权重（WTS）和配置（CFG）文件路径
    std::string onnxFilePath = initParams->onnxFilePath;
    std::string wtsFilePath = initParams->modelFilePath;
    std::string cfgFilePath = initParams->customNetworkConfigFilePath;

    // 判断模型类型（ONNX 或 Darknet）
    std::string yoloType = !onnxFilePath.empty() ? "onnx" : "darknet";
    // 提取模型名称
    std::string modelName = yoloType == "onnx" ?
        onnxFilePath.substr(0, onnxFilePath.find(".onnx")).substr(onnxFilePath.rfind("/") + 1) :
        cfgFilePath.substr(0, cfgFilePath.find(".cfg")).substr(cfgFilePath.rfind("/") + 1);

    // 转换模型名称为小写
    std::transform(modelName.begin(), modelName.end(), modelName.begin(), [](uint8_t c) {
        return std::tolower(c);
    });

    // 填充 networkInfo 结构体
    networkInfo.inputBlobName = "input";
    networkInfo.networkType = yoloType;
    networkInfo.modelName = modelName;
    networkInfo.onnxFilePath = onnxFilePath;
    networkInfo.wtsFilePath = wtsFilePath;
    networkInfo.cfgFilePath = cfgFilePath;
    networkInfo.batchSize = initParams->maxBatchSize;
    networkInfo.implicitBatch = initParams->forceImplicitBatchDimension;
    networkInfo.int8CalibPath = initParams->int8CalibrationFilePath;
    networkInfo.deviceType = initParams->useDLA ? "kDLA" : "kGPU";
    networkInfo.numDetectedClasses = initParams->numDetectedClasses;
    networkInfo.clusterMode = initParams->clusterMode;
    networkInfo.scaleFactor = initParams->networkScaleFactor;
    networkInfo.offsets = initParams->offsets;
    networkInfo.workspaceSize = initParams->workspaceSize;
    networkInfo.inputFormat = initParams->networkInputFormat;

    // 设置网络计算精度（FP32、FP16、INT8）
    if (initParams->networkMode == NvDsInferNetworkMode_FP32) {
        networkInfo.networkMode = "FP32";
    } else if (initParams->networkMode == NvDsInferNetworkMode_INT8) {
        networkInfo.networkMode = "INT8";
    } else if (initParams->networkMode == NvDsInferNetworkMode_FP16) {
        networkInfo.networkMode = "FP16";
    }

    // 检查文件是否存在
    if (yoloType == "onnx") {
        if (!fileExists(networkInfo.onnxFilePath)) {
            std::cerr << "ONNX file does not exist\n" << std::endl;
            return false;
        }
    } else {
        if (!fileExists(networkInfo.wtsFilePath)) {
            std::cerr << "Darknet weights file does not exist\n" << std::endl;
            return false;
        }
        if (!fileExists(networkInfo.cfgFilePath)) {
            std::cerr << "Darknet cfg file does not exist\n" << std::endl;
            return false;
        }
    }
    return true;
}

#if !USE_CUDA_ENGINE_GET_API
/**
 * @brief 旧版本 TensorRT 使用的模型解析器。
 * @param initParams 初始化参数。
 * @return IModelParser* 指向新创建的 Yolo 解析器实例。
 */
IModelParser* NvDsInferCreateModelParser(const NvDsInferContextInitParams* initParams) {
    NetworkInfo networkInfo;
    if (!getYoloNetworkInfo(networkInfo, initParams)) {
        return nullptr;
    }
    return new Yolo(networkInfo);
}
#else

#if NV_TENSORRT_MAJOR >= 8
/**
 * @brief TensorRT 8+ 版本使用的 CUDA 引擎创建函数。
 */
extern "C" bool NvDsInferYoloCudaEngineGet(
    nvinfer1::IBuilder* const builder,
    nvinfer1::IBuilderConfig* const builderConfig,
    const NvDsInferContextInitParams* const initParams,
    nvinfer1::DataType dataType,
    nvinfer1::ICudaEngine*& cudaEngine);
#else
/**
 * @brief 旧版本 TensorRT 使用的 CUDA 引擎创建函数。
 */
extern "C" bool NvDsInferYoloCudaEngineGet(
    nvinfer1::IBuilder* const builder,
    const NvDsInferContextInitParams* const initParams,
    nvinfer1::DataType dataType,
    nvinfer1::ICudaEngine*& cudaEngine);
#endif

/**
 * @brief 创建 YOLO 的 TensorRT CUDA 引擎。
 * @param builder       TensorRT Builder。
 * @param builderConfig TensorRT BuilderConfig（仅适用于 TensorRT 8+）。
 * @param initParams    传入的初始化参数。
 * @param dataType      指定的数据类型（FP32、FP16、INT8）。
 * @param cudaEngine    输出的 CUDA 引擎指针。
 * @return 若成功返回 true，否则返回 false。
 */
extern "C" bool NvDsInferYoloCudaEngineGet(
    #if NV_TENSORRT_MAJOR >= 8
    nvinfer1::IBuilder* const builder,
    nvinfer1::IBuilderConfig* const builderConfig,
    #endif
    const NvDsInferContextInitParams* const initParams,
    nvinfer1::DataType dataType,
    nvinfer1::ICudaEngine*& cudaEngine) {
    
    NetworkInfo networkInfo;
    if (!getYoloNetworkInfo(networkInfo, initParams)) {
        return false;
    }

    Yolo yolo(networkInfo);

    // 创建 TensorRT 推理引擎
    #if NV_TENSORRT_MAJOR >= 8
    cudaEngine = yolo.createEngine(builder, builderConfig);
    #else
    cudaEngine = yolo.createEngine(builder);
    #endif

    if (cudaEngine == nullptr) {
        std::cerr << "Failed to build CUDA engine" << std::endl;
        return false;
    }
    return true;
}
#endif

