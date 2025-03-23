#include "calibrator.h"

#include <fstream>
#include <iterator>

// INT8 校准器构造函数
Int8EntropyCalibrator2::Int8EntropyCalibrator2(const int& batchSize, const int& channels, const int& height,
    const int& width, const float& scaleFactor, const float* offsets, const int& inputFormat,
    const std::string& imgPath, const std::string& calibTablePath) : batchSize(batchSize), inputC(channels),
    inputH(height), inputW(width), scaleFactor(scaleFactor), offsets(offsets), inputFormat(inputFormat),
    calibTablePath(calibTablePath), imageIndex(0)
{
  // 计算输入数据总大小
  inputCount = batchSize * channels * height * width;
  // 读取图片路径
  std::fstream f(imgPath);
  if (f.is_open()) {
      std::string temp;
      while (std::getline(f, temp)) {
        imgPaths.push_back(temp);
      }
  }
  // 分配 batch 数据内存
  batchData = new float[inputCount];
  // 在 GPU 上分配输入数据缓冲区
  CUDA_CHECK(cudaMalloc(&deviceInput, inputCount * sizeof(float)));
}

// 析构函数，释放 CUDA 资源和分配的内存
Int8EntropyCalibrator2::~Int8EntropyCalibrator2()
{
  CUDA_CHECK(cudaFree(deviceInput));
  if (batchData) {
    delete[] batchData;
  }
}

// 获取 batch 大小
int
Int8EntropyCalibrator2::getBatchSize() const noexcept
{
  return batchSize;
}

// 读取 batch 数据并拷贝到 GPU
bool
Int8EntropyCalibrator2::getBatch(void** bindings, const char** names, int nbBindings) noexcept
{
  if (imageIndex + batchSize > uint(imgPaths.size())) {
    return false;
  }

  float* ptr = batchData;
  for (size_t i = imageIndex; i < imageIndex + batchSize; ++i) {
    // 读取图片
    cv::Mat img = cv::imread(imgPaths[i]);
    if (img.empty()){
      std::cerr << "Failed to read image for calibration" << std::endl;
      return false;
    }
  
    // 预处理图片
    std::vector<float> inputData = prepareImage(img, inputC, inputH, inputW, scaleFactor, offsets, inputFormat);

    size_t len = inputData.size();
    memcpy(ptr, inputData.data(), len * sizeof(float));
    ptr += inputData.size();

    std::cout << "Load image: " << imgPaths[i] << std::endl;
    std::cout << "Progress: " << (i + 1) * 100. / imgPaths.size() << "%" << std::endl;
  }

  imageIndex += batchSize;

  // 将数据拷贝到 GPU
  CUDA_CHECK(cudaMemcpy(deviceInput, batchData, inputCount * sizeof(float), cudaMemcpyHostToDevice));
  bindings[0] = deviceInput;

  return true;
}

// 读取校准缓存
const void*
Int8EntropyCalibrator2::readCalibrationCache(std::size_t &length) noexcept
{
  calibrationCache.clear();
  std::ifstream input(calibTablePath, std::ios::binary);
  input >> std::noskipws;
  if (readCache && input.good()) {
    std::copy(std::istream_iterator<char>(input), std::istream_iterator<char>(), std::back_inserter(calibrationCache));
  }
  length = calibrationCache.size();
  return length ? calibrationCache.data() : nullptr;
}

// 写入校准缓存
void
Int8EntropyCalibrator2::writeCalibrationCache(const void* cache, std::size_t length) noexcept
{
  std::ofstream output(calibTablePath, std::ios::binary);
  output.write(reinterpret_cast<const char*>(cache), length);
}

// 预处理图片数据，将其转换为 float 数组
std::vector<float>
prepareImage(cv::Mat& img, int inputC, int inputH, int inputW, float scaleFactor, const float* offsets, int inputFormat)
{
  cv::Mat out;

  // 根据输入格式转换颜色空间
  if (inputFormat == 0) {
    cv::cvtColor(img, out, cv::COLOR_BGR2RGB);
  }
  else if (inputFormat == 2) {
    cv::cvtColor(img, out, cv::COLOR_BGR2GRAY);
  }
  else {
    out = img;
  }

  int imageW = img.cols;
  int imageH = img.rows;

  // 调整图片尺寸，确保匹配输入大小
  if (imageW != inputW || imageH != inputH) {
    float resizeFactor = std::max(inputW / (float) imageW, inputH / (float) imageH);
    cv::resize(out, out, cv::Size(0, 0), resizeFactor, resizeFactor, cv::INTER_CUBIC);
    cv::Rect crop(cv::Point(0.5 * (out.cols - inputW), 0.5 * (out.rows - inputH)), cv::Size(inputW, inputH));
    out = out(crop);
  }

  // 转换为 float 类型，并应用缩放因子
  out.convertTo(out, CV_32F, scaleFactor);

  // 减去均值
  if (inputFormat == 2) {
    cv::subtract(out, cv::Scalar(offsets[0] / 255), out);
  }
  else {
    cv::subtract(out, cv::Scalar(offsets[0] / 255, offsets[1] / 255, offsets[3] / 255), out);
  }

  // 将图片数据拆分到多个通道，并存入 float 数组
  std::vector<cv::Mat> inputChannels(inputC);
  cv::split(out, inputChannels);
  std::vector<float> result(inputH * inputW * inputC);
  auto data = result.data();
  int channelLength = inputH * inputW;
  for (int i = 0; i < inputC; ++i) {
    memcpy(data, inputChannels[i].data, channelLength * sizeof(float));
    data += channelLength;
  }

  return result;
}

