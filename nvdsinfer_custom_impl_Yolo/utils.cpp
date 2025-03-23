#include "utils.h"

#include <iomanip>
#include <algorithm>
#include <experimental/filesystem>

// 去除字符串左侧的空格
static void leftTrim(std::string& s)
{
  s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](int ch) { return !isspace(ch); }));
}

// 去除字符串右侧的空格
static void rightTrim(std::string& s)
{
  s.erase(std::find_if(s.rbegin(), s.rend(), [](int ch) { return !isspace(ch); }).base(), s.end());
}

// 去除字符串两端的空格
std::string trim(std::string s)
{
  leftTrim(s);
  rightTrim(s);
  return s;
}

// 限制数值范围，确保 val 介于 minVal 和 maxVal 之间
float clamp(const float val, const float minVal, const float maxVal)
{
  assert(minVal <= maxVal);
  return std::min(maxVal, std::max(minVal, val));
}

// 检查文件是否存在
bool fileExists(const std::string fileName, bool verbose)
{
  if (!std::experimental::filesystem::exists(std::experimental::filesystem::path(fileName))) {
    if (verbose) {
      std::cout << "\nFile does not exist: " << fileName << std::endl;
    }
    return false;
  }
  return true;
}

// 加载权重文件
std::vector<float> loadWeights(const std::string weightsFilePath)
{
  assert(fileExists(weightsFilePath));
  std::cout << "\nLoading pre-trained weights" << std::endl;

  std::vector<float> weights;

  // 处理 .weights 文件
  if (weightsFilePath.find(".weights") != std::string::npos) {
    std::ifstream file(weightsFilePath, std::ios_base::binary);
    assert(file.good());
    
    // 处理不同 YOLO 版本的权重文件
    if (weightsFilePath.find("yolov2") != std::string::npos &&
        weightsFilePath.find("yolov2-tiny") == std::string::npos) {
      file.ignore(4 * 4); // 忽略前 4 个 int32 头部信息
    }
    else {
      file.ignore(4 * 5); // 忽略前 5 个 int32 头部信息
    }

    char floatWeight[4];
    while (!file.eof()) {
      file.read(floatWeight, 4);
      assert(file.gcount() == 4);
      weights.push_back(*reinterpret_cast<float*>(floatWeight));
      if (file.peek() == std::istream::traits_type::eof()) {
        break;
      }
    }
  }
  else {
    std::cerr << "\nFile " << weightsFilePath << " is not supported" << std::endl;
    assert(0);
  }

  std::cout << "Loading " << weightsFilePath << " complete" << std::endl;
  std::cout << "Total weights read: " << weights.size() << std::endl;

  return weights;
}

// 将 TensorRT 的维度转换为字符串格式
std::string dimsToString(const nvinfer1::Dims d)
{
  assert(d.nbDims >= 1);

  std::stringstream s;
  s << "[";
  for (int i = 1; i < d.nbDims - 1; ++i) {
    s << d.d[i] << ", ";
  }
  s << d.d[d.nbDims - 1] << "]";

  return s.str();
}

// 获取 TensorRT 张量的通道数
int getNumChannels(nvinfer1::ITensor* t)
{
  nvinfer1::Dims d = t->getDimensions();
  assert(d.nbDims == 4);
  return d.d[1];
}

// 打印层信息
void printLayerInfo(std::string layerIndex, std::string layerName, std::string layerInput, 
    std::string layerOutput, std::string weightPtr)
{
  std::cout << std::setw(7) << std::left << layerIndex << std::setw(40) << std::left << layerName;
  std::cout << std::setw(19) << std::left << layerInput << std::setw(19) << std::left << layerOutput;
  std::cout << weightPtr << std::endl;
}

