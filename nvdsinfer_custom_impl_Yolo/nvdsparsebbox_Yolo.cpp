#include <gst/gst.h>
#include <glib.h>
#include "nvdsmeta.h"
#include "nvdsinfer.h"
#include "nvdsinfer_custom_impl.h"
#include "nvds_obj_encode.h"
#include <algorithm>
#include "utils.h"
#include <ros/ros.h>

// 声明一个外部 C 风格的函数，用于解析 YOLO 推理的输出，填充检测到的目标列表
extern "C" bool NvDsInferParseYolo(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList);


// 限制数值范围
static float clamp(float val, float minVal, float maxVal) {
  return std::max(minVal, std::min(val, maxVal));
}

// 将YOLO网络输出的边界框坐标转换为符合目标检测格式的边界框信息
static NvDsInferParseObjectInfo
convertBBox(const float& bx1, const float& by1, const float& bx2, const float& by2, const uint& netW, const uint& netH)
{
  NvDsInferParseObjectInfo b; // 存储转换后的边界框信息

  float x1 = bx1;
  float y1 = by1;
  float x2 = bx2;
  float y2 = by2;

  // 限制边界框坐标在图像范围内
  x1 = clamp(x1, 0, netW);
  y1 = clamp(y1, 0, netH);
  x2 = clamp(x2, 0, netW);
  y2 = clamp(y2, 0, netH);

  // 计算边界框的位置和大小
  b.left = x1;
  b.width = clamp(x2 - x1, 0, netW);
  b.top = y1;
  b.height = clamp(y2 - y1, 0, netH);

  return b;
}

// 添加一个边界框检测结果到检测对象列表中
static void
addBBoxProposal(const float bx1, const float by1, 
                const float bx2, const float by2, 
                const uint& netW, const uint& netH,
                const int maxIndex, const float maxProb, 
                std::vector<NvDsInferParseObjectInfo>& binfo)
{
  // 先转换边界框
  NvDsInferParseObjectInfo bbi = convertBBox(bx1, by1, 
                                             bx2, by2, 
                                             netW, netH);

  // 过滤掉无效的边界框
  if (bbi.width < 1 || bbi.height < 1) {
    return;
  }

  // 赋值检测置信度和类别ID
  bbi.detectionConfidence = maxProb;
  bbi.classId = maxIndex;

  // 添加到检测对象列表
  binfo.push_back(bbi);
}
// 解析 YOLO 输出张量，提取检测到的目标
static std::vector<NvDsInferParseObjectInfo>
decodeTensorYolo(const float* output, const uint& outputSize, 
                 const uint& netW, const uint& netH,
                 const std::vector<float>& preclusterThreshold)
{
  std::vector<NvDsInferParseObjectInfo> binfo; // 存储解析后的边界框信息

  for (uint b = 0; b < outputSize; ++b) {
      float maxProb = output[b * 6 + 4]; // 获取该检测框的最大置信度
      int maxIndex = (int) output[b * 6 + 5]; // 获取该检测框对应的类别索引

      // 如果检测置信度低于设定的阈值，则忽略该检测框
      if (maxProb < preclusterThreshold[maxIndex]) {
          continue;
      }

      // 提取边界框的坐标信息
      float bx1 = output[b * 6 + 0];
      float by1 = output[b * 6 + 1];
      float bx2 = output[b * 6 + 2];
      float by2 = output[b * 6 + 3];

      // 添加边界框到检测对象列表
      addBBoxProposal(bx1, by1, bx2, by2, netW, netH, maxIndex, maxProb, binfo);
  }

    return binfo;
}

// 解析 YOLO 推理输出，并填充检测对象列表
static bool NvDsInferParseCustomYolo(std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
				      NvDsInferNetworkInfo const& networkInfo,
				      NvDsInferParseDetectionParams const& detectionParams,
				      std::vector<NvDsInferParseObjectInfo>& objectList) 
{
  // 检查输出层是否为空
  if (outputLayersInfo.empty()) {
      std::cerr << "ERROR: Could not find output layer in bbox parsing" << std::endl;
      return false;
    }

  std::vector<NvDsInferParseObjectInfo> objects;

  // 只处理第一个输出层
  const NvDsInferLayerInfo& output = outputLayersInfo[0];
  const uint outputSize = output.inferDims.d[0]; // 获取输出层的大小

  // 解析YOLO输出张量
  std::vector<NvDsInferParseObjectInfo> outObjs = decodeTensorYolo(
      (const float*) (output.buffer), outputSize,
      networkInfo.width, networkInfo.height, 
      detectionParams.perClassPreclusterThreshold);

  // 合并解析出的目标
  objects.insert(objects.end(), outObjs.begin(), outObjs.end());

  // 赋值最终的检测对象列表
  objectList = objects;

  return true;
}

// C风格的外部接口，调用解析函数
extern "C" bool NvDsInferParseYolo(std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
                                   NvDsInferNetworkInfo const& networkInfo,
                                   NvDsInferParseDetectionParams const& detectionParams,
                                   std::vector<NvDsInferParseObjectInfo>& objectList) {
    return NvDsInferParseCustomYolo(outputLayersInfo, networkInfo, detectionParams, objectList);
}

// 检查解析函数的声明是否符合要求
CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseYolo);

// OSD 显示检测框及类别标签
static GstPadProbeReturn osd_sink_pad_buffer_probe(GstPad *pad, GstPadProbeInfo *info, gpointer user_data) {
    GstBuffer *buf = (GstBuffer *)info->data;
    NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta(buf);
    NvDsObjectMeta *obj_meta = NULL;

    for (NvDsMetaList *l_frame = batch_meta->frame_meta_list; l_frame != NULL; l_frame = l_frame->next) {
        NvDsFrameMeta *frame_meta = (NvDsFrameMeta *)(l_frame->data);

        for (NvDsMetaList *l_obj = frame_meta->obj_meta_list; l_obj != NULL; l_obj = l_obj->next) {
            obj_meta = (NvDsObjectMeta *)(l_obj->data);

            // 设置边界框颜色 (红色)
            obj_meta->rect_params.border_color.red = 1.0;
            obj_meta->rect_params.border_color.green = 0.0;
            obj_meta->rect_params.border_color.blue = 0.0;
            obj_meta->rect_params.border_color.alpha = 1.0;

            // 增加 BBox 线宽
            obj_meta->rect_params.border_width = 3;

            // 添加类别标签 (例如："person" 或 "car")
            gchar label[128];
            snprintf(label, sizeof(label), "%s %.2f", obj_meta->obj_label, obj_meta->confidence);

            // 在检测框旁边绘制类别标签
            NvOSD_DrawText(frame_meta->frame, label, obj_meta->rect_params.left, obj_meta->rect_params.top,
                           obj_meta->text_params.font_color, obj_meta->text_params.font_size);

            // 计算边界框的中心点坐标
            float center_x = (obj_meta->rect_params.left + obj_meta->rect_params.left + obj_meta->rect_params.width) / 2.0;
            float center_y = (obj_meta->rect_params.top + obj_meta->rect_params.top + obj_meta->rect_params.height) / 2.0;
            
            // 发送中心点坐标
            ros::param::set("target_pixel_x", center_x);
            ros::param::set("target_pixel_y", center_y);
            ros::param::set("target_label", std::string(label));
            
            ROS_INFO("Object center: x = %f, y = %f, label = %s", center_x, center_y, label);
        }
    }

    return GST_PAD_PROBE_OK;
}

// 在 DeepStream Pipeline 中，绑定 OSD 处理的函数
void setup_osd(GstElement* pipeline) {
    GstElement *osd = gst_bin_get_by_name(GST_BIN(pipeline), "nvosd");
    GstPad *osd_sink_pad = gst_element_get_static_pad(osd, "sink");

    gst_pad_add_probe(osd_sink_pad, GST_PAD_PROBE_TYPE_BUFFER, osd_sink_pad_buffer_probe, NULL, NULL);
    gst_object_unref(osd_sink_pad);
}