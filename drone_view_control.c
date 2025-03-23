#include <ros/ros.h>
#include <opencv2/opencv.hpp>
#include <geometry_msgs/Twist.h>
#include <std_msgs/Float32.h>
#include <iostream>
#include <cmath>
#include <algorithm>

// 相机参数结构体
struct CameraParams {
    float fx = 800;
    float fy = 800;
    float cx = 320;
    float cy = 240;
};

// PID控制器结构体
class PIDController {
public:
    float Kp, Ki, Kd;
    float integral, previous_error;

    PIDController(float Kp, float Ki, float Kd)
        : Kp(Kp), Ki(Ki), Kd(Kd), integral(0), previous_error(0) {}

    float compute(float target, float current) {
        float error = target - current;
        integral += error;
        float derivative = error - previous_error;
        previous_error = error;

        // PID控制公式
        return Kp * error + Ki * integral + Kd * derivative;
    }
};

// 动态调整PID增益
void adjustPIDGains(PIDController &pid, float error) {
    float small_error_threshold = 0.1;  // 小误差阈值
    float large_error_threshold = 1.0;  // 大误差阈值

    if (std::fabs(error) < small_error_threshold) {
        pid.Kp = 0.05f;
        pid.Ki = 0.005f;
        pid.Kd = 0.01f;
    } else if (std::fabs(error) > large_error_threshold) {
        pid.Kp = 0.2f;
        pid.Ki = 0.02f;
        pid.Kd = 0.05f;
    } else {
        pid.Kp = 0.1f;
        pid.Ki = 0.01f;
        pid.Kd = 0.05f;
    }
}

// 限速函数
float limit_speed(float speed, float max_speed) {
    return std::clamp(speed, -max_speed, max_speed);
}

int main(int argc, char **argv) {
    ros::init(argc, argv, "drone_view_control");  // 初始化ROS节点
    ros::NodeHandle nh;

    CameraParams cam_params;
    PIDController pidX(0.1f, 0.01f, 0.05f);  // X轴PID控制器
    PIDController pidY(0.1f, 0.01f, 0.05f);  // Y轴PID控制器
    PIDController pidZ(0.1f, 0.01f, 0.05f);  // Z轴PID控制器

    // 定义相机内参和畸变系数
    cv::Mat cameraMatrix = (cv::Mat_<double>(3,3) << 800, 0, 320, 
                                                     0, 800, 240, 
                                                     0, 0, 1);
    cv::Mat distCoeffs = (cv::Mat_<double>(1,5) << -0.2, 
                                                    0.1, 
                                                    0, 
                                                    0, 
                                                    -0.05);

    ros::Rate loop_rate(10);  // 设定循环频率 10Hz
    float max_speed = 0.5;    // 最大速度

    while (ros::ok()) {
        int u = 0, v = 0;
        float depth = 0.0;

        // 获取目标像素坐标 (u, v) 和目标深度信息
        if (nh.hasParam("/target_pixel_x") && nh.hasParam("/target_pixel_y") && nh.hasParam("/target_depth")) {
            nh.getParam("/target_pixel_x", u);
            nh.getParam("/target_pixel_y", v);
            nh.getParam("/target_depth", depth);

            if (depth <= 0) {
                ROS_WARN("Invalid depth value: %.2f", depth);
                ros::spinOnce();
                loop_rate.sleep();
                continue;
            }

            // 计算目标在相机坐标系中的坐标
            float X_cam = (u - cam_params.cx) * depth / cam_params.fx;
            float Y_cam = (v - cam_params.cy) * depth / cam_params.fy;
            float Z_cam = depth;

            // 变换至无人机坐标系
            float X_d = Y_cam;
            float Y_d = -X_cam;
            float Z_d = Z_cam;

            // 设定目标位置
            float target_x = 0.0, target_y = 0.0, target_z = 2.0;

            // 计算PID控制输出前，动态调整PID增益
            float errorX = target_x - X_d;
            float errorY = target_y - Y_d;
            float errorZ = target_z - Z_d;

            adjustPIDGains(pidX, errorX);  // 调整X轴PID增益
            adjustPIDGains(pidY, errorY);  // 调整Y轴PID增益
            adjustPIDGains(pidZ, errorZ);  // 调整Z轴PID增益

            float vx_pid = pidX.compute(target_x, X_d);
            float vy_pid = pidY.compute(target_y, Y_d);
            float vz_pid = pidZ.compute(target_z, Z_d);

            // 限制速度范围
            float vx = limit_speed(vx_pid, max_speed);
            float vy = limit_speed(vy_pid, max_speed);
            float vz = limit_speed(vz_pid, max_speed);

            // 通过ROS参数服务器发布速度调整值
            ros::param::set("adjust_vel_x", vx);
            ros::param::set("adjust_vel_y", vy);
            ros::param::set("adjust_vel_z", vz);

            // 输出控制速度信息
            ROS_INFO("Control Speed: Vx=%.2f m/s, Vy=%.2f m/s, Vz=%.2f m/s", vx, vy, vz);
        }

        ros::spinOnce();
        loop_rate.sleep();
    }

    return 0;
}
