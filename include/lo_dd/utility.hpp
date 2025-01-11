/*
 * @Author: DMU zhangxianglong
 * @Date: 2024-11-18 11:41:09
 * @LastEditTime: 2025-01-10 10:33:45
 * @LastEditors: DMUZhangXianglong 347913076@qq.com
 * @FilePath: /LO-DD/include/lo_dd/utility.hpp
 * @Description: 
 */
#pragma once // 只包含一次当前头文件

#include <sensor_msgs/msg/point_cloud2.hpp>    //点云消息类型
#include <rmw/qos_profiles.h>                  // qos设置

// ROS2
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <std_msgs/msg/header.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <nav_msgs/msg/path.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_ros/transform_broadcaster.h>
#include <visualization_msgs/msg/marker.hpp>
#include <sensor_msgs/msg/imu.hpp>



// C++标准库
#include <mutex> 
#include <functional>
#include <string>
#include <chrono>
#include <omp.h>       // 并行处理
#include <fstream>     // 文件读写
#include <memory>      // std::unique_ptr

#include <queue>
#include <execution>

#include <cstdlib>
#include <boost/bind/bind.hpp>
#include <vector>
#include <cassert>


// PCL
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/kdtree/kdtree_flann.h> 
#include <pcl/filters/voxel_grid.h>  // 点云下采样


//eigen3
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <Eigen/Sparse>


// 李群李代数
#include "se2.hpp"
#include "se3.hpp"

// 自定义消息
#include "lo_dd/msg/pose6_d.hpp"




#define VEC_FROM_ARRAY(v)        v[0],v[1],v[2]
#define MAT_FROM_ARRAY(v)        v[0],v[1],v[2],v[3],v[4],v[5],v[6],v[7],v[8]



using Pose6D = lo_dd::msg::Pose6D;

using SE2 = Sophus::SE2d;
using SE2f = Sophus::SE2f;
using SO2 = Sophus::SO2d;
using SE3 = Sophus::SE3d;
using SE3f = Sophus::SE3f;
using SO3 = Sophus::SO3d;

using Vec3d = Eigen::Vector3d;
using Vec3f = Eigen::Vector3f;
using Mat3d = Eigen::Matrix3d;
using Mat3f = Eigen::Matrix3f;

Vec3d ZeroV3d(0, 0, 0);
Vec3f ZeroV3f(0, 0, 0);
Mat3d Identity3d(Mat3d::Identity());
Mat3f Identity3f(Mat3f::Identity());


// 定义常用点云类型
using PointType = pcl::PointXYZINormal;
using PointCloudType = pcl::PointCloud<PointType>; 
using PointVector = std::vector<PointType, Eigen::aligned_allocator<PointType>>;


using namespace std;
// 常量定义
constexpr double kDEG2RAD = M_PI / 180.0;  // deg->rad
constexpr double kRAD2DEG = 180.0 / M_PI;  // rad -> deg
constexpr double G_m_s2 = 9.81;  // rad -> deg

enum class SensorType {VELODYNE, ROBOSENSE, OUSTER};

// 当前需要处理的 IMU 和 LiDAR 数据
struct Measurements     
{
    Measurements()
    {
        lidar_begin_time = 0.0;
        this->lidar.reset(new PointCloudType());
    };
    double lidar_begin_time;
    double lidar_end_time;
    PointCloudType::Ptr lidar;
    std::deque<sensor_msgs::msg::Imu::ConstSharedPtr> imu;
};




class ParamServer : public rclcpp::Node
{
public:
    string pointCloudTopic;
    string imuTopic;
    string lidarFrame;
    string mapFrame;
    int num_scans;
    int min_edge_pts;
    int min_surf_pts;
    int max_iteration;
    bool use_edge_points;
    bool use_surf_points;
    double max_line_distance;
    double max_plane_distance;
    int min_effective_pts;
    double eps;
    double kf_distance;
    double kf_angle_deg;
    int num_kfs_in_local_map;
    std::string sensorStr;
    double min_filter_size_surf;
    double min_filter_size_map;

    SensorType sensor = SensorType::VELODYNE;
    
    ParamServer(std::string node_name, const rclcpp::NodeOptions & options) : Node(node_name, options)
    {
        declare_parameter("pointCloudTopic", "points");
        get_parameter("pointCloudTopic", pointCloudTopic);

        declare_parameter("imuTopic", "imuTopic");
        get_parameter("imuTopic", imuTopic);

        declare_parameter("lidarFrame", "lidarFrame");
        get_parameter("lidarFrame", lidarFrame);

        declare_parameter("num_scans", 16);
        get_parameter("num_scans", num_scans);

        declare_parameter("min_edge_pts", 20);
        get_parameter("min_edge_pts", min_edge_pts);

        declare_parameter("min_surf_pts", 20);
        get_parameter("min_surf_pts", min_surf_pts);

        declare_parameter("max_iteration", 5);
        get_parameter("max_iteration", max_iteration);

        declare_parameter("use_edge_points", true);
        get_parameter("use_edge_points", use_edge_points);

        declare_parameter("use_surf_points", true);
        get_parameter("use_surf_points", use_surf_points);

        declare_parameter("max_line_distance", 0.5);
        get_parameter("max_line_distance", max_line_distance);

        declare_parameter("max_plane_distance", 0.05);
        get_parameter("max_plane_distance", max_plane_distance);

        declare_parameter("min_effective_pts", 10);
        get_parameter("min_effective_pts", min_effective_pts);   

        declare_parameter("eps", 1e-3);
        get_parameter("eps", eps);  
        
        declare_parameter("kf_distance", 1.0);
        get_parameter("kf_distance", kf_distance);  

        declare_parameter("kf_angle_deg", 15.0);
        get_parameter("kf_angle_deg", kf_angle_deg);  

        declare_parameter("num_kfs_in_local_map", 30);
        get_parameter("num_kfs_in_local_map", num_kfs_in_local_map);  

        declare_parameter("mapFrame", "map");
        get_parameter("mapFrame", mapFrame);  

        declare_parameter("sensor", "velodyne");
        get_parameter("sensor", sensorStr);

        declare_parameter("min_filter_size_surf", 0.5);
        get_parameter("min_filter_size_surf", min_filter_size_surf);  


        declare_parameter("min_filter_size_map", 0.5);
        get_parameter("min_filter_size_map", min_filter_size_map); 

        if (sensorStr == "velodyne")
        {
            sensor = SensorType::VELODYNE;
        }
        else if(sensorStr == "robosense")
        {
            sensor = SensorType::ROBOSENSE;
        }
        else if(sensorStr == "ouster")
        {
            sensor = SensorType::OUSTER;    
        }
        else
        {
            RCLCPP_ERROR_STREAM(this->get_logger(), "Invalid sensor type (must be either 'velodyne' or 'ouster' or 'livox'): " << sensorStr);
            rclcpp::shutdown();
        }



    }
    

    /**
     * @brief 计算耗时
     * 
     * @param start 开始时间
     * @param end   结束时间
     */
    void calculateExecutionTime(std::string text, std::chrono::high_resolution_clock::time_point start, std::chrono::high_resolution_clock::time_point end)
    {
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        RCLCPP_INFO_STREAM(this->get_logger(), text << duration.count() << " milliseconds.");
        
        // 写到 txt 文件中, 统计特征提取的耗时
        // std::ofstream file("./common.txt", std::ios::app); // 追加模式
        // if (file.is_open()) {
        //     // 写入持续时间并换行
        //     file << duration.count() << std::endl;

        //     // 关闭文件
        //     file.close();
        //     // std::cout << "Duration saved to durations.txt" << std::endl;
        // } else {
        //     std::cerr << "Failed to open the file." << std::endl;
        // }
    }
    
};



void publishPath(nav_msgs::msg::Path path, rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr thisPub, rclcpp::Time thisStamp, std::string thisFrame)
{
    // auto pose_msg = geometry_msgs::msg::PoseStamped();
    path.header.stamp = thisStamp;
    path.header.frame_id = thisFrame; 

    // pose_msg.pose.position.x = pose.translation()(0);
    // pose_msg.pose.position.y = pose.translation()(1);
    // pose_msg.pose.position.z = pose.translation()(2);

    // Eigen::Quaterniond q = pose.so3().unit_quaternion();  // 获取单位四元数
    // pose_msg.pose.orientation.x = q.x();
    // pose_msg.pose.orientation.y = q.y();
    // pose_msg.pose.orientation.z = q.z();
    // pose_msg.pose.orientation.w = q.w();



    thisPub->publish(path);
}

/**
 * @brief 发布点云
 * 
 * @param thisPub   发布器
 * @param thisCloud 被发布的点云PCL格式
 * @param thisStamp 时间戳
 * @param thisFrame frame_id
 */
void publishPointCloud(rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr thisPub, PointCloudType::Ptr thisCloud, rclcpp::Time thisStamp, std::string thisFrame)
{
    sensor_msgs::msg::PointCloud2 tempCloud;
    pcl::toROSMsg(*thisCloud, tempCloud);
    tempCloud.header.stamp = thisStamp;
    tempCloud.header.frame_id = thisFrame;
    if (thisPub->get_subscription_count() != 0)
    {
        thisPub->publish(tempCloud);
    }
    
}

/**
 * @brief Get the ros time object
 * 
 * @param timestamp 
 * @return * rclcpp::Time 
 */
rclcpp::Time getRosTime(double timestamp)
{
    int32_t sec = std::floor(timestamp);
    auto nanosec_d = (timestamp - std::floor(timestamp)) * 1e9;
    uint32_t nanosec = nanosec_d;
    return rclcpp::Time(sec, nanosec);
}

/**
 * @brief Get the time sec object
 * 
 * @param time 
 * @return double 
 */
double getTimeSec(const builtin_interfaces::msg::Time &time)
{
    return rclcpp::Time(time).seconds();
}

/**
 * @brief 计算2个数的平均值
 * 
 * @param x1 
 * @param x2 
 * @return double 
 */
double computeAverage(double x1, double x2)
{   
    double result;
    result = 0.5 * (x1 + x2);
    return result;
}


template<typename T>
/**
 * @brief 
 * 
 */
auto setPose6D(const double t, const Eigen::Matrix<T, 3, 1> &a, const Eigen::Matrix<T, 3, 1> &g, \
                const Eigen::Matrix<T, 3, 1> &v, const Eigen::Matrix<T, 3, 1> &p, const Eigen::Matrix<T, 3, 3> &R)
{
    Pose6D rot_kp;
    rot_kp.offset_time = t;
    for (int i = 0; i < 3; i++)
    {
        rot_kp.acceleration[i] = a(i);
        rot_kp.gyroscope[i] = g(i);
        rot_kp.velocity[i] = v(i);
        rot_kp.position[i] = p(i);
        for (int j = 0; j < 3; j++)  rot_kp.rotation_matrix[i*3+j] = R(i,j);
    }
    return move(rot_kp);
}

/**
 * 计算一个容器内数据的均值与对角形式协方差
 * @tparam C    容器类型
 * @tparam D    结果类型
 * @tparam Getter   获取数据函数, 接收一个容器内数据类型，返回一个D类型
 */
template <typename C, typename D, typename Getter>
void computeMeanAndCovDiag(const C& data, D& mean, D& cov_diag, Getter&& getter) {
    size_t len = data.size();
    assert(len > 1);
    // clang-format off
    mean = std::accumulate(data.begin(), data.end(), D::Zero().eval(),
                           [&getter](const D& sum, const auto& data) -> D { return sum + getter(data); }) / len;
    cov_diag = std::accumulate(data.begin(), data.end(), D::Zero().eval(),
                               [&mean, &getter](const D& sum, const auto& data) -> D {
                                   return sum + (getter(data) - mean).cwiseAbs2().eval();
                               }) / (len - 1);
    // clang-format on
}



/**
 * @brief 转换为Eigen::Vector3f的点
 * 
 * @param pt 输入的pcl形式的点
 * @return Eigen::Vector3f 
 */
inline Eigen::Vector3f ToVec3f(const PointType &pt) { return pt.getVector3fMap(); }


/**
 * @brief 转换为Eigen::Vector3d的点
 * 
 * @param pt 输入的pcl形式的点
 * @return Eigen::Vector3d 
 */
inline Eigen::Vector3d ToVec3d(const PointType &pt) { return pt.getVector3fMap().cast<double>(); }

/**
 * @brief 
 * 
 */
template <typename S>
inline PointType ToPointXYZI(const Eigen::Matrix<S, 3, 1>& pt) {
    PointType p;
    p.x = pt.x();
    p.y = pt.y();
    p.z = pt.z();
    return p;
}


/**
 * @brief 获取当前时间
 * 
 * @return 当前时间 
 */
std::chrono::high_resolution_clock::time_point getCurrentTime()
{
    return std::chrono::high_resolution_clock::now();
}


/**
 * @brief 计算2点之间距离平方
 * 
 * @param p1 
 * @param p2 
 * @return double 
 */
inline double calcDistanceSquared(const PointType &p1, const PointType &p2) {
    double diffX = p1.x - p2.x;
    double diffY = p1.y - p2.y;
    double diffZ = p1.z - p2.z;
    return diffX * diffX + diffY * diffY + diffZ * diffZ;
}


/**
 * @brief 计算2点之间距离平方Eigen形式
 * 
 * @param p1 
 * @param p2 
 * @return float 
 */
inline float Dis2(const Eigen::Vector3f &p1, const Eigen::Vector3f &p2) { return (p1 - p2).squaredNorm(); }

/**
 * @brief 拟合直线
 * 
 * @tparam S 
 * @param data 
 * @param origin 
 * @param dir 
 * @param eps 
 * @return true 
 * @return false 
 */
template <typename S>
bool fitLine(std::vector<Eigen::Matrix<S, 3, 1>>& data, Eigen::Matrix<S, 3, 1>& origin, Eigen::Matrix<S, 3, 1>& dir,
             double eps = 0.2) {
    if (data.size() < 2) {
        return false;
    }

    origin = std::accumulate(data.begin(), data.end(), Eigen::Matrix<S, 3, 1>::Zero().eval()) / data.size();

    Eigen::MatrixXd Y(data.size(), 3);
    for (int i = 0; i < data.size(); ++i) {
        Y.row(i) = (data[i] - origin).transpose();
    }

    Eigen::JacobiSVD svd(Y, Eigen::ComputeFullV);
    dir = svd.matrixV().col(0);

    // check eps
    for (const auto& d : data) {
        if (dir.template cross(d - origin).template squaredNorm() > eps) {
            return false;
        }
    }

    return true;
}

/**
 * @brief 拟合平面
 * 
 * @tparam S 
 * @param data 
 * @param plane_coeffs 
 * @param eps 
 * @return true 
 * @return false 
 */
template <typename S>
bool fitPlane(std::vector<Eigen::Matrix<S, 3, 1>>& data, Eigen::Matrix<S, 4, 1>& plane_coeffs, double eps = 1e-2) {
    if (data.size() < 3) {
        return false;
    }

    Eigen::MatrixXd A(data.size(), 4);
    for (int i = 0; i < data.size(); ++i) {
        A.row(i).head<3>() = data[i].transpose();
        A.row(i)[3] = 1.0;
    }

    Eigen::JacobiSVD svd(A, Eigen::ComputeThinV);
    plane_coeffs = svd.matrixV().col(3);

    // check error eps
    for (int i = 0; i < data.size(); ++i) {
        double err = plane_coeffs.template head<3>().dot(data[i]) + plane_coeffs[3];
        if (err * err > eps) {
            return false;
        }
    }

    return true;
}

/**
 * @brief 体素处理
 * 
 * @param cloud 
 * @param voxel_size 
 * @return PointCloudType::Ptr 
 */
inline PointCloudType::Ptr voxelCloud(PointCloudType::Ptr cloud, float voxel_size = 0.1) {
    pcl::VoxelGrid<PointType> voxel;
    voxel.setLeafSize(voxel_size, voxel_size, voxel_size);
    voxel.setInputCloud(cloud);

    PointCloudType::Ptr output(new PointCloudType());
    voxel.filter(*output);
    return output;
}




rmw_qos_profile_t qos_profile{
    RMW_QOS_POLICY_HISTORY_KEEP_LAST,
    1,
    // RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT,
    RMW_QOS_POLICY_RELIABILITY_RELIABLE,  // 13代处理器
    RMW_QOS_POLICY_DURABILITY_VOLATILE,
    RMW_QOS_DEADLINE_DEFAULT,
    RMW_QOS_LIFESPAN_DEFAULT,
    RMW_QOS_POLICY_LIVELINESS_SYSTEM_DEFAULT,
    RMW_QOS_LIVELINESS_LEASE_DURATION_DEFAULT,
    false
};

auto qos = rclcpp::QoS(
    rclcpp::QoSInitialization(
        qos_profile.history,
        qos_profile.depth
    ),
    qos_profile);


rmw_qos_profile_t qos_profile_lidar{
    RMW_QOS_POLICY_HISTORY_KEEP_LAST,
    5,
    // RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT,
    RMW_QOS_POLICY_RELIABILITY_RELIABLE,  // 13代处理器
    RMW_QOS_POLICY_DURABILITY_VOLATILE,
    RMW_QOS_DEADLINE_DEFAULT,
    RMW_QOS_LIFESPAN_DEFAULT,
    RMW_QOS_POLICY_LIVELINESS_SYSTEM_DEFAULT,
    RMW_QOS_LIVELINESS_LEASE_DURATION_DEFAULT,
    false
};

auto qos_lidar = rclcpp::QoS(
    rclcpp::QoSInitialization(
        qos_profile_lidar.history,
        qos_profile_lidar.depth
    ),
    qos_profile_lidar);

rmw_qos_profile_t qos_profile_imu{
    RMW_QOS_POLICY_HISTORY_KEEP_LAST,
    2000,
    RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT,
    RMW_QOS_POLICY_DURABILITY_VOLATILE,
    RMW_QOS_DEADLINE_DEFAULT,
    RMW_QOS_LIFESPAN_DEFAULT,
    RMW_QOS_POLICY_LIVELINESS_SYSTEM_DEFAULT,
    RMW_QOS_LIVELINESS_LEASE_DURATION_DEFAULT,
    false
};

auto qos_imu = rclcpp::QoS(
    rclcpp::QoSInitialization(
        qos_profile_imu.history,
        qos_profile_imu.depth
    ),
    qos_profile_imu);

