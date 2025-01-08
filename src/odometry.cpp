/*
 * @Author: DMU zhangxianglong
 * @Date: 2024-11-18 11:34:52
 * @LastEditTime: 2025-01-08 23:50:08
 * @LastEditors: DMUZhangXianglong 347913076@qq.com
 * @FilePath: /LO-DD/src/lidarOdometry.cpp
 * @Description: 实现激光里程计
 * 订阅 预处理后的LiDAR点云
 * 订阅 IMU原始数据     
 */

#include "utility.hpp"
#include "kdtree.hpp"
#include "imuProcess.hpp"
#include "eskfom.hpp"
#include "IKFoM.hpp"
#include "ikd_Tree.h"

#define INIT_TIME (0.1)

class Odometry : public ParamServer
{
private:
    


public:
    // 节点配置
    // rclcpp::NodeOptions feature_extraction_options;
    // feature_extraction_options.use_intra_process_comms(true);

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subEdgeCloud;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subSurfaceCloud;
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr subImu;
    rclcpp::TimerBase::SharedPtr time;

    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubGlobalMap;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubcurrentScan;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pubPose;
    nav_msgs::msg::Path globalPath;

    // 用于发布TF
    std::unique_ptr<tf2_ros::TransformBroadcaster> br;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr pubMarker;
    
    PointCloudType::Ptr currentCloud;
    PointCloudType::Ptr currentEdgeCloud;
    PointCloudType::Ptr currentSurfaceCloud;
    PointCloudType::Ptr localMapEdge;
    PointCloudType::Ptr localMapSurface;
    PointCloudType::Ptr surfaceWorld;
    PointCloudType::Ptr feature_undistort;  // 去畸变后的点云
    PointCloudType::Ptr feature_dsf_body;   // 去畸变后降采样后的点云，lidar坐标系下 dsf 表示down size filter
    PointCloudType::Ptr feature_dsf_world;  // 去畸变后降采样后的点云，世界坐标系下
    

    // IMU 和 LiDAR
    std::deque<sensor_msgs::msg::Imu::ConstSharedPtr> imu_buffer;
    std::deque<PointCloudType::Ptr> lidar_buffer;
    std::deque<double> time_buffer;
    bool isFirstLiDAR, lidar_pushed;
    double last_timestamp_imu, last_timestamp_lidar, lidar_end_time;
    double lidar_mean_scantime;
    int scan_num;
    Measurements measurement;
    bool FIRST_SCAN;
    double first_lidar_time;  // 当前测量第一个lidar点时间
    // imuprocess 对象
    std::shared_ptr<ImuProcess> imu_process;
    esekfom::eskf kf;       // 误差状态卡尔曼滤波器
    state_ikfom state_now;  // 状态
    Vec3d LiDAR_position_w; // LiDAR在世界坐标系下的位置
    bool ekf_is_inited; 
    pcl::VoxelGrid<PointType> downSizeFilterSurf; // 降采样器
    int feature_dsf_size; // 降采样后点云大小


    
    // ikd-tree
    vector<BoxPointType> cube_need_remove;
    vector<PointVector> nearest_points;
    int kdtree_delete_counter, add_point_size;
    // KD_TREE(PointType) ikdtree;
    KD_TREE<PointType> ikdtree;
    
    // 全局地图
    PointCloudType::Ptr globalMap;

    // KdTree
    KdTree kdtreeEdge;
    KdTree kdtreeSurf;
    // pcl::KdTreeFLANN<PointType>::Ptr kdtreeEdge;
    // pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurf;


    // 存放角点和面点的队列
    std::deque<PointCloudType::Ptr> edge_deque;
    std::deque<PointCloudType::Ptr> surf_deque;


    // 存放估计得到的位姿
    std::vector<SE3> estimated_poses;

    // 关键帧序号
    int frameID;
    int lastKeyFrameID;

    // 上一次关键帧位姿
    SE3 lastKeyFramePose;

    // 测试
    std_msgs::msg::Header header;



    // flag
    bool isInit;

    // 退化检测
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver;  // 求解特征向量
    std::vector<Eigen::Vector3d> Ft;
    std::vector<Eigen::Vector3d> temp;
    Eigen::Matrix<double, 3, 3> Arr;
    Eigen::Matrix<double, 3, 3> Att;
    std::mutex ft_mutex;
    Eigen::Vector3d minV;
    Eigen::Vector3d maxV;


    // 线程锁
    std::mutex mtx_buffer;
    std::condition_variable sig_buffer;

    
    Odometry(const rclcpp::NodeOptions & options) : ParamServer("lo_dd_odometry", options)
    {   
        // 订阅角点和面点
        subEdgeCloud = create_subscription<sensor_msgs::msg::PointCloud2>("/lo_dd/edge_points", qos_lidar, std::bind(&Odometry::edgeCloudHandler, this, std::placeholders::_1));
        subSurfaceCloud = create_subscription<sensor_msgs::msg::PointCloud2>("/lo_dd/surf_points", qos_lidar, std::bind(&Odometry::surfaceCloudHandler, this, std::placeholders::_1));
        
        // subEdgeCloud = create_subscription<sensor_msgs::msg::PointCloud2>(pointCloudTopic, qos, std::bind(&Odometry::edgeCloudHandler, this, std::placeholders::_1));
        // subSurfaceCloud = create_subscription<sensor_msgs::msg::PointCloud2>(pointCloudTopic, qos, std::bind(&Odometry::surfaceCloudHandler, this, std::placeholders::_1));
        
        // 订阅imu
        subImu = create_subscription<sensor_msgs::msg::Imu>(imuTopic, qos_imu, std::bind(&Odometry::imuHandler, this, std::placeholders::_1));
        
        // 里程计
        auto period_ms = std::chrono::milliseconds(static_cast<int64_t>(1000.0 / 100.0));
        time = rclcpp::create_timer(this, this->get_clock(), period_ms, std::bind(&Odometry::odometryHandler, this));
        
        pubGlobalMap = create_publisher<sensor_msgs::msg::PointCloud2>("/lo_dd/golobalMap", 1);
        pubcurrentScan = create_publisher<sensor_msgs::msg::PointCloud2>("/lo_dd/currentScan", 1);
        pubPose = create_publisher<nav_msgs::msg::Path>("/lo_dd/pose", 1);
        pubMarker = create_publisher<visualization_msgs::msg::Marker>("/degeneracy_direction", 1);

        // 发布TF
        br = std::make_unique<tf2_ros::TransformBroadcaster>(this);
        
        allocateMemory();
    }

    
    ~Odometry(){}

    /**
     * @brief 分配内存
     * 
     */
    void allocateMemory()
    {
        currentCloud.reset(new PointCloudType());
        currentEdgeCloud.reset(new PointCloudType());
        currentSurfaceCloud.reset(new PointCloudType());
        globalMap.reset(new PointCloudType());
        surfaceWorld.reset(new PointCloudType());
        feature_undistort.reset(new PointCloudType());
        feature_dsf_body.reset(new PointCloudType());
        feature_dsf_world.reset(new PointCloudType());



        localMapEdge = nullptr;
        localMapSurface = nullptr;
        kdtreeEdge.setEnableANN();
        kdtreeSurf.setEnableANN();

        // kdtreeEdge.reset(new pcl::KdTreeFLANN<PointType>());
        // kdtreeSurf.reset(new pcl::KdTreeFLANN<PointType>());

        frameID = 0;
        lastKeyFrameID = 0;
        isInit = false;
        isFirstLiDAR = true;
        lidar_pushed = false;
        FIRST_SCAN = true;

        last_timestamp_imu = -1.0;
        last_timestamp_lidar = 0.0;
        lidar_end_time = 0.0;
        scan_num = 0;
        lidar_mean_scantime = 0.0;

        // 地图分割相关
        kdtree_delete_counter = 0;
        add_point_size = 0;
        downSizeFilterSurf.setLeafSize(min_filter_size_surf, min_filter_size_surf, min_filter_size_surf);

        // imu_process.reset(new ImuProcess());
        imu_process = std::make_shared<ImuProcess>();
        
    }
    
    /**
     * @brief 
     * 
     */
    void odometryHandler()
    {
        if (synchronizeMeasurements(measurement))
        {
            if (FIRST_SCAN)
            {   
                first_lidar_time = measurement.lidar_begin_time;
                imu_process->first_lidar_time = first_lidar_time;
                FIRST_SCAN = false;
                return;
            }

            // 前向传播，反向传播，点云运动补偿 最后得到去畸变后的点云 feature_undistort
            imu_process->process(measurement, kf, feature_undistort);

            if (feature_undistort->empty() || feature_undistort == NULL)
            {
                RCLCPP_WARN_STREAM(this->get_logger(), "The feature_undistort is empty skip this scan.");
                return;
            }

            state_now = kf.getState();
            // 计算lidar在世界坐标系下的位置
            // 将 LiDAR 在局部坐标系中的偏移量（offset_T_L_I）通过旋转矩阵转换到世界坐标系下。
            // 再加上当前系统（例如 IMU）在世界坐标系中的位置（position），最终得到 LiDAR 的世界坐标位置 LiDAR_position_w。
            LiDAR_position_w = state_now.position + state_now.rotation_matrix.matrix() * state_now.offset_T_L_I;
            
            // 判断滤波器是否初始化成功
            if ((measurement.lidar_begin_time - first_lidar_time) < INIT_TIME) 
            {
                ekf_is_inited = false;
            } 
            else {
                ekf_is_inited = true;
            }

            // 未实现
            segmentLaserMap();

            // 去畸变后的点云降采样, 得到降采样后的点云
            downSizeFilterSurf.setInputCloud(feature_undistort);
            downSizeFilterSurf.filter(*feature_dsf_body);
            feature_dsf_size = feature_dsf_body->size();
            
            // 点的个数小于5就返回
            if (feature_dsf_size < 5)
            {
                RCLCPP_WARN_STREAM(this->get_logger(), "\033[33mNo point, skip this scan!\033[0m");
                return; 
            }

            // 初始化ikdtree
            if (ikdtree.Root_Node == nullptr)
            {
                
            }
            
            
        }
        
    }


    void segmentLaserMap()
    {
        cube_need_remove.clear();
        kdtree_delete_counter = 0;

        Vec3d p_lidar_w = LiDAR_position_w;
        
        
    }


    /**
     * @brief 打包当前要处理的测量数据
     *  
     * @param currernt_measurement 
     * @return true 
     * @return false 
     */
    bool synchronizeMeasurements(Measurements &currernt_measurement)
    {
        if(lidar_buffer.empty() || imu_buffer.empty())
        {
            RCLCPP_WARN_STREAM(this->get_logger(), "no lidar or imu , please check.");
            return false;
        }

        if (!lidar_pushed)
        {
            currernt_measurement.lidar = lidar_buffer.front();
            currernt_measurement.lidar_begin_time = time_buffer.front();
            
            if (currernt_measurement.lidar->points.size() <= 1)
            {
                lidar_end_time = currernt_measurement.lidar_begin_time + lidar_mean_scantime;              // 这里雷达平均扫描时间为 0 
                RCLCPP_WARN_STREAM(this->get_logger(), "Too few input point cloud.");
            }
            else if (currernt_measurement.lidar->back().curvature / double(1000) < 0.5 * lidar_mean_scantime) 
            {
                lidar_end_time = currernt_measurement.lidar_begin_time + lidar_mean_scantime;
            }
            else
            {
                scan_num++;
                lidar_end_time  = currernt_measurement.lidar_begin_time + currernt_measurement.lidar->points.back().curvature / double(1000.0);
                lidar_mean_scantime += (currernt_measurement.lidar->points.back().curvature / double(1000) - lidar_mean_scantime) / scan_num;
            }

            currernt_measurement.lidar_end_time = lidar_end_time;
            lidar_pushed = true;
            
        }

        if (last_timestamp_imu < lidar_end_time)
        {
            RCLCPP_WARN_STREAM(this->get_logger(), "The latest imu is in LiDAR time range");
            return false;
        }

        // buffer 中的第 i 个imu的时间
        double imu_time_i = getTimeSec(imu_buffer.front()->header.stamp);
        currernt_measurement.imu.clear();
        while (!imu_buffer.empty() && (imu_time_i < lidar_end_time))
        {
            imu_time_i = getTimeSec(imu_buffer.front()->header.stamp);
            if (imu_time_i < lidar_end_time)
            {
                break;
            }
            currernt_measurement.imu.push_back(imu_buffer.front());
            imu_buffer.pop_front();
        }

        lidar_buffer.pop_front();
        time_buffer.pop_front();
        lidar_pushed = false;
        
        return false;
    }   



    /**
     * @brief 订阅IMU数据
     * 
     * @param imuMsg 
     */
    void imuHandler(const sensor_msgs::msg::Imu::SharedPtr imuMsg)
    {
        sensor_msgs::msg::Imu::SharedPtr imu_msg(new sensor_msgs::msg::Imu(*imuMsg));

        // 处理时间
        imu_msg->header.stamp = getRosTime(getTimeSec(imuMsg->header.stamp));  // 此处忽略了 imu 与lidar之间的时间漂移

        // 获取纠偏后的时间戳 
        double timestamp = getTimeSec(imu_msg->header.stamp);
        
        mtx_buffer.lock();

        if (timestamp < last_timestamp_lidar)
        {
            RCLCPP_WARN_STREAM(this->get_logger(), "imu data is in lidar range, clear buffer");
            imu_buffer.clear();
        }
        
        last_timestamp_imu = timestamp;

        imu_buffer.push_back(imu_msg);

        mtx_buffer.unlock();
        sig_buffer.notify_all();
        
    }



    void edgeCloudHandler(const sensor_msgs::msg::PointCloud2::SharedPtr edgeCloudMsg)
    {
        // sensor_msgs::msg::PointCloud2 output;
        // transformPointCloud(edgeCloudMsg, output);
        
        PointCloudType::Ptr temp(new PointCloudType());
        pcl::moveFromROSMsg(*edgeCloudMsg, *temp);
        *currentEdgeCloud = *temp;

        
        // 测试
        // edgeHeader = edgeCloudMsg->header;
    }
    
    void surfaceCloudHandler(const sensor_msgs::msg::PointCloud2::SharedPtr surfaceCloudMsg)
    {   
        mtx_buffer.lock();
        PointCloudType::Ptr current_point_cloud(new PointCloudType());
        pcl::moveFromROSMsg(*surfaceCloudMsg, *current_point_cloud);

        double current_lidar_time = getTimeSec(surfaceCloudMsg->header.stamp);
        if(!isFirstLiDAR && current_lidar_time < last_timestamp_lidar)
        {
            RCLCPP_WARN_STREAM(this->get_logger(), "lidar loop back, clear buffer.");
            lidar_buffer.clear();
        }
        if(isFirstLiDAR)
        {
            isFirstLiDAR = false;
        }

        lidar_buffer.push_back(current_point_cloud);
        time_buffer.push_back(current_lidar_time);
        last_timestamp_lidar = current_lidar_time;

        mtx_buffer.unlock();
        sig_buffer.notify_all();

        
        header = surfaceCloudMsg->header;
       *currentSurfaceCloud = *current_point_cloud;
        
        



        // 第一帧初始化
        // if(!isInit)
        // {
        //     init();
        //     return;
        // }
        

        // 如果初始化成功了 处理点云、位姿估计、构建地图
        // processPointCloud();

        // 退化检测
        // degeneracyDetection();

        // 发布全局地图和位姿，改为了发布局部地图
        // publishPointCloud(pubGlobalMap, globalMap, header.stamp, mapFrame);
        // publishPointCloud(pubcurrentScan, surfaceWorld, header.stamp, mapFrame);
        // RCLCPP_INFO_STREAM(this->get_logger(), "localMapSurface size: " << localMapSurface->size());

        // 发布轨迹    
        // updatePath(estimated_poses.back(), globalPath);
        // publishPath(globalPath, pubPose, header.stamp, mapFrame);
        
        // 发布TF等
        // publisOdometry();

        // 测试
        // surfaceHeader = surfaceCloudMsg->header; 
        // RCLCPP_INFO_STREAM(this->get_logger(), "edge"<< edgeHeader.stamp.sec << "." << edgeHeader.stamp.nanosec << 
        // " surf" << surfaceHeader.stamp.sec << "." << surfaceHeader.stamp.nanosec);
    } 
    
    void degeneracyDetection()
    {
        
        if (!Ft.empty())
        {
            size_t N = Ft.size();
            Eigen::MatrixXd mat_Ft(3, N);
            Eigen::MatrixXd It(3, N);
            // 把Ft变成 3 x N的矩阵
            for (size_t i = 0; i < N; i++)
            {
                mat_Ft.col(i) = Ft[i];
            }

            // 计算特征向量 
            Eigen::Matrix<double, 3, 3> Vt; 
            Eigen::Matrix<double, 3, 3> Vr;      
            Eigen::Vector3d r;   
            solver.compute(Att);
            if (solver.info() == Eigen::Success)
            {
                Vt = solver.eigenvectors();
                r = solver.eigenvalues();
            
            } else
            {
                RCLCPP_WARN_STREAM(this->get_logger(), "can not calculate eigenvector.");
            }

            It.resize(N, 3);
            It = mat_Ft.transpose() * Vt;
            It = It.cwiseAbs();

            Eigen::Vector3d L;
            L = It.colwise().sum();

            int max_value_index, min_value_index;
            L.maxCoeff(&max_value_index);
            L.minCoeff(&min_value_index);

            Eigen::Vector3d Vmax = Vt.col(max_value_index);
            Eigen::Vector3d Vmin = Vt.col(min_value_index);

            minV = Vmin;
            maxV = Vmax;

            Eigen::Vector3d vt1 = Vt.col(0); 
            Eigen::Vector3d vt2 = Vt.col(1); 
            Eigen::Vector3d vt3 = Vt.col(2); 
            
            // RCLCPP_INFO_STREAM(this->get_logger(), "max V: " << Vmax << " min V: " << Vmin);
            // RCLCPP_INFO_STREAM(this->get_logger(), "vt1 x vt2: " << vt1.dot(vt2));
            // RCLCPP_INFO_STREAM(this->get_logger(), "vt2 x vt3: " << vt2.dot(vt3));
            // RCLCPP_INFO_STREAM(this->get_logger(), "vt3 x vt1: " << vt3.dot(vt1));

            // RCLCPP_INFO_STREAM(this->get_logger(), "################");
            // RCLCPP_INFO_STREAM(this->get_logger(), "vt1: " << vt1);
            // RCLCPP_INFO_STREAM(this->get_logger(), "vt2: " << vt2);
            // RCLCPP_INFO_STREAM(this->get_logger(), "vt3: " << vt3);
            // RCLCPP_INFO_STREAM(this->get_logger(), "eigen value: " << r);


            // RCLCPP_INFO_STREAM(this->get_logger(), " min V: \n" << Vmin);
        } else{
            RCLCPP_WARN_STREAM(this->get_logger(), "Ft is empty.");
        }


        Att.setZero();
        Ft.clear();  
    }
    
    
    // 
    void publisOdometry()
    {
        // 获取最新位姿
        SE3 latestPose = estimated_poses.back();
        geometry_msgs::msg::TransformStamped transformStamped;
        // 设置时间戳和坐标系
        transformStamped.header.stamp = header.stamp;
        transformStamped.header.frame_id = mapFrame;  // 时间坐标系
        transformStamped.child_frame_id = "base_link";  // 子坐标系
        
        // 从 pose 中提取平移
        transformStamped.transform.translation.x = latestPose.translation().x();
        transformStamped.transform.translation.y = latestPose.translation().y();
        transformStamped.transform.translation.z = latestPose.translation().z();

        // 从 pose 中提取旋转，假设 pose.rotation() 返回四元数
        Eigen::Quaterniond q = latestPose.so3().unit_quaternion();
        transformStamped.transform.rotation.x = q.x();
        transformStamped.transform.rotation.y = q.y();
        transformStamped.transform.rotation.z = q.z();
        transformStamped.transform.rotation.w = q.w();

        
        br->sendTransform(transformStamped);
        
        // 发布marker
        // 创建 Marker 消息
        visualization_msgs::msg::Marker arrow_marker;

        // 设置时间戳和坐标系
        arrow_marker.header.frame_id = "base_link";  // 父坐标系
        arrow_marker.header.stamp = header.stamp;
        arrow_marker.ns = "vector_arrow";
        arrow_marker.id = 0;  // Marker 的唯一 ID
        arrow_marker.type = visualization_msgs::msg::Marker::ARROW;  // 类型为箭头
        arrow_marker.action = visualization_msgs::msg::Marker::ADD;

        // 设置箭头起点和终点
        geometry_msgs::msg::Point start_point;
        start_point.x = 0.0;
        start_point.y = 0.0;
        start_point.z = 0.0;

        geometry_msgs::msg::Point end_point;
        end_point.x = minV.x();   // 向量的 x 分量
        end_point.y = minV.y();   // 向量的 y 分量
        end_point.z = minV.z();   // 向量的 z 分量

        arrow_marker.points.push_back(start_point);  // 起点
        arrow_marker.points.push_back(end_point);    // 终点

        // 设置箭头颜色
        arrow_marker.color.r = 1.0;  // 红色
        arrow_marker.color.g = 0.0;  // 绿色
        arrow_marker.color.b = 0.0;  // 蓝色
        arrow_marker.color.a = 1.0;  // 透明度

        // 设置箭头的比例
        arrow_marker.scale.x = 0.05;  // 箭头的轴线直径
        arrow_marker.scale.y = 0.1;   // 箭头的头部直径
        arrow_marker.scale.z = 0.1;   // 箭头的头部长度

        // 第二个箭头 
        visualization_msgs::msg::Marker arrow_marker2;
        // 设置第二个箭头的时间戳和坐标系
        arrow_marker2.header.frame_id = "base_link";  // 父坐标系
        arrow_marker2.header.stamp = header.stamp;
        arrow_marker2.ns = "vector_arrow";
        arrow_marker2.id = 1;  // 第二个 Marker 的唯一 ID
        arrow_marker2.type = visualization_msgs::msg::Marker::ARROW;  // 类型为箭头
        arrow_marker2.action = visualization_msgs::msg::Marker::ADD;
        // 设置第二个箭头的起点和终点
        geometry_msgs::msg::Point start_point2;
        start_point2.x = 0.0;  // 起点偏移一些
        start_point2.y = 0.0;
        start_point2.z = 0.0;

        geometry_msgs::msg::Point end_point2;
        end_point2.x = maxV.x();   // 第二个箭头的向量 x 分量
        end_point2.y = maxV.y();   // 第二个箭头的向量 y 分量
        end_point2.z = maxV.z();   // 第二个箭头的向量 z 分量

        arrow_marker2.points.push_back(start_point2);  // 起点
        arrow_marker2.points.push_back(end_point2);    // 终点
        // 设置第二个箭头的颜色
        arrow_marker2.color.r = 0.0;  // 红色
        arrow_marker2.color.g = 1.0;  // 绿色
        arrow_marker2.color.b = 0.0;  // 蓝色
        arrow_marker2.color.a = 1.0;  // 透明度
        // 设置第二个箭头的比例
        arrow_marker2.scale.x = 0.05;  // 箭头的轴线直径
        arrow_marker2.scale.y = 0.1;   // 箭头的头部直径
        arrow_marker2.scale.z = 0.1;   // 箭头的头部长度


        // 发布 Marker 消息
        pubMarker->publish(arrow_marker);
        pubMarker->publish(arrow_marker2);
    }
    


    /**
     * @brief 更新轨迹
     * 
     * @param pose 
     * @param path 
     */
    void updatePath(SE3 pose, nav_msgs::msg::Path &path)
    {
        geometry_msgs::msg::PoseStamped pose_;
        pose_.header.stamp = header.stamp;
        pose_.header.frame_id = mapFrame;
        pose_.pose.position.x = pose.translation()(0);
        pose_.pose.position.y = pose.translation()(1);
        pose_.pose.position.z = pose.translation()(2);

        Eigen::Quaterniond q = pose.so3().unit_quaternion();  // 获取单位四元数
        pose_.pose.orientation.x = q.x();
        pose_.pose.orientation.y = q.y();
        pose_.pose.orientation.z = q.z();
        pose_.pose.orientation.w = q.w();

        path.poses.push_back(pose_);

    }


    void processPointCloud()
    {   
        frameID++;
        RCLCPP_INFO_STREAM(this->get_logger(), "processing cloud" << frameID);
        
        // pcl::io::savePCDFile("./"+std::to_string(frameID)+".pcd", *currentSurfaceCloud);
        // 判断特征点数目
        if (currentEdgeCloud->size() < min_edge_pts)
        {
            RCLCPP_WARN_STREAM(this->get_logger(), "not enough edge points");
            return;
        }

        if (currentSurfaceCloud->size() < min_surf_pts)
        {
            RCLCPP_WARN_STREAM(this->get_logger(), "not enough surface points");
            return;
        }
    
        // RCLCPP_INFO_STREAM(this->get_logger(), "edge_size: " << currentEdgeCloud->size() << " surf_size: " << currentSurfaceCloud->size());

        // 合并一下点云，就是又把角点和面点合起来
        *currentCloud = *currentEdgeCloud;
        *currentCloud += *currentSurfaceCloud;
        
        // 与局部地图匹配
        // auto start = getCurrentTime();
        SE3 pose = alignWithLocalMap(currentEdgeCloud, currentSurfaceCloud);
        // SE3 pose = alignWithLocalMapEfficient(currentEdgeCloud, currentSurfaceCloud);
        // auto end = getCurrentTime();
        // calculateExecutionTime("与局部地图匹配耗时 ",start, end);
        
        PointCloudType::Ptr scan_world(new PointCloudType());
        
        // 把当前帧点云变换到世界坐标系下
        pcl::transformPointCloud(*currentCloud, *scan_world, pose.matrix());
        
        // 当前帧角点面点变换到世界坐标系下
        PointCloudType::Ptr edge_world(new PointCloudType());
        PointCloudType::Ptr surf_world(new PointCloudType());
        
        pcl::transformPointCloud(*currentEdgeCloud, *edge_world, pose.matrix());
        pcl::transformPointCloud(*currentSurfaceCloud, *surf_world, pose.matrix());

        surfaceWorld = surf_world;
        

        if(isKeyFrame(pose)) {
            RCLCPP_INFO_STREAM(this->get_logger(), "Inserting KeyFrame");
            lastKeyFramePose = pose;
            lastKeyFrameID = frameID;

            // 构建局部地图
            edge_deque.emplace_back(edge_world);
            surf_deque.emplace_back(surf_world);

            // 队列大于30就弹出最旧的
            if (edge_deque.size() > num_kfs_in_local_map)
            {
                edge_deque.pop_front();
            }
            if (surf_deque.size() > num_kfs_in_local_map)
            {
                surf_deque.pop_front();
            }
            
            localMapEdge.reset(new PointCloudType());
            localMapSurface.reset(new PointCloudType());

            // 拼接一下局部地图
            for (auto& s : edge_deque)
            {
                *localMapEdge += *s;
            }
            for (auto& s : surf_deque)
            {
                *localMapSurface += *s;
            }
            
            

            // 降采样
            localMapEdge = voxelCloud(localMapEdge, 0.25);
            localMapSurface = voxelCloud(localMapSurface,  0.25);

            // RCLCPP_INFO_STREAM(this->get_logger(), "Insert KeyFrame edge pts: " << localMapEdge->size() << ", surf pts: " << localMapSurface->size());
            // RCLCPP_INFO_STREAM(this->get_logger(), "kdtreeSurf " << kdtreeSurf.size_ << " kdtreeEdge " << kdtreeEdge.size_);
            // 构建新的kdtree
            // kdtreeEdge->setInputCloud(localMapEdge);
            // kdtreeSurf->setInputCloud(localMapSurface);

            // auto start = getCurrentTime();
            kdtreeEdge.buildTree(localMapEdge);
            kdtreeSurf.buildTree(localMapSurface); 
            // auto end = getCurrentTime();
            // calculateExecutionTime("构建kdtree耗时 ", start, end);
            
            *globalMap += *scan_world;
            // 降采样全局地图
            globalMap = voxelCloud(globalMap, 0.05);            
        }       
        
    }
    
    bool isKeyFrame(const SE3 &current_pose)
    {
        if(frameID - lastKeyFrameID > 30)
        {
            return true;
        }

        SE3 delta = lastKeyFramePose.inverse() * current_pose;
        return delta.translation().norm() > kf_distance || delta.so3().log().norm() > kf_angle_deg * kDEG2RAD;

    }

    void init()
    {
        if (localMapEdge == nullptr || localMapSurface == nullptr)
        {
            localMapEdge = currentEdgeCloud;
            localMapSurface = currentSurfaceCloud;


            kdtreeEdge.buildTree(localMapEdge);
            kdtreeSurf.buildTree(localMapSurface);

            // kdtreeEdge->setInputCloud(localMapEdge);
            // kdtreeSurf->setInputCloud(localMapSurface);

            edge_deque.emplace_back(currentEdgeCloud);
            surf_deque.emplace_back(currentSurfaceCloud);
            isInit = true;
            return;
        }
    }

    /**
     * @brief 与局部地图匹配，位姿估计
     * 
     * @param currentEdgeCloud 
     * @param currentSurfaceCloud 
     * @return SE3 
     */
    SE3 alignWithLocalMap(PointCloudType::Ptr currentEdgeCloud, PointCloudType::Ptr currentSurfaceCloud)
    {   
        SE3 pose;
        // 推断最新的位姿世界坐标系下
        if (estimated_poses.size() >= 2) 
        {
            SE3 T1_w = estimated_poses[estimated_poses.size() - 1];
            SE3 T2_w = estimated_poses[estimated_poses.size() - 2];
            pose = T1_w * (T2_w.inverse() * T1_w); // 作为ICP的初值
        }
        
        int edge_size = currentEdgeCloud->size();
        int surf_size = currentSurfaceCloud->size();

        // 我们来写一些并发代码
        for (int iter = 0; iter < max_iteration; ++iter) {
            // auto start = getCurrentTime();
            
            std::vector<bool> effect_surf(surf_size, false);
            std::vector<Eigen::Matrix<double, 1, 6>> jacob_surf(surf_size);  // 点面的残差是1维的
            std::vector<double> errors_surf(surf_size);

            std::vector<bool> effect_edge(edge_size, false);
            std::vector<Eigen::Matrix<double, 3, 6>> jacob_edge(edge_size);  // 点线的残差是3维的
            std::vector<Eigen::Vector3d> errors_edge(edge_size);

            std::vector<int> index_surf(surf_size);
            std::iota(index_surf.begin(), index_surf.end(), 0);  // 填入
            std::vector<int> index_edge(edge_size);
            std::iota(index_edge.begin(), index_edge.end(), 0);  // 填入

            // gauss-newton 迭代
            // 最近邻，角点部分
            if (use_edge_points) {
                std::for_each(std::execution::par_unseq, index_edge.begin(), index_edge.end(), [&](int idx) {
                    Eigen::Vector3d q = ToVec3d(currentEdgeCloud->points[idx]);
                    Eigen::Vector3d qs = pose * q;

                    // 检查最近邻
                    std::vector<int> nn_indices;
                    kdtreeEdge.getClosestPoint(ToPointXYZI(qs), nn_indices, 5);
                    effect_edge[idx] = false;
                    // RCLCPP_INFO_STREAM(this->get_logger(), "edge nn_indices size:" << nn_indices.size());
                    if (nn_indices.size() >= 3) {
                        std::vector<Eigen::Vector3d> nn_eigen;
                        for (auto& n : nn_indices) {
                            nn_eigen.emplace_back(ToVec3d(localMapEdge->points[n]));
                        }

                        // point to point residual
                        Eigen::Vector3d d, p0;
                        if (!fitLine(nn_eigen, p0, d, max_line_distance)) {
                            return;
                        }

                        Eigen::Vector3d err = SO3::hat(d) * (qs - p0);
                        if (err.norm() > max_line_distance) {
                            return;
                        }

                        effect_edge[idx] = true;

                        // build residual
                        Eigen::Matrix<double, 3, 6> J;
                        J.block<3, 3>(0, 0) = -SO3::hat(d) * pose.so3().matrix() * SO3::hat(q);
                        J.block<3, 3>(0, 3) = SO3::hat(d);

                        jacob_edge[idx] = J;
                        errors_edge[idx] = err;
                    }
                });
            }

            /// 最近邻，平面点部分
            if (use_surf_points) {
                std::for_each(std::execution::par_unseq, index_surf.begin(), index_surf.end(), [&](int idx) {
                    Eigen::Vector3d q = ToVec3d(currentSurfaceCloud->points[idx]);
                    Eigen::Vector3d qs = pose * q;
                    // 检查最近邻
                    std::vector<int> nn_indices;
                    kdtreeSurf.getClosestPoint(ToPointXYZI(qs), nn_indices, 5);
                    effect_surf[idx] = false;
                    
                    if (nn_indices.size() == 5) {
                        std::vector<Eigen::Vector3d> nn_eigen;
                        for (auto& n : nn_indices) {
                            nn_eigen.emplace_back(ToVec3d(localMapSurface->points[n]));
                        }

                        

                        // 点面残差
                        Eigen::Vector4d n;
                        if (!fitPlane(nn_eigen, n)) {
                            return;
                        }

                        /*退化检测*/ 
                        // 只用第一次迭代的点
                        if (iter == 0)
                        {
                            // 该点的法向量
                            Eigen::Vector3d n_k;
                            n_k = n.head<3>();
                            
                            // 平移部分
                            Eigen::Matrix<double, 3, 3> att;
                            // 旋转部分
                            Eigen::Matrix<double, 3, 3> arr;
                            
                            // 计算Att并累加
                            att = n_k * n_k.transpose();
                            Att += att;
                            
                            // 计算ck
                            Eigen::Vector3d c_k;
                            c_k = qs.cross(n_k);
                           
                            // 计算Arr
                            arr = c_k * c_k.transpose();
                            Arr += arr;   
                            
                            std::lock_guard<std::mutex> lock(ft_mutex);
                            Ft.emplace_back(n_k); 
                                                
                        }
                        
                        
                        double dis = n.head<3>().dot(qs) + n[3];
                        if (fabs(dis) > max_plane_distance) {
                            return;
                        }

                        effect_surf[idx] = true;

                        // build residual
                        Eigen::Matrix<double, 1, 6> J;
                        J.block<1, 3>(0, 0) = -n.head<3>().transpose() * pose.so3().matrix() * SO3::hat(q);
                        J.block<1, 3>(0, 3) = n.head<3>().transpose();

                        jacob_surf[idx] = J;
                        errors_surf[idx] = dis;
                    }
                });
            }

            // 累加Hessian和error,计算dx
            // 原则上可以用reduce并发，写起来比较麻烦，这里写成accumulate
            double total_res = 0;
            int effective_num = 0;

            Eigen::Matrix<double, 6, 6> H = Eigen::Matrix<double, 6, 6>::Zero();
            Eigen::Matrix<double, 6, 1> err = Eigen::Matrix<double, 6, 1>::Zero();

            for (const auto& idx : index_surf) {
                if (effect_surf[idx]) {
                    H += jacob_surf[idx].transpose() * jacob_surf[idx];
                    err += -jacob_surf[idx].transpose() * errors_surf[idx];
                    effective_num++;
                    total_res += errors_surf[idx] * errors_surf[idx];
                }
            }

            for (const auto& idx : index_edge) {
                if (effect_edge[idx]) {
                    H += jacob_edge[idx].transpose() * jacob_edge[idx];
                    err += -jacob_edge[idx].transpose() * errors_edge[idx];
                    effective_num++;
                    total_res += errors_edge[idx].norm();
                }
            }

            if (effective_num < min_effective_pts) {
                RCLCPP_WARN_STREAM(this->get_logger(), "effective num too small: " << effective_num);
                return pose;
            }

            Eigen::Matrix<double, 6, 1> dx = H.inverse() * err;
            pose.so3() = pose.so3() * SO3::exp(dx.head<3>());
            pose.translation() += dx.tail<3>();

            // 更新
            // RCLCPP_INFO_STREAM(this->get_logger(),"iter " << iter << " total res: " << total_res << ", eff: " << effective_num << ", mean res: " << total_res / effective_num << ", dxn: " << dx.norm());


            if (dx.norm() < eps) {
                RCLCPP_INFO_STREAM(this->get_logger(), "converged, dx = " << dx.transpose());
                break;
            }
            // auto end = getCurrentTime();
            // calculateExecutionTime("本次迭代耗时", start, end);  
        
        }

        estimated_poses.emplace_back(pose);
        return pose;

    }

    /**
     * @brief 与局部地图匹配，高效版本
     * 
     * @param currentEdgeCloud 
     * @param currentSurfaceCloud 
     * @return SE3 
     */
    SE3 alignWithLocalMapEfficient(PointCloudType::Ptr currentEdgeCloud, PointCloudType::Ptr currentSurfaceCloud) 
    {
        SE3 pose;

        // 推断最新的位姿世界坐标系下
        if (estimated_poses.size() >= 2) {
            SE3 T1_w = estimated_poses[estimated_poses.size() - 1];
            SE3 T2_w = estimated_poses[estimated_poses.size() - 2];
            pose = T1_w * (T2_w.inverse() * T1_w); // 作为ICP的初值
        }

        int edge_size = currentEdgeCloud->size();
        int surf_size = currentSurfaceCloud->size();

        // Parallelized storage
        std::vector<bool> effect_surf(surf_size, false);
        std::vector<Eigen::Matrix<double, 1, 6>> jacob_surf(surf_size);
        std::vector<double> errors_surf(surf_size);

        std::vector<bool> effect_edge(edge_size, false);
        std::vector<Eigen::Matrix<double, 3, 6>> jacob_edge(edge_size);
        std::vector<Eigen::Vector3d> errors_edge(edge_size);

        std::vector<int> index_surf(surf_size);
        std::iota(index_surf.begin(), index_surf.end(), 0); 
        std::vector<int> index_edge(edge_size);
        std::iota(index_edge.begin(), index_edge.end(), 0);

        for (int iter = 0; iter < max_iteration; ++iter) {
            // auto start = getCurrentTime();
            double total_res = 0;
            int effective_num = 0;

            Eigen::Matrix<double, 6, 6> H = Eigen::Matrix<double, 6, 6>::Zero();
            Eigen::Matrix<double, 6, 1> err = Eigen::Matrix<double, 6, 1>::Zero();

            // Parallelize edge points processing
            if (use_edge_points) {
                std::for_each(std::execution::par_unseq, index_edge.begin(), index_edge.end(), [&](int idx) {
                    Eigen::Vector3d q = ToVec3d(currentEdgeCloud->points[idx]);
                    Eigen::Vector3d qs = pose * q;
                    if (!std::isfinite(qs.x()) || !std::isfinite(qs.y()) || !std::isfinite(qs.z())) return;

                    std::vector<int> nn_indices;
                    kdtreeEdge.getClosestPoint(ToPointXYZI(qs), nn_indices, 5);

                    effect_edge[idx] = false;
                    if (nn_indices.size() >= 3) {
                        std::vector<Eigen::Vector3d> nn_eigen;
                        for (auto& n : nn_indices) {
                            nn_eigen.emplace_back(ToVec3d(localMapEdge->points[n]));
                        }

                        Eigen::Vector3d d, p0;
                        if (!fitLine(nn_eigen, p0, d, max_line_distance)) return;

                        Eigen::Vector3d err = SO3::hat(d) * (qs - p0);
                        if (err.norm() > max_line_distance) return;

                        effect_edge[idx] = true;

                        // build residual
                        Eigen::Matrix<double, 3, 6> J;
                        J.block<3, 3>(0, 0) = -SO3::hat(d) * pose.so3().matrix() * SO3::hat(q);
                        J.block<3, 3>(0, 3) = SO3::hat(d);

                        jacob_edge[idx] = J;
                        errors_edge[idx] = err;
                    }
                });
            }

            // Parallelize surface points processing
            if (use_surf_points) {
                std::for_each(std::execution::par_unseq, index_surf.begin(), index_surf.end(), [&](int idx) {
                    Eigen::Vector3d q = ToVec3d(currentSurfaceCloud->points[idx]);
                    Eigen::Vector3d qs = pose * q;

                    if (!std::isfinite(qs.x()) || !std::isfinite(qs.y()) || !std::isfinite(qs.z())) return;

                    std::vector<int> nn_indices;
                    // auto start = getCurrentTime();
                    kdtreeSurf.getClosestPoint(ToPointXYZI(qs), nn_indices, 5);
                    // auto end = getCurrentTime();
                    // calculateExecutionTime("最近邻搜索耗时: ", start, end);

                    effect_surf[idx] = false;
                    if (nn_indices.size() == 5) {
                        std::vector<Eigen::Vector3d> nn_eigen;
                        for (auto& n : nn_indices) {
                            nn_eigen.emplace_back(ToVec3d(localMapSurface->points[n]));
                        }

                        Eigen::Vector4d n;
                        if (!fitPlane(nn_eigen, n)) return;

                        double dis = n.head<3>().dot(qs) + n[3];
                        if (fabs(dis) > max_plane_distance) return;

                        effect_surf[idx] = true;

                        // build residual
                        Eigen::Matrix<double, 1, 6> J;
                        J.block<1, 3>(0, 0) = -n.head<3>().transpose() * pose.so3().matrix() * SO3::hat(q);
                        J.block<1, 3>(0, 3) = n.head<3>().transpose();

                        jacob_surf[idx] = J;
                        errors_surf[idx] = dis;
                    }
                });
            }

            // Accumulate Hessian and error, calculate dx
            for (const auto& idx : index_surf) {
                if (effect_surf[idx]) {
                    H += jacob_surf[idx].transpose() * jacob_surf[idx];
                    err += -jacob_surf[idx].transpose() * errors_surf[idx];
                    effective_num++;
                    total_res += errors_surf[idx] * errors_surf[idx];
                }
            }

            for (const auto& idx : index_edge) {
                if (effect_edge[idx]) {
                    H += jacob_edge[idx].transpose() * jacob_edge[idx];
                    err += -jacob_edge[idx].transpose() * errors_edge[idx];
                    effective_num++;
                    total_res += errors_edge[idx].norm();
                }
            }

            if (effective_num < min_effective_pts) {
                RCLCPP_WARN_STREAM(this->get_logger(), "effective num too small: " << effective_num);
                return pose;
            }

            Eigen::Matrix<double, 6, 1> dx = H.inverse() * err;
            pose.so3() = pose.so3() * SO3::exp(dx.head<3>());
            pose.translation() += dx.tail<3>();

            // Update
            RCLCPP_INFO_STREAM(this->get_logger(), "iter " << iter << " total res: " << total_res << ", eff: " << effective_num << ", mean res: " << total_res / effective_num << ", dxn: " << dx.norm());

            if (dx.norm() < eps) {
                RCLCPP_INFO_STREAM(this->get_logger(), "converged, dx = " << dx.transpose());
                break;  // Early exit if converged
            }
            // auto end = getCurrentTime();
            // calculateExecutionTime("本次迭代耗时", start, end);
        }

        return pose;
    }


};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);

    rclcpp::NodeOptions options;
    options.use_intra_process_comms(true);
    rclcpp::executors::MultiThreadedExecutor exec;
    auto odme = std::make_shared<Odometry>(options);
    exec.add_node(odme);
    RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "\033[1;32m Odometry Started.\033[0m");
    exec.spin();

    rclcpp::shutdown();
    
    return 0;
}




