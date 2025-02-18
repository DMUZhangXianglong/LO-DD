/*
 * @Author: DMUZhangXianglong 347913076@qq.com
 * @Date: 2024-11-18 23:16:36
 * @LastEditors: DMUZhangXianglong 347913076@qq.com
 * @LastEditTime: 2025-02-14 20:41:43
 * @FilePath: /LO-DD/src/featureExtraction.cpp
 * @Description: 实现特征提取类
 */

#include "utility.hpp"


// 定义velodyne点云结构
// struct VelodynePointXYZIRT
// {
//     PCL_ADD_POINT4D
//     PCL_ADD_INTENSITY;
//     uint16_t ring;
//     float time;
//     EIGEN_MAKE_ALIGNED_OPERATOR_NEW
// } EIGEN_ALIGN16;
// POINT_CLOUD_REGISTER_POINT_STRUCT (VelodynePointXYZIRT,
//     (float, x, x) (float, y, y) (float, z, z) (float, intensity, intensity)
//     (uint16_t, ring, ring) (float, time, time)
// )


// ouster点云结构
struct OusterPointXYZIRT {
    PCL_ADD_POINT4D;
    float intensity;
    uint32_t t;
    uint16_t reflectivity;
    uint8_t ring;
    // uint16_t noise;
    uint16_t ambient;
    uint32_t range;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;
POINT_CLOUD_REGISTER_POINT_STRUCT(OusterPointXYZIRT,
    (float, x, x) (float, y, y) (float, z, z)
    (float, intensity, intensity)
    (uint32_t, t, t) 
    (uint16_t, reflectivity, reflectivity)
    (uint8_t, ring, ring) 
    // (uint16_t, noise, noise) 
    (uint16_t, ambient, ambient)
    (uint32_t, range, range)
)

namespace velodyne
{
struct EIGEN_ALIGN16 Point {
    PCL_ADD_POINT4D;
    float intensity;
    float time;
    uint16_t ring;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};
}
POINT_CLOUD_REGISTER_POINT_STRUCT(velodyne::Point,
    (float, x, x)
    (float, y, y)
    (float, z, z)
    (float, intensity, intensity)
    (float, time, time)
    (uint16_t, ring, ring))

                                      
struct RoboSensePointXYZIRT
{
    PCL_ADD_POINT4D
    PCL_ADD_INTENSITY;
    uint16_t ring;
    double timestamp;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;
POINT_CLOUD_REGISTER_POINT_STRUCT (RoboSensePointXYZIRT,
    (float, x, x) (float, y, y) (float, z, z) (float, intensity, intensity)
    (uint16_t, ring, ring) (double, timestamp, timestamp)
)

class FeatureExtraction : public ParamServer
{   
    // 
    struct IdAndValue
    {
        IdAndValue(){}


        IdAndValue(int id, double value) : id_(id), value_(value){}
        int id_ = 0;
        double value_ = 0; 
    };

    private:
        /* data */
    public:
        std::mutex mtx_buffer; // buffer 锁
        std::deque<sensor_msgs::msg::PointCloud2> cloud_msg_queue; // 原始点云队列 
        
        
        int scan_count;
        bool is_first_lidar;
        
        sensor_msgs::msg::PointCloud2 currentCloudMsg; // 当前要处理的原始点云 ROS2 形式
        std_msgs::msg::Header cloudHeader;
        
        // 发布edge点和surface点
        rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubEdgePoints;
        rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubSurfacePoints;
        

        pcl::PointCloud<velodyne::Point>::Ptr velodyne_cloud;
        pcl::PointCloud<RoboSensePointXYZIRT>::Ptr tmpRoboSenseCloudIn;
        pcl::PointCloud<OusterPointXYZIRT>::Ptr tmpOusterCloudIn;

        PointCloudType::Ptr edge_points; // 角点
        PointCloudType::Ptr surf_points; // 面点
        PointCloudType::Ptr full_points; // 所有点
        


        // 订阅点云原始数据
        rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subLaserCloud;

        // 构造函数
        FeatureExtraction(const rclcpp::NodeOptions &options) : ParamServer("lo_dd_featureExtraction", options)
        {
            // 订阅
            subLaserCloud = create_subscription<sensor_msgs::msg::PointCloud2>(pointCloudTopic, qos_lidar, std::bind(&FeatureExtraction::cloudHandler, this, std::placeholders::_1));
            // 发布
            pubEdgePoints = create_publisher<sensor_msgs::msg::PointCloud2>("/lo_dd/edge_points", 1);
            pubSurfacePoints = create_publisher<sensor_msgs::msg::PointCloud2>("/lo_dd/surf_points", 1);
            // 分配内存
            allocateMemory();
            // 重置
            reset();
        }
        
        // 析构函数
        ~FeatureExtraction(){}
        
        /**
         * @brief 订阅点云数据进行特征提取并且发布
         * 
         * @param laserCloudMsg 
         */
        void cloudHandler(const sensor_msgs::msg::PointCloud2::SharedPtr laserCloudMsg)
        {   
            // RCLCPP_INFO_STREAM(this->get_logger(), "cloudHandler is running");

            // 缓存点云，如果缓存失败就返回
            if (!cachePointCloud(laserCloudMsg))
            {
                return;
            }
            
            
            // 计时开始
            // auto start = getCurrentTime();
            // featureExtractEfficient(velodyne_cloud, edge_points, surf_points);
            // featureExtract(cloud, edge_points, surf_points);
            // auto end = getCurrentTime();
            // calculateExecutionTime(start, end);
            // 计时结束

            //  发布点云
            if(edge_points->size() > 0 && surf_points->size() > 0)
            {
                publishPointCloud(pubEdgePoints, edge_points, laserCloudMsg->header.stamp, lidarFrame);
                publishPointCloud(pubSurfacePoints, surf_points, laserCloudMsg->header.stamp, lidarFrame);
            }
            else
            {
                RCLCPP_WARN_STREAM(this->get_logger(), "点云发布出现问题");

            }
            // RCLCPP_INFO_STREAM(this->get_logger(), "edge_points "  <<  edge_points->size() <<" surf_points " << surf_points->size());
        
            // reset();
            
        }

        void allocateMemory()
        {
            
            scan_count = 0;
            is_first_lidar = true;
            velodyne_cloud.reset(new pcl::PointCloud<velodyne::Point>());
            
            tmpRoboSenseCloudIn.reset(new pcl::PointCloud<RoboSensePointXYZIRT>());
            tmpOusterCloudIn.reset(new pcl::PointCloud<OusterPointXYZIRT>());       

            edge_points.reset(new PointCloudType()); 
            surf_points.reset(new PointCloudType());  
            full_points.reset(new PointCloudType()); 
                 
        }


        /**
         * @brief 缓存点云信息，根据激光雷达类型处理点云
         * 
         * @param laserCloudMsg 
         * @return true 
         * @return false 
         */
        bool cachePointCloud(const sensor_msgs::msg::PointCloud2::SharedPtr& laserCloudMsg)
        {
            // 点云msg的个数
            scan_count++;
            
            // 原始点云队列
            cloud_msg_queue.push_back(*laserCloudMsg);
            
            if (cloud_msg_queue.size() <= 2)
            {
                return false;
            }
            
            currentCloudMsg = std::move(cloud_msg_queue.front());
            cloud_msg_queue.pop_front();
            cloudHeader = currentCloudMsg.header;
            
            // 处理 velodyne 雷达数据
            if (sensor == SensorType::VELODYNE)
            {            
                processVelodyne(currentCloudMsg);
            }
            else if(sensor == SensorType::ROBOSENSE)
            {
                // pcl::moveFromROSMsg(currentCloudMsg, *tmpRoboSenseCloudIn);
                // cloud->points.resize(tmpRoboSenseCloudIn->size());
                // cloud->is_dense = tmpRoboSenseCloudIn->is_dense;
                // for (size_t i = 0; i < tmpRoboSenseCloudIn->size(); i++)
                // {
                //     auto &src = tmpRoboSenseCloudIn->points[i];
                //     auto &dst = cloud->points[i];
                //     dst.x = src.x;
                //     dst.y = src.y;
                //     dst.z = src.z;
                //     dst.intensity = src.intensity;
                //     dst.ring = src.ring;
                //     // dst.time = src.timestamp - timeScanCur;
                //     dst.time = src.timestamp;
                // }
                
            }
            else if(sensor == SensorType::OUSTER)
            {
                // RCLCPP_INFO_STREAM(this->get_logger(), "IN OUSTER");
                // pcl::moveFromROSMsg(currentCloudMsg, *tmpOusterCloudIn);
                // cloud->points.resize(tmpOusterCloudIn->size());
                // for (size_t i = 0; i < tmpOusterCloudIn->size(); i++)
                // {
                //     auto &src = tmpOusterCloudIn->points[i];
                //     auto &dst = cloud->points[i];
                //     dst.x = src.x;
                //     dst.y = src.y;
                //     dst.z = src.z;
                //     dst.intensity = src.intensity;
                //     dst.ring = src.ring;
                //     dst.time = src.t * 1e-9f;
                // }
                
            }
            else
            {
                RCLCPP_ERROR_STREAM(get_logger(), "Unknown sensor type: " << int(sensor));
                rclcpp::shutdown();
            }
            
            if (currentCloudMsg.is_dense == false)
            {
                RCLCPP_ERROR(get_logger(), "Point cloud is not in dense format, please remove NaN points first!");
                rclcpp::shutdown();
            }

            static int ringFlag = 0;
            if (ringFlag == 0)
            {
                ringFlag = -1;
                for (int i = 0; i < (int)currentCloudMsg.fields.size(); ++i)
                {
                    if (currentCloudMsg.fields[i].name == "ring")
                    {
                        ringFlag = 1;
                        break;
                    }
                }
                if (ringFlag == -1)
                {
                    RCLCPP_ERROR(get_logger(), "Point cloud ring channel not available, please configure your point cloud data!");
                    rclcpp::shutdown();
                }
            }
            
            
            return true;
        }
        
        /**
         * @brief 处理velodyne雷达点云
         * 
         * @param currentCloudMsg 
         * @return * void 
         */
        void processVelodyne(sensor_msgs::msg::PointCloud2 &currentCloudMsg)
        {               
            std::vector<PointCloudType::Ptr> points_buff;  // 每条线束上的点
            
            edge_points->clear();
            surf_points->clear();
            full_points->clear();
            
            // 把 ros2 msg 转 PCL
            pcl::PointCloud<velodyne::Point>::Ptr original_points(new pcl::PointCloud<velodyne::Point>());
            pcl::moveFromROSMsg(currentCloudMsg, *original_points);
            
            // 先去除无效点
            if(original_points->is_dense == false)
            {   
                // 去除无效点
                pcl::PointCloud<velodyne::Point>::Ptr filteredCloud(new pcl::PointCloud<velodyne::Point>());
                for (const auto& point : original_points->points) 
                {
                    if (std::isfinite(point.x) && std::isfinite(point.y) && std::isfinite(point.z)) {
                        filteredCloud->points.push_back(point);
                    }         
                }
                filteredCloud->width = filteredCloud->points.size();
                filteredCloud->height = 1;
                filteredCloud->is_dense = true;
                original_points->swap(*filteredCloud);
                currentCloudMsg.is_dense = true;
            }
            
            int points_size = original_points->size();
            if (points_size == 0)
            {
                RCLCPP_WARN_STREAM(this->get_logger(), "Size of points is 0.");
                return;
            }
            

            if (original_points->points[points_size - 1].time > 0)
            {
                // 这里肯定有时间所以先 pass
                
            }
            
            // 每条线是vector的一个元素
            points_buff.reserve(num_scans);
            for (int i = 0; i < num_scans; i++)
            {
                // 为每条线束分配内存
                points_buff.emplace_back(std::make_shared<PointCloudType>()); 

            }
            
            // 遍历每一个点 分配在每一条线束上
            for (int i = 0; i < points_size; i++)
            {
                int num_ring = original_points->points[i].ring;
                if(num_ring >= num_scans)
                {
                    continue;
                }
                
                PointType temp_pt;
                temp_pt.normal_x = 0.0;
                temp_pt.normal_y = 0.0;
                temp_pt.normal_z = 0.0;
                temp_pt.x = original_points->points[i].x;
                temp_pt.y = original_points->points[i].y;
                temp_pt.z = original_points->points[i].z;
                temp_pt.intensity = original_points->points[i].intensity;
                // 把这个点的时间放在曲率字段里，这个时间相对于该帧点云的相对时间
                temp_pt.curvature = original_points->points[i].time * 1000;
                points_buff[num_ring]->points.push_back(temp_pt); 
            }
            
            // 提取特征 面点 和 角点
            featureExtractEfficient(points_buff, edge_points, surf_points);
            RCLCPP_INFO_STREAM(this->get_logger(), "point: " << surf_points->points.back().curvature);
            
        }



        /**
         * @brief 重置参数
         * 
         * @return * void 
         */
        void reset()
        {
            velodyne_cloud->clear();
            tmpRoboSenseCloudIn->clear();
        }

        /**
         * @brief 
         * 
         * @param edge_points 
         * @param surf_points 
         */
        void featureExtract(const pcl::PointCloud<velodyne::Point>::Ptr current_scans, pcl::PointCloud<PointType>::Ptr edge_points, pcl::PointCloud<PointType>::Ptr surf_points)
        {
            // 每一条线束上的点 初始化
            std::vector<pcl::PointCloud<PointType>::Ptr> scans_in_each_line;
            for (int i = 0; i < num_scans; i++)
            {
                scans_in_each_line.emplace_back(new pcl::PointCloud<PointType>());
            }
            
            for (auto &pt : current_scans->points)
            {
                assert(pt.ring >= 0 && pt.ring < num_scans);
                PointType p;
                p.x = pt.x;
                p.y = pt.y;
                p.z = pt.z;
                p.intensity = pt.intensity;
                
                // 把对应线束上的点放进对应线束点云中
                scans_in_each_line[pt.ring]->points.emplace_back(p);
            }

            // 计算曲率
            for (int i = 0; i < num_scans; i++)
            {
                // 判断点的数量是不是够
                if (scans_in_each_line[i]->points.size() < 131)
                {
                    continue;
                }

                // 定义存放曲率的结构体
                std::vector<IdAndValue> cloud_curvature;
                int total_points = scans_in_each_line[i]->points.size() - 10;
                for (int j = 5; j < (int)scans_in_each_line[i]->points.size() - 5; j++)
                {
                    // 两头留一定余量，采样周围10个点取平均值
                    double diffX = scans_in_each_line[i]->points[j - 5].x + scans_in_each_line[i]->points[j - 4].x +
                                scans_in_each_line[i]->points[j - 3].x + scans_in_each_line[i]->points[j - 2].x +
                                scans_in_each_line[i]->points[j - 1].x - 10 * scans_in_each_line[i]->points[j].x +
                                scans_in_each_line[i]->points[j + 1].x + scans_in_each_line[i]->points[j + 2].x +
                                scans_in_each_line[i]->points[j + 3].x + scans_in_each_line[i]->points[j + 4].x +
                                scans_in_each_line[i]->points[j + 5].x;

                    double diffY = scans_in_each_line[i]->points[j - 5].y + scans_in_each_line[i]->points[j - 4].y +
                                scans_in_each_line[i]->points[j - 3].y + scans_in_each_line[i]->points[j - 2].y +
                                scans_in_each_line[i]->points[j - 1].y - 10 * scans_in_each_line[i]->points[j].y +
                                scans_in_each_line[i]->points[j + 1].y + scans_in_each_line[i]->points[j + 2].y +
                                scans_in_each_line[i]->points[j + 3].y + scans_in_each_line[i]->points[j + 4].y +
                                scans_in_each_line[i]->points[j + 5].y;
                                
                    double diffZ = scans_in_each_line[i]->points[j - 5].z + scans_in_each_line[i]->points[j - 4].z +
                                scans_in_each_line[i]->points[j - 3].z + scans_in_each_line[i]->points[j - 2].z +
                                scans_in_each_line[i]->points[j - 1].z - 10 * scans_in_each_line[i]->points[j].z +
                                scans_in_each_line[i]->points[j + 1].z + scans_in_each_line[i]->points[j + 2].z +
                                scans_in_each_line[i]->points[j + 3].z + scans_in_each_line[i]->points[j + 4].z +
                                scans_in_each_line[i]->points[j + 5].z;
                    IdAndValue distance(j, diffX * diffX + diffY * diffY + diffZ * diffZ);
                    cloud_curvature.push_back(distance);
                }

                // 分成6个区间处理
                for (int j = 0; j < 6; j++)
                {
                    int sector_length = (int)(total_points / 6);
                    int sector_start = sector_length * j;
                    int sector_end = sector_length * (j + 1) - 1;
                    if (j == 5)
                    {
                        sector_end = total_points - 1;
                    }
                    std::vector<IdAndValue> sub_cloud_curvature(cloud_curvature.begin() + sector_start, cloud_curvature.begin() + sector_end);
                
                    // 从一个区间上提取点
                    extractFromSector(scans_in_each_line[i], sub_cloud_curvature, edge_points, surf_points);       
                }     
                
            }
            
        }

        /**
         * @brief 从区间上挑选角点和面点
         * 
         * @param current_scans_i 原始点
         * @param cloud_curvature 点云曲率
         * @param edge_points 输出角点
         * @param surf_points 输出面点
         */
        void extractFromSector(const pcl::PointCloud<PointType>::Ptr &current_scans_i, std::vector<IdAndValue> &cloud_curvature, pcl::PointCloud<PointType>::Ptr &edge_points, pcl::PointCloud<PointType>::Ptr &surf_points)
        {
            std::sort(cloud_curvature.begin(), cloud_curvature.end(), [](const IdAndValue &a, const IdAndValue &b) { return a.value_ < b.value_; });

            int largest_picked_num = 0;
            int point_info_count = 0; 
            
            // 选取曲率最大的点
            std::vector<int> picked_points;
            for (int i = cloud_curvature.size() - 1; i >= 0; i--)
            {
                int p_id = cloud_curvature[i].id_;
                // 检查点是否被选中
                if (std::find(picked_points.begin(), picked_points.end(), p_id) == picked_points.end())
                {
                    if (cloud_curvature[i].value_ <= 0.1)
                    {
                        break;
                    }
                    
                    largest_picked_num++;
                    picked_points.push_back(p_id);
                    if (largest_picked_num <= 20)
                    {
                        edge_points->push_back(current_scans_i->points[p_id]);
                        point_info_count++;
                    } else{
                        break;
                    }

                    for (int k = 1; k < 5; k++)
                    {
                        double diffX = current_scans_i->points[p_id + k].x - current_scans_i->points[p_id + k - 1].x;
                        double diffY = current_scans_i->points[p_id + k].y - current_scans_i->points[p_id + k - 1].y;
                        double diffZ = current_scans_i->points[p_id + k].z - current_scans_i->points[p_id + k - 1].z;
                        if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05) {
                            break;
                        }
                        picked_points.push_back(p_id + k);
                    }

                    for (int k = -1; k >= -5; k--) {
                        double diffX = current_scans_i->points[p_id + k].x - current_scans_i->points[p_id + k + 1].x;
                        double diffY = current_scans_i->points[p_id + k].y - current_scans_i->points[p_id + k + 1].y;
                        double diffZ = current_scans_i->points[p_id + k].z - current_scans_i->points[p_id + k + 1].z;
                        if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05) {
                            break;
                        }
                        picked_points.push_back(p_id + k);
                    }                   
                }               
            }

            for (int i = 0; i <= (int)cloud_curvature.size() - 1; i++) {
                int p_id = cloud_curvature[i].id_;
                if (std::find(picked_points.begin(), picked_points.end(), p_id) == picked_points.end()) 
                {
                    surf_points->push_back(current_scans_i->points[p_id]);
                }
            }
        }

        /**
         * @brief 从区间上挑选角点和面点(提升效率版本)
         * 
         * @param current_scans_i 原始点
         * @param cloud_curvature 点云曲率
         * @param edge_points 输出角点
         * @param surf_points 输出面点
         */
        void featureExtractEfficient(const std::vector<PointCloudType::Ptr> current_scans, pcl::PointCloud<PointType>::Ptr edge_points, pcl::PointCloud<PointType>::Ptr surf_points) 
        {
            // 提前分配内存并初始化每条线束的点云
            std::vector<pcl::PointCloud<PointType>::Ptr> scans_in_each_line(num_scans);
            for (int i = 0; i < num_scans; i++)
            {
                scans_in_each_line[i] = current_scans[i];
            }
            
            // for (int i = 0; i < num_scans; i++) {
            //     scans_in_each_line[i] = std::make_shared<pcl::PointCloud<PointType>>();
            //     scans_in_each_line[i]->reserve(current_scans->points.size() / num_scans); // 假设平均分配
            // }

            // // 分类点云到每条线束
            // for (const auto &pt : current_scans->points) {
            //     assert(pt.ring >= 0 && pt.ring < num_scans);
            //     scans_in_each_line[pt.ring]->push_back({pt.x, pt.y, pt.z, pt.intensity});
            // }

            // 并行化处理每条线束
            #pragma omp parallel for
            for (int i = 0; i < num_scans; i++) {
                const auto &line_points = scans_in_each_line[i];
                if (line_points->points.size() < 131) continue;

                int total_points = line_points->size() - 10;
                std::vector<IdAndValue> cloud_curvature;
                cloud_curvature.reserve(total_points);

                // 计算曲率并缓存结果
                for (int j = 5; j < (int)line_points->size() - 5; j++) {
                    double diffX = -10 * line_points->points[j].x;
                    double diffY = -10 * line_points->points[j].y;
                    double diffZ = -10 * line_points->points[j].z;
                    for (int k = -5; k <= 5; k++) {
                        if (k != 0) {
                            diffX += line_points->points[j + k].x;
                            diffY += line_points->points[j + k].y;
                            diffZ += line_points->points[j + k].z;
                        }
                    }
                    double curvature = diffX * diffX + diffY * diffY + diffZ * diffZ;
                    cloud_curvature.emplace_back(j, curvature);
                }

                // 分区提取角点和面点
                for (int j = 0; j < 6; j++) {
                    int sector_start = j * total_points / 6;
                    int sector_end = (j + 1) * total_points / 6;
                    if (j == 5) sector_end = total_points;

                    extractFromSectorEfficient(line_points, {cloud_curvature.begin() + sector_start, cloud_curvature.begin() + sector_end}, edge_points, surf_points);
                }
            }
        }
        

        /**
         * @brief 从区间上挑选角点和面点(提升效率版本)
         * 
         * @param current_scans_i 原始点
         * @param cloud_curvature 点云曲率
         * @param edge_points 输出角点
         * @param surf_points 输出面点
         */
        void extractFromSectorEfficient(const pcl::PointCloud<PointType>::Ptr &current_scans_i,std::vector<IdAndValue> &&cloud_curvature, pcl::PointCloud<PointType>::Ptr &edge_points, pcl::PointCloud<PointType>::Ptr &surf_points) 
        {
            // 按曲率降序排序
            std::sort(cloud_curvature.begin(), cloud_curvature.end(),
                    [](const IdAndValue &a, const IdAndValue &b) { return a.value_ > b.value_; });

            std::unordered_set<int> picked_points; // 使用哈希表提高点查找效率
            int largest_picked_num = 0;

            // 提取角点
            for (const auto &curvature : cloud_curvature) {
                int p_id = curvature.id_;
                if (picked_points.count(p_id)) continue;

                if (curvature.value_ <= 0.1) break;

                largest_picked_num++;
                picked_points.insert(p_id);

                if (largest_picked_num <= 20) {
                    edge_points->push_back(current_scans_i->points[p_id]);
                } else {
                    break;
                }

                // 标记附近点避免重复选择
                for (int k = 1; k < 5; k++) {
                    if (p_id + k >= (int)current_scans_i->points.size()) break;
                    double diff = calcDistanceSquared(
                        current_scans_i->points[p_id], current_scans_i->points[p_id + k]);
                    if (diff > 0.05) break;
                    picked_points.insert(p_id + k);
                }
                for (int k = -1; k > -5; k--) {
                    if (p_id + k < 0) break;
                    double diff = calcDistanceSquared(
                        current_scans_i->points[p_id], current_scans_i->points[p_id + k]);
                    if (diff > 0.05) break;
                    picked_points.insert(p_id + k);
                }
            }

            // 提取面点
            for (const auto &curvature : cloud_curvature) {
                int p_id = curvature.id_;
                if (!picked_points.count(p_id)) {
                    surf_points->push_back(current_scans_i->points[p_id]);
                }
            }
        }


};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);

    rclcpp::NodeOptions options;
    options.use_intra_process_comms(true);
    rclcpp::executors::MultiThreadedExecutor exec;
    auto feat = std::make_shared<FeatureExtraction>(options);
    exec.add_node(feat);
    RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "\033[1;32m FeatureExtraction Started.\033[0m");
    exec.spin();

    rclcpp::shutdown();
    
    return 0;
}


