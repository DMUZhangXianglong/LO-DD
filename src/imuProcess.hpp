/*
 * @Author: DMUZhangXianglong 347913076@qq.com
 * @Date: 2024-12-14 15:35:15
 * @LastEditors: DMUZhangXianglong 347913076@qq.com
 * @LastEditTime: 2024-12-21 03:18:13
 * @FilePath: /LO-DD/src/imuProcess.hpp
 * @Description: 实现IMU处理部分
 */

#include "utility.hpp"
#include "IKFoM.hpp"
#include "eskfom.hpp"




#define MAX_INI_COUNT (10)  //最大迭代次数




/**
 * @brief 
 * 
 * @param x 
 * @param y 
 * @return true 
 * @return false 
 */
const bool timeList(PointType &x, PointType &y) 
{
    return (x.curvature < y.curvature);
};


class ImuProcess
{
    private:
        PointCloudType::Ptr current_points_distorted; // 当前帧未去畸变的点云
        sensor_msgs::msg::Imu::ConstSharedPtr last_imu;     // 上一帧IMU指针
        std::vector<Pose6D> IMUpose;                  // Pose6D IMU 位姿向量（反向传播）
        Mat3d R_LiDAR_2_IMU;                          //           
        Vec3d T_LidAR_2_IMU;            
        Vec3d mean_acceleration;                        
        Vec3d mean_gyroscope;
        
        Vec3d last_gyroscope;
        Vec3d last_acceleration;
        Vec3d last_mean_acceleration;

        double start_time_;                           // 开始时间
        double last_lidar_end_time_;                  // 上一帧lidar结束时间戳
        int init_iter_num;                            // 初始迭代次数
        bool FIRST_FRAME_IMU = true;                  // 是否为第一帧
        bool imu_need_init = true;                    // IMU是否需要初始化

        Eigen::Matrix<double, 12, 12> Q;              // 噪声协方差矩阵

        
    public:
        Vec3d cov_acceleration;                       // 加速度协方差
        Vec3d cov_gyroscope;                          // 角速度协方差
        Vec3d cov_acceleration_scale;                 // 初始加速度协方差
        Vec3d cov_gyroscope_scale;                    // 初始角速度协方差
        Vec3d cov_bias_gyroscope;                     // 角速度偏置协方差
        Vec3d cov_bias_acceleration;                  // 加速度偏置协方差
        double first_lidar_time;                      // 当前帧第一个点的时间


        ImuProcess():FIRST_FRAME_IMU(true), imu_need_init(true), start_time_(-1)
        {
            init_iter_num = 1;
            Q = process_noise_cov();
            cov_acceleration      = Vec3d(0.1, 0.1, 0.1);
            cov_gyroscope         = Vec3d(0.1, 0.1, 0.1);
            cov_bias_acceleration = Vec3d(0.0001, 0.0001, 0.0001);
            cov_bias_gyroscope    = Vec3d(0.0001, 0.0001, 0.0001);
            mean_acceleration     = Vec3d(0 ,0, -1.0);
            mean_gyroscope        = Vec3d(0, 0, 0);
            last_gyroscope        = ZeroV3d;
            R_LiDAR_2_IMU         = Identity3d;
            T_LidAR_2_IMU         = ZeroV3d;
            last_imu.reset(new sensor_msgs::msg::Imu());        
        }

        ~ImuProcess() {}
        
        /**
         * @brief 重置IMU数据
         * 
         */
        void imuReset()
        {
            mean_acceleration     = Vec3d(0 ,0, -1.0);
            mean_gyroscope        = Vec3d(0, 0, 0);
            last_gyroscope        = ZeroV3d;
            imu_need_init         = true;
            start_time_       = -1;
            init_iter_num         = 1;
            // v_imu  ？
            IMUpose.clear();
            last_imu.reset(new sensor_msgs::msg::Imu());   
            current_points_distorted.reset(new PointCloudType());
        }

        /**
         * @brief Set the Parameters object
         * 
         * @param t_L_2_I 
         * @param R_L_2_I 
         * @param cov_gyroscope_scale_in 
         * @param cov_acceleration_scale_in 
         * @param cov_bias_gyroscope_in 
         * @param cov_bias_acceleration_in 
         */
        void setParameters(const Vec3d &t_L_2_I, const Mat3d &R_L_2_I, const Vec3d &cov_gyroscope_scale_in, const Vec3d &cov_acceleration_scale_in, const Vec3d &cov_bias_gyroscope_in,  const Vec3d &cov_bias_acceleration_in)
        {
            T_LidAR_2_IMU = t_L_2_I;
            R_LiDAR_2_IMU = R_L_2_I;
            cov_gyroscope_scale = cov_gyroscope_scale_in;
            cov_acceleration_scale = cov_acceleration_scale_in;
            cov_bias_gyroscope = cov_bias_gyroscope_in;
            cov_bias_acceleration = cov_bias_acceleration_in; 
        }

        void imuInit(const Measurements &measurement, esekfom::eskf &kf, int &N)
        {
            Vec3d current_acceleration, current_gyroscope; // 当前加速度和角速度

            if (FIRST_FRAME_IMU)
            {
                imuReset();
                N = 1;
                const auto &imu_acceleration = measurement.imu.front()->linear_acceleration;
                const auto &imu_gyroscope = measurement.imu.front()->angular_velocity;
                mean_acceleration << imu_acceleration.x, imu_acceleration.y, imu_acceleration.z;
                mean_gyroscope << imu_gyroscope.x, imu_gyroscope.y, imu_gyroscope.z;
                first_lidar_time = measurement.lidar_begin_time;                                    // 当前测量的一个雷达点时间
                FIRST_FRAME_IMU = false;
            }

            // 遍历全部imu数据
            for (const auto &imu : measurement.imu)
            {
                const auto &imu_acceleration = imu->linear_acceleration;
                const auto &imu_gyroscope = imu->angular_velocity;
                current_acceleration << imu_acceleration.x, imu_acceleration.y, imu_acceleration.z;
                current_gyroscope << imu_gyroscope.x, imu_gyroscope.y, imu_gyroscope.z;
                
                // 计算平均值
                mean_acceleration += (current_acceleration - mean_acceleration) / N;
                mean_gyroscope    += (current_gyroscope - mean_gyroscope) / N;

                // 加速度和角速度方差更新
                cov_acceleration = cov_acceleration * (N - 1.0) / N + (current_acceleration - mean_acceleration).cwiseProduct(current_acceleration - mean_acceleration) / N;
                cov_gyroscope = cov_gyroscope * (N - 1.0) / N + (current_gyroscope - mean_gyroscope).cwiseProduct(current_gyroscope - mean_gyroscope)  / N / N * (N-1);
               
                N++;
            }

            state_ikfom init_state = kf.getState();
            init_state.gravity = - mean_acceleration / mean_acceleration.norm() * G_m_s2;
            init_state.bg = mean_gyroscope; // 初始的角速度偏置为角速度的平均值
            init_state.offset_R_L_I = SO3(R_LiDAR_2_IMU);
            init_state.offset_T_L_I = T_LidAR_2_IMU;
            kf.setState(init_state); // 设置kf的初始状态
            
            // 初始协方差矩阵
            Eigen::Matrix<double, 24, 24> init_P =  Eigen::MatrixXd::Identity(24,24);
            init_P(6,6) = init_P(7,7) = init_P(8,8) = 0.00001;
            init_P(9,9) = init_P(10,10) = init_P(11,11) = 0.00001;
            init_P(15,15) = init_P(16,16) = init_P(17,17) = 0.0001;
            init_P(18,18) = init_P(19,19) = init_P(20,20) = 0.001;
            init_P(21,21) = init_P(22,22) = init_P(23,23) = 0.00001; 
            kf.setCovarianceMatrix(init_P);
     
            last_imu = measurement.imu.back(); 
        }



        /**
         * @brief IMU初始化 点云运动补偿
         * 
         * @param measurement 
         * @param kf 
         * @param points_distorted 
         */
        void process(const Measurements &measurement, esekfom::eskf &kf, PointCloudType::Ptr &points_distorted)
        {
            if(measurement.imu.empty())
            {
                RCLCPP_WARN_STREAM(rclcpp::get_logger("IMU_Process"), "imu is empty.");
                return;
            }
            
            assert(measurement.lidar != nullptr);

            // 判断是否需要初始化
            if(imu_need_init)
            {
                imuInit(measurement, kf, init_iter_num);
                
                imu_need_init = true;

                last_imu = measurement.imu.back();

                state_ikfom imu_state = kf.getState();
            
                if(init_iter_num > MAX_INI_COUNT) // 至少得有10个imu数据才行？
                {
                    cov_acceleration *= std::pow(G_m_s2 / mean_acceleration.norm(), 2); // 这一句代码有什么用？下面又被覆盖掉了
                    imu_need_init = false;

                    cov_acceleration = cov_acceleration_scale;
                    cov_gyroscope    = cov_gyroscope_scale;
                    RCLCPP_INFO_STREAM(rclcpp::get_logger("IMU_Process"), "IMU has inited.");
                }

                return;
            }

            undistortPointCloud(measurement, kf, *points_distorted);
        }

        void undistortPointCloud(const Measurements &measurement, esekfom::eskf &kf, PointCloudType &points_distorted)
        {
            // 获取当前测量的imu队列
            auto current_imu_buffer = measurement.imu;
            current_imu_buffer.push_front(last_imu);
            const double &imu_end_time = getTimeSec(current_imu_buffer.back()->header.stamp);  // 最后一个imu的时间
            // pc is means point cloud
            const double &pc_begin_time = measurement.lidar_begin_time;
            const double &pc_end_time = measurement.lidar_end_time;

            // 取出本次测量中的点云并且排序
            points_distorted = *(measurement.lidar);
            std::sort(points_distorted.points.begin(), points_distorted.points.end(), timeList);

            // 本次IMU预测开始状态为上次KF估计的后验证
            state_ikfom imu_state_now = kf.getState();
            IMUpose.clear();
            // setPose6D()第一个参数是时间间隔
            IMUpose.push_back(setPose6D(0.0, last_acceleration, last_gyroscope, imu_state_now.velocity, imu_state_now.position, imu_state_now.rotation_matrix.matrix()));


            /********** 预测(前向传播) **********/
            Vec3d average_gyroscope, average_acceleration, imu_gyroscope_k, imu_acceleration_k, imu_velocity_k, imu_position_k;
            Mat3d R_imu_k;

            double dt = 0.0;
            input_ikfom kf_predict_in;
            for (auto imu_k = current_imu_buffer.begin(); imu_k < (current_imu_buffer.end() - 1); imu_k++)
            {
                auto &&imu_now = *(imu_k);
                auto &&imu_next = *(imu_k + 1);

                double imu_now_time  = getTimeSec(imu_now->header.stamp);
                double imu_next_time = getTimeSec(imu_next->header.stamp);
                // 判断时间
                if (getTimeSec(imu_next->header.stamp) < last_lidar_end_time_)
                {
                    continue;
                }
                
                // 角速度平均值
                average_gyroscope << computeAverage(imu_now->angular_velocity.x, imu_next->angular_velocity.x),
                                     computeAverage(imu_now->angular_velocity.y, imu_next->angular_velocity.y),
                                     computeAverage(imu_now->angular_velocity.z, imu_next->angular_velocity.z);

                // 加速度平均值
                average_acceleration << computeAverage(imu_now->linear_acceleration.x, imu_next->linear_acceleration.x),
                                        computeAverage(imu_now->linear_acceleration.y, imu_next->linear_acceleration.y),
                                        computeAverage(imu_now->linear_acceleration.z, imu_next->linear_acceleration.z);

                average_acceleration = average_acceleration * G_m_s2 / mean_acceleration.norm(); 

                if (imu_now_time < last_lidar_end_time_)
                {
                    dt = imu_next_time - last_lidar_end_time_;
                }
                else
                {
                    dt = imu_next_time - imu_now_time;
                }
                
                // KF 预测的输入
                kf_predict_in.acceleration = average_acceleration;
                kf_predict_in.gyroscope = average_gyroscope;
                
                Q.block<3, 3>(0, 0).diagonal() = cov_gyroscope;         // 配置协方差矩阵
                Q.block<3, 3>(3, 3).diagonal() = cov_acceleration;
                Q.block<3, 3>(6, 6).diagonal() = cov_bias_gyroscope;
                Q.block<3, 3>(9, 9).diagonal() = cov_bias_acceleration;

                // 预测
                kf.satePredict(dt, Q, kf_predict_in);

                // 更新imu状态
                imu_state_now = kf.getState();
                last_gyroscope = Vec3d(imu_next->angular_velocity.x, imu_next->angular_velocity.y, imu_next->angular_velocity.z);
                
                last_acceleration = Vec3d(imu_next->linear_acceleration.x, imu_next->linear_acceleration.y, imu_next->linear_acceleration.z);
                last_acceleration = imu_state_now.rotation_matrix * (last_acceleration - imu_state_now.ba) + imu_state_now.gravity;

                // 时间是后一个imu时间到雷达首时间
                double &&offset_t = imu_next_time - pc_begin_time;
                IMUpose.push_back(setPose6D(offset_t, last_acceleration, last_gyroscope, imu_state_now.velocity, imu_state_now.position, imu_state_now.rotation_matrix.matrix()));
            }
            // 最后一帧IMU数据
            dt = std::abs(pc_end_time - imu_end_time);
            kf.satePredict(dt, Q, kf_predict_in);
            imu_state_now = kf.getState();
            // 保存最后一个imu，下一次使用
            last_imu = measurement.imu.back();
            last_lidar_end_time_ = pc_end_time;
            /********** 预测(前向传播) **********/

            /********** 运动补偿(反向传播) **********/
            

            if (points_distorted.begin() == points_distorted.end())
            {
                return;
            }

            auto point_k = points_distorted.points.end() - 1;

            // 遍历预测得到的imu状态
            for (auto imu_pose_k = IMUpose.end() - 1; imu_pose_k != IMUpose.begin(); imu_pose_k--)
            {
                auto imu_pose_last = imu_pose_k - 1;
                auto imu_pose_now  = imu_pose_k;

                // 从IMUpose6D中取出
                R_imu_k << MAT_FROM_ARRAY(imu_pose_last->rotation_matrix);
                imu_position_k << VEC_FROM_ARRAY(imu_pose_last->position);

                imu_acceleration_k << VEC_FROM_ARRAY(imu_pose_now->acceleration);
                imu_gyroscope_k <<  VEC_FROM_ARRAY(imu_pose_now->gyroscope);

                // note: 一个imu状态, 可能对应多个点
                for (; point_k->curvature / double(1000) > imu_pose_last->offset_time; point_k--)
                {
                    // 该点相对于imu_k的时间
                    dt = point_k->curvature / double(1000) - imu_pose_last->offset_time;

                    // 该IMU时间段内 LiDAR 坐标系下的点相对于最后一个点时刻的旋转
                    Mat3d R_i(R_imu_k * SO3::exp(imu_gyroscope_k * dt).matrix());
                    // 该IMU时间段内 LiDAR 坐标系下的点 
                    Vec3d p_i(point_k->x, point_k->y, point_k->z);
                    Vec3d T_ei(imu_position_k + imu_velocity_k * dt + 0.5 * imu_acceleration_k * dt * dt - imu_state_now.position);
                    // 计算运动补偿后lidar坐标系下的点
                    Vec3d p_compensated = imu_state_now.offset_R_L_I.matrix().transpose() * (imu_state_now.rotation_matrix.matrix().transpose() * (T_ei + R_i * (imu_state_now.offset_R_L_I.matrix() * p_i + imu_state_now.offset_T_L_I)) - imu_state_now.offset_T_L_I);
                    
                    point_k->x =  p_compensated(0);
                    point_k->x =  p_compensated(1);
                    point_k->x =  p_compensated(2);
                        
                }

                if (point_k == points_distorted.points.begin())
                {
                    // 反向传播完毕
                    break;
                }
                

            }
            /********** 运动补偿(反向传播) **********/    
        }

        
};