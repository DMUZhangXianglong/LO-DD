/*
 * @Author: DMUZhangXianglong 347913076@qq.com
 * @Date: 2024-12-15 12:42:30
 * @LastEditors: DMUZhangXianglong 347913076@qq.com
 * @LastEditTime: 2025-01-18 03:56:16
 * @FilePath: /LO-DD/include/lo_dd/eskfom.hpp
 * @Description: 
 */
#pragma once // 只包含一次当前头文件

#include "IKFoM.hpp"
#include "utility.hpp"
#include "ikd_Tree.h"

using covariance_matrix = Eigen::Matrix<double, 24, 24>;
using state_vector = Eigen::Matrix<double, 24, 1>;

namespace esekfom{
    PointCloudType::Ptr normvec(new PointCloudType(100000, 1));		  //特征点在地图中对应的平面参数(平面的单位法向量,以及当前点到平面距离)
    PointCloudType::Ptr laserCloudOri(new PointCloudType(100000, 1)); //有效特征点
    PointCloudType::Ptr corr_normvector(new PointCloudType(100000, 1)); //有效特征点对应点法相量
    bool point_selected_surf[100000] = {1};							  //判断是否是有效特征点(c++ 1表示true 0表示false)



    struct  dynamic_share_data
    {
        bool valid;                                                // 有效特征点数是否满足需求
        bool converge;                                             // 迭代时是否收敛
        Eigen::Matrix<double, Eigen::Dynamic, 1> h;	               // 残差 (公式(14)中的z)
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> h_x; //雅可比矩阵H (公式(14)中的H)
    };
    


    class eskf
    {
        private:
            state_ikfom x_;                                         // 状态量24维                                     
            covariance_matrix P_ = covariance_matrix::Identity();   // 协方差矩阵

        public:
            eskf(){}
            ~eskf(){}


            // 观测方程
            /**
             * @brief 计算观测方程的雅可比和残差
             * 
             * @param ekfom_data        观测方程的数据，主要包含是否收敛 残差 雅可比 特征点是否有效
             * @param features_dsf_body lidar坐标系下的下采样后的点
             * @param ikdtree           ikdtree    
             * @param nearest_points    最近邻的点
             * @param extrinsic_est     是否需要估计外参    
             * @return * void 
             */
            void h_share_model(dynamic_share_data &ekfom_data, PointCloudType::Ptr &features_dsf_body, KD_TREE<PointType> &ikdtree, vector<PointVector> &nearest_points, bool extrinsic_est)    
            {
                int features_dsf_size = features_dsf_body->points.size();
                laserCloudOri->clear();
                corr_normvector->clear();
                
                // 此处需要参考高翔的代码需要修改一下
                #ifdef MP_EN
                   omp_set_num_threads(MP_PROC_NUM);
                    #pragma omp parallel for
                #endif
                for(int i = 0; i < features_dsf_size; i++)
                {   

                    PointType &point_body = features_dsf_body->points[i];
                    PointType point_world;
                    Vec3d p_body(point_body.x, point_body.y, point_body.z);
                    Vec3d p_world(x_.rotation_matrix * (x_.offset_R_L_I * p_body + x_.offset_T_L_I) + x_.position);
                    point_world.x = p_world(0);
                    point_world.y = p_world(1);
                    point_world.z = p_world(2);
                    point_world.intensity = point_body.intensity;

                    // 点搜索距离
                    vector<float> pointSearchSqDis(NUM_MATCH_POINTS);
                    auto &points_near = nearest_points[i]; // Nearest_Points[i]打印出来发现是按照离point_world距离，从小到大的顺序的vector

                    // 如果迭代滤波器收敛了
                    if (ekfom_data.converge) 
                    {
                        // point_world是被查找的点， point_world找点的个数，point_world找到的最近点，pointSearchSqDis最近点到点的距离
                        ikdtree.Nearest_Search(point_world, NUM_MATCH_POINTS, points_near, pointSearchSqDis);
                        // 筛选点是否为有效点
                        if (points_near.size() < NUM_MATCH_POINTS) {
                            point_selected_surf[i] = false;  // 最近点数量不足
                        } else if (pointSearchSqDis[NUM_MATCH_POINTS - 1] > 5) {
                            point_selected_surf[i] = false;  // 最近点的距离过大
                        } else {
                            point_selected_surf[i] = true;   // 满足所有条件
                        }
                    }
                    // 如果该点无效，那么结束本次for循环
                    if (!point_selected_surf[i])
                    {
                        continue;
                    }
                    
                    Eigen::Matrix<float, 4, 1> plane_abcd; // 平面点    
                    point_selected_surf[i] = false; 

                    // 平面拟合ax+by+cz+d=0, 实际上points_near就是地图中与point_world最近的点用来你和平面
                    if (esti_plane(plane_abcd, points_near, 0.1f)) 
                    {
                        float point2plane_dis = plane_abcd(0) * point_world.x + plane_abcd(1) * point_world.y + plane_abcd(2) * point_world.z + plane_abcd(3);
                        
                        //如果残差大于经验阈值，则认为该点是有效点  简言之，距离原点越近的lidar点  要求点到平面的距离越苛刻
                        float s = 1 - 0.9 * fabs(point2plane_dis) / sqrt(p_body.norm()); // 这是一个经验公式了
                        if (s > 0.9) 
                        {   
                            // 储蓄有效点到平面的距离以及该平面的法向量
                            point_selected_surf[i] = true;
                            normvec->points[i].x = plane_abcd(0);
                            normvec->points[i].y = plane_abcd(1);
                            normvec->points[i].z = plane_abcd(2);
                            normvec->points[i].intensity = point2plane_dis;
                        }
                    }   
                }

                int effct_feat_num = 0;
                for (int i = 0; i < features_dsf_size; i++)
                {
                    if (point_selected_surf[i] == true)
                    {
                        // 保存有效特征点和其对应的平面法向量
                        laserCloudOri->points[effct_feat_num] = features_dsf_body->points[i];
                        corr_normvector->points[effct_feat_num] = normvec->points[i];
                        effct_feat_num++;
                    }
                }

                if (effct_feat_num < 1)
                {
                    ekfom_data.valid = false;
                    RCLCPP_WARN_STREAM(rclcpp::get_logger("ESKFOM"), "No Effective Points!");
                    return;
                }

                // 计算雅可比和残差
                ekfom_data.h_x = Eigen::MatrixXd::Zero(effct_feat_num, 12); 
                ekfom_data.h.resize(effct_feat_num);
                for (int i = 0; i < effct_feat_num; i++)
                {
                    Vec3d point_lidar(laserCloudOri->points[i].x, laserCloudOri->points[i].y, laserCloudOri->points[i].z);
                    // point_lidar 的反对称矩阵形式
                    Mat3d point_lidar_crossmat;
                    point_lidar_crossmat << SKEW_SYM_MATRX(point_lidar);
                    
                    // imu坐标系下的点
                    Vec3d point_imu = x_.offset_R_L_I * point_lidar + x_.offset_T_L_I;
                    // 其反对称矩阵形式
                    Mat3d point_imu_crossmat;
                    point_imu_crossmat << SKEW_SYM_MATRX(point_imu);

                    // 第i个点对应的平面法向量
                    const PointType &norm_p = corr_normvector->points[i];
                    Vec3d norm_p_vec(norm_p.x, norm_p.y, norm_p.z);

                    // 计算雅可比矩阵H, 公式中的法向量应该是 1x3 的
                    Vec3d C(x_.rotation_matrix.matrix().transpose() * norm_p_vec);
                    Vec3d A(point_imu_crossmat * C);
                    // 如果需要估计外参
                    if(extrinsic_est)
                    {
                        Vec3d B(point_lidar_crossmat * x_.offset_R_L_I.matrix().transpose() * C);
                        ekfom_data.h_x.block<1, 12>(i, 0) << norm_p.x, norm_p.y, norm_p.z, VEC_FROM_ARRAY(A), VEC_FROM_ARRAY(B), VEC_FROM_ARRAY(C);
                    }
                    else
                    {
                        ekfom_data.h_x.block<1, 12>(i, 0) << norm_p.x, norm_p.y, norm_p.z, VEC_FROM_ARRAY(A), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
                    }

                    // 残差
                    ekfom_data.h(i) = -norm_p.intensity; 
                }
            }


            // 滤波器更新
            /**
             * @brief 滤波器更新
             * 
             * @param R 
             * @param features_dsf_body lidar坐标系下的降采样后的点
             * @param ikdtree           ikdtree对象
             * @param nearest_points    最近邻点
             * @param max_iter          最大迭代次数
             * @param extrinsic_est     是够估计外参
             */
            void eskfUpdate(double R, PointCloudType::Ptr &features_dsf_body, KD_TREE<PointType> &ikdtree, vector<PointVector> &nearest_points, int max_iter, bool extrinsic_est)
            {
                normvec->resize(int(features_dsf_body->points.size()));
                
                dynamic_share_data dyna_share;
                dyna_share.valid = true;
                dyna_share.converge = true;
                int t = 0;

                // 前向传播得到状态和协方差矩阵，x_ 是预测得到的状态
                state_ikfom x_propagated = x_;
                covariance_matrix P_propagated = P_;

                // 24 x 1 的状态向量
                state_vector delta_x_new = state_vector::Zero();

                // 误差状态迭代，max_iter 是卡尔曼滤波的最大迭代次数
                for (int i = -1; i < max_iter; i++)
                {
                    // 特征点数满足要求
                    dyna_share.valid = true;

                    // 计算观测方程里的雅可比
                    h_share_model(dyna_share, features_dsf_body, ikdtree, nearest_points, extrinsic_est);

                    // 如果这次点里有无效的点，那么这次迭代就跳过
                    if (!dyna_share.valid)
                    {
                        continue;
                    }

                    // 计算更新的增量
                    state_vector delta_x;
                    delta_x_new = boxminus(x_, x_propagated); // 公式(18)中的 x^k - x^， 第k次迭代的误差状态

                    // 最后的 H 矩阵是 m x 24 的矩阵， 其中 m 是特征点的个数，但是只有前12列不为0
                    Eigen::Matrix<double, Eigen::Dynamic, 24> H;
                    int rows = dyna_share.h_x.rows();
                    H = Eigen::MatrixXd::Zero(rows, 24);
                    H.block(0, 0, rows, 12) = dyna_share.h_x;

                    auto H_block = dyna_share.h_x; // m x 12 的矩阵， 非 0 部分
                    Eigen::Matrix<double, 24, 24> HTH = Eigen::Matrix<double, 24, 24>::Zero(); // H^T 乘 H 24 x 24 的矩阵
                    HTH.block<12, 12>(0, 0) = H_block.transpose() * H_block; // 把非0块放进去
                    
                    // 计算卡尔曼增益 K
                    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> K;
                    auto K_1 = ((HTH / R) + P_.inverse()).inverse(); 
                    auto K_2 = H.transpose() / R; 
                    K = K_1 * K_2;

                    // 2025年1月18日03:56:32
                    

                    
                }
                

                                                                                                       


            }


            state_ikfom getState()
            {
                return x_;
            }

            covariance_matrix getCovarianceMatrix()
            {
                return P_;
            }

            void setState(state_ikfom &input_state)
            {
                x_ = input_state;
            }

            void setCovarianceMatrix(covariance_matrix &input_cov_mat)
            {   
                P_ = input_cov_mat;
            }

            /**
             * @brief 利用imu进行状态预测
             * 
             * @param dt 时间
             * @param Q  噪声的协方差矩阵
             * @param u  输入
             */
            void satePredict(double &dt, Eigen::Matrix<double, 12, 12> &Q, const input_ikfom &u)
            {
                Eigen::Matrix<double, 24, 1> f_ = get_f(x_, u);	  
                Eigen::Matrix<double, 24, 24> f_x_ = df_dx(x_, u); 
                Eigen::Matrix<double, 24, 12> f_w_ = df_dw(x_, u); 

                x_ = boxplus(x_, f_ * dt); //前向传播 公式(4)

                f_x_ = Eigen::Matrix<double, 24, 24>::Identity() + f_x_ * dt;   //之前Fx矩阵里的项没加单位阵，没乘dt   这里补上

                P_ = (f_x_)*P_ * (f_x_).transpose() + (dt * f_w_) * Q * (dt * f_w_).transpose(); 
            
            }

            /**
             * @brief 定义广义加法
             * 
             * @param x x_i
             * @param f_in f(x_i,u,w)
             * @return state_ikfom 
             */
            state_ikfom boxplus(state_ikfom x, Eigen::Matrix<double, 24, 1> f_in)
            {
                state_ikfom x_return;
                x_return.position = x.position + f_in.block<3, 1>(0, 0);
                x_return.rotation_matrix  = x.rotation_matrix * SO3::exp(f_in.block<3, 1>(3, 0));
                
                x_return.offset_R_L_I = x.offset_R_L_I * SO3::exp(f_in.block<3, 1>(6, 0));
                x_return.offset_T_L_I = x.offset_T_L_I + f_in.block<3, 1>(9, 0); 
                
                x_return.velocity = x.velocity + f_in.block<3, 1>(12, 0);
                x_return.bg = x.bg + f_in.block<3, 1>(15, 0);
                x_return.ba = x.ba + f_in.block<3, 1>(18, 0);
                x_return.gravity =  x.gravity + f_in.block<3, 1>(21, 0);

                return x_return;
            }

            /**
             * @brief 定义广义减法
             * 
             * @param x1 
             * @param x2 
             * @return state_vector 
             */
            state_vector boxminus(state_ikfom x1, state_ikfom x2)
            {
                state_vector x_return = state_vector::Zero();
                x_return.block<3, 1>(0, 0) = x1.position - x2.position;
                
                x_return.block<3, 1>(3, 0) = SO3(x2.rotation_matrix.matrix().transpose() * x1.rotation_matrix.matrix()).log();
                x_return.block<3, 1>(6, 0) = SO3(x2.offset_R_L_I.matrix().transpose() * x1.offset_R_L_I.matrix()).log();

                x_return.block<3, 1>(9, 0) = x1.offset_T_L_I - x2.offset_T_L_I;
                x_return.block<3, 1>(12, 0) = x1.velocity - x2.velocity;
                x_return.block<3, 1>(15, 0) = x1.bg - x2.bg;
                x_return.block<3, 1>(18, 0) = x1.ba - x2.ba;
                x_return.block<3, 1>(21, 0) = x1.gravity - x2.gravity;

                return x_return;
            }

            
            
    };
}


