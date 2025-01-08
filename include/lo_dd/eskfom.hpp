/*
 * @Author: DMUZhangXianglong 347913076@qq.com
 * @Date: 2024-12-15 12:42:30
 * @LastEditors: DMUZhangXianglong 347913076@qq.com
 * @LastEditTime: 2024-12-21 02:38:01
 * @FilePath: /LO-DD/include/lo_dd/eskfom.hpp
 * @Description: 
 */
#pragma once // 只包含一次当前头文件

#include "IKFoM.hpp"
#include "utility.hpp"

using covariance_matrix = Eigen::Matrix<double, 24, 24>;
using state_vector = Eigen::Matrix<double, 24, 1>;

namespace esekfom{
    


    class eskf
    {
        private:
            state_ikfom x_;                                         // 状态量24维                                     
            covariance_matrix P_ = covariance_matrix::Identity();   // 协方差矩阵

        public:
            eskf(){}
            ~eskf(){}

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

