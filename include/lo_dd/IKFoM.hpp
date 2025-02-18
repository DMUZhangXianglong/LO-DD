/*
 * @Author: DMUZhangXianglong 347913076@qq.com
 * @Date: 2024-12-13 00:34:20
 * @LastEditors: DMUZhangXianglong 347913076@qq.com
 * @LastEditTime: 2025-02-15 11:25:35
 * @FilePath: /LO-DD/include/lo_dd/IKFoM.hpp
 * @Description: 
 */
#pragma once // 只包含一次当前头文件
#include "utility.hpp"


// 状态量
struct state_ikfom
{
    Vec3d position = Vec3d(0, 0 ,0);                            // 位置
    SO3 rotation_matrix = SO3(Eigen::Matrix3d::Identity());     // 姿态(IMU到世界坐标系 应该世界坐标系下位姿)
    SO3 offset_R_L_I = SO3(Eigen::Matrix3d::Identity());        // LidAR -> IMU 旋转外参
    Vec3d offset_T_L_I = Vec3d(0, 0, 0);                        // LidAR -> IMU 平移外参
    Vec3d velocity = Vec3d(0, 0, 0);                            // 速度
    Vec3d bg = Vec3d(0, 0, 0);                                  // 角速度偏置
    Vec3d ba = Vec3d(0, 0, 0);                                  // 加速度偏置
    Vec3d gravity = Vec3d(0, 0, -G_m_s2);                       // 重力加速度
};

// 输入u
struct input_ikfom
{
    Vec3d acceleration = Vec3d(0, 0, 0);                    	// 加速度测量值
    Vec3d gyroscope = Vec3d(0 ,0, 0);                       	// 角速度测量值    
};


// 噪声协方差矩阵 Q
Eigen::Matrix<double, 12, 12> process_noise_cov()
{
	Eigen::Matrix<double, 12, 12> Q = Eigen::MatrixXd::Zero(12, 12);
	Q.block<3, 3>(0, 0) = 0.0001 * Eigen::Matrix3d::Identity();
	Q.block<3, 3>(3, 3) = 0.0001 * Eigen::Matrix3d::Identity();
	Q.block<3, 3>(6, 6) = 0.00001 * Eigen::Matrix3d::Identity();
	Q.block<3, 3>(9, 9) = 0.00001 * Eigen::Matrix3d::Identity();

	return Q;
}

// f 公式 3
Eigen::Matrix<double, 24, 1> get_f(state_ikfom state, input_ikfom input)	
{
    // 对应顺序为速度(3)，角速度(3), 外参T(3), 外参旋转R(3)，加速度(3), 角速度偏置(3), 加速度偏置(3), 位置(3)，与论文公式顺序不一致
	Eigen::Matrix<double, 24, 1> f = Eigen::Matrix<double, 24, 1>::Zero();

	// 角速度 = 测量值 - 角速度偏置
    Vec3d omega = input.gyroscope - state.bg;		                                			// 输入的imu的角速度(也就是实际测量值) - 估计的bias值(对应公式的第1行)
	// 世界坐标系下的 加速度 = 旋转矩阵I_2_W * （加速度测量值 - 偏置）
	Vec3d a_inertial = state.rotation_matrix.matrix() * (input.acceleration - state.ba);		// 输入的imu的加速度，先转到世界坐标系（对应公式的第3行）
	
	
	for (int i = 0; i < 3; i++)
	{
		f(i) = state.velocity[i];		                                            			//速度（对应公式第2行）
		f(i + 3) = omega[i];	                                                    			//角速度（对应公式第1行）
		f(i + 12) = a_inertial[i] + state.gravity[i];		                        			//加速度（对应公式第3行）
	}

	return f;
}


// Fx
Eigen::Matrix<double, 24, 24> df_dx(state_ikfom state, input_ikfom input)
{
	Eigen::Matrix<double, 24, 24> Fx = Eigen::Matrix<double, 24, 24>::Zero();
	
    Fx.block<3, 3>(0, 12) = Eigen::Matrix3d::Identity();	                    //对应公式(7)第2行第3列   I
	Vec3d acceleration_ = input.acceleration - state.ba;   	                    //测量加速度 = a_m - bias	

	Fx.block<3, 3>(12, 3) = -state.rotation_matrix.matrix() * SO3::hat(acceleration_);		//对应公式(7)第3行第1列
	Fx.block<3, 3>(12, 18) = -state.rotation_matrix.matrix(); 				                //对应公式(7)第3行第5列 

	Fx.template block<3, 3>(12, 21) = Eigen::Matrix3d::Identity();		        //对应公式(7)第3行第6列   I
	Fx.template block<3, 3>(3, 15) = -Eigen::Matrix3d::Identity();		        //对应公式(7)第1行第4列 (简化为-I)
	return Fx;
}

// Fw
Eigen::Matrix<double, 24, 12> df_dw(state_ikfom state, input_ikfom input)
{
	Eigen::Matrix<double, 24, 12> fw = Eigen::Matrix<double, 24, 12>::Zero();
	fw.block<3, 3>(12, 3) = -state.rotation_matrix.matrix();					// 对应公式(7)第3行第2列  -R 
	fw.block<3, 3>(3, 0) = -Eigen::Matrix3d::Identity();						// 对应公式(7)第1行第1列  -A(w dt)简化为-I
	fw.block<3, 3>(15, 6) = Eigen::Matrix3d::Identity();						// 对应公式(7)第4行第3列  I
	fw.block<3, 3>(18, 9) = Eigen::Matrix3d::Identity();						// 对应公式(7)第5行第4列  I
	return fw;
}