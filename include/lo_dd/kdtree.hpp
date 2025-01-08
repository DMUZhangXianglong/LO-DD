/*
 * @Author: DMUZhangXianglong 347913076@qq.com
 * @Date: 2024-11-25 02:15:25
 * @LastEditors: DMUZhangXianglong 347913076@qq.com
 * @LastEditTime: 2024-12-11 02:10:08
 * @FilePath: /LO-DD/include/lo_dd/kdtree.hpp
 * @Description: 实现kdtree
 * */

#pragma once // 只包含一次当前头文件
#include "utility.hpp"

#include <map>
#include <set>

// Kd树节点，二叉树结构，内部用祼指针，对外一个root的shared_ptr
struct KdTreeNode {
    int id_ = -1;
    int point_idx_ = 0;            // 点的索引
    int axis_index_ = 0;           // 分割轴
    float split_thresh_ = 0.0;     // 分割位置
    KdTreeNode* left_ = nullptr;   // 左子树
    KdTreeNode* right_ = nullptr;  // 右子树

    bool IsLeaf() const { return left_ == nullptr && right_ == nullptr; }  // 是否为叶子
};

// 用于记录knn结果
struct NodeAndDistance {
    NodeAndDistance(KdTreeNode* node, float dis2) : node_(node), distance2_(dis2) {}
    KdTreeNode* node_ = nullptr;
    float distance2_ = 0;  // 平方距离，用于比较

    bool operator<(const NodeAndDistance& other) const { return distance2_ < other.distance2_; }
};

class KdTree
{
    private:
        
    public:
        int k_ = 5;                                   // knn最近邻数量
        std::shared_ptr<KdTreeNode> root_ = nullptr;  // 根节点
        std::vector<Eigen::Vector3f> cloud_;          // 输入点云
        std::unordered_map<int, KdTreeNode*> nodes_;  // for bookkeeping
        size_t size_ = 0;                             // 叶子节点数量
        int tree_node_id_ = 0;                        // 为kdtree node 分配id
        // 近似最近邻
        bool approximate_ = true;
        float alpha_ = 0.1;

        explicit KdTree() = default;
        
        ~KdTree() { clear(); }

        // 构建kdtree
        bool buildTree(const PointCloudType::Ptr &cloud)
        {
            if (cloud->empty()) {
                return false;
            }
            
            cloud_.clear();
            cloud_.resize(cloud->size());
            for (size_t i = 0; i < cloud->points.size(); ++i) {
                cloud_[i] = ToVec3f(cloud->points[i]);
            }

            clear();
            reset();

            std::vector<int> idx(cloud->size());
            for (int i = 0; i < cloud->points.size(); ++i) {
                idx[i] = i;
            }

            insert(idx, root_.get());
            return true;
        }


        /// 这个被用于计算最近邻的倍数
        void setEnableANN(bool use_ann = true, float alpha = 0.1) {
            approximate_ = use_ann;
            alpha_ = alpha;
        }

        // 
        void insert(const std::vector<int> &points, KdTreeNode *node)
        {
            nodes_.insert({node->id_, node});

            if (points.empty()) {
                return;
            }

            if (points.size() == 1) {

                size_++;
                node->point_idx_ = points[0];
                return;
            }

            std::vector<int> left, right;
            if (!findSplitAxisAndThresh(points, node->axis_index_, node->split_thresh_, left, right)) {
                size_++;
                node->point_idx_ = points[0];
                return;
            }

            const auto create_if_not_empty = [&node, this](KdTreeNode *&new_node, const std::vector<int> &index) {
                if (!index.empty()) {
                    new_node = new KdTreeNode;
                    new_node->id_ = tree_node_id_++;
                    insert(index, new_node);
                }
            };

            create_if_not_empty(node->left_, left);
            create_if_not_empty(node->right_, right);            
        }

        bool findSplitAxisAndThresh(const std::vector<int> &point_idx, int &axis, float &th, std::vector<int> &left, std::vector<int> &right)
        {
            // 计算三个轴上的散布情况，我们使用math_utils.h里的函数
            Eigen::Vector3f var;
            Eigen::Vector3f mean;
            computeMeanAndCovDiag(point_idx, mean, var, [this](int idx) { return cloud_[idx]; });
            int max_i, max_j;
            var.maxCoeff(&max_i, &max_j);
            axis = max_i;
            th = mean[axis];

            for (const auto &idx : point_idx) {
                if (cloud_[idx][axis] < th) {
                    // 中位数可能向左取整
                    left.emplace_back(idx);
                } else {
                    right.emplace_back(idx);
                }
            }

            // 边界情况检查：输入的points等于同一个值，上面的判定是>=号，所以都进了右侧
            // 这种情况不需要继续展开，直接将当前节点设为叶子就行
            if (point_idx.size() > 1 && (left.empty() || right.empty())) {
                return false;
            }

            // RCLCPP_INFO_STREAM(rclcpp::get_logger("KdTree"), "findSplitAxisAndThresh return true");
            return true;            
        }

        // 
        void clear()
        {
            for (const auto &np : nodes_) 
            {
                if (np.second != root_.get()) {
                    delete np.second;
                }
            }
            nodes_.clear();
            root_ = nullptr;
            size_ = 0;
            tree_node_id_ = 0;
        }

        // 
        void reset() {
            tree_node_id_ = 0;
            root_.reset(new KdTreeNode());
            root_->id_ = tree_node_id_++;
            size_ = 0;
        }  

        // 
        bool getClosestPoint(const PointType &pt, std::vector<int> &closest_idx, int k)
        {   
            if (cloud_.empty())
            {
                RCLCPP_ERROR_STREAM(rclcpp::get_logger("KdTree"), "Tree is empty, cannot search.");
                return false;
            }
            if (k > size_) {
                RCLCPP_ERROR_STREAM(rclcpp::get_logger("KdTree"), "\033[31m" << "cannot set k larger than cloud size: " << k << ", " << size_);
                
                return false;
            }
            k_ = k;

            std::priority_queue<NodeAndDistance> knn_result;
            knn(ToVec3f(pt), root_.get(), knn_result);

            // 排序并返回结果
            closest_idx.resize(knn_result.size());
            for (int i = closest_idx.size() - 1; i >= 0; --i) {
                // 倒序插入
                closest_idx[i] = knn_result.top().node_->point_idx_;
                knn_result.pop();
            }
            return true;            
        }

        // 
        void knn(const Eigen::Vector3f &pt, KdTreeNode *node, std::priority_queue<NodeAndDistance> &knn_result) const
        {
            if (node->IsLeaf()) {
                // 如果是叶子，检查叶子是否能插入
                computeDisForLeaf(pt, node, knn_result);
                return;
            }

            // 看pt落在左还是右，优先搜索pt所在的子树
            // 然后再看另一侧子树是否需要搜索
            KdTreeNode *this_side, *that_side;
            if (pt[node->axis_index_] < node->split_thresh_) {
                this_side = node->left_;
                that_side = node->right_;
            } else {
                this_side = node->right_;
                that_side = node->left_;
            }

            knn(pt, this_side, knn_result);
            if (NeedExpand(pt, node, knn_result)) {  // 注意这里是跟自己比
                knn(pt, that_side, knn_result);
            }            
        }

        // 
        void computeDisForLeaf(const Eigen::Vector3f &pt, KdTreeNode *node, std::priority_queue<NodeAndDistance> &knn_result) const
        {
            // 比较与结果队列的差异，如果优于最远距离，则插入
            float dis2 = Dis2(pt, cloud_[node->point_idx_]);
            if (knn_result.size() < k_) {
                // results 不足k
                knn_result.emplace(node, dis2);
            } else {
                // results等于k，比较current与max_dis_iter之间的差异
                if (dis2 < knn_result.top().distance2_) {
                    knn_result.emplace(node, dis2);
                    knn_result.pop();
                }
            }            
        }

        // 
        bool NeedExpand(const Eigen::Vector3f &pt, KdTreeNode *node, std::priority_queue<NodeAndDistance> &knn_result) const
        {
            if (knn_result.size() < k_) {
                return true;
            }

            if (approximate_) {
                float d = pt[node->axis_index_] - node->split_thresh_;
                if ((d * d) < knn_result.top().distance2_ * alpha_) {
                    return true;
                } else {
                    return false;
                }
            } else {
                // 检测切面距离，看是否有比现在更小的
                float d = pt[node->axis_index_] - node->split_thresh_;
                if ((d * d) < knn_result.top().distance2_) {
                    return true;
                } else {
                    return false;
                }
            }            
        }


              
        
};