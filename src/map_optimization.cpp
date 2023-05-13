// 只用纯激光雷达，去掉IMU和GPS，估计激光里程计，可达到近似LIO-SAM的效果
// 参考LIO-SAM的mapOptimization.cpp和A-LOAM的laserMapping.cpp
// 将A-LOAM中的scan-to-scan的粗略激光里程计估计结果作为scan-to-map的初始值（原本的LIO-SAM采用的IMU预积分作为初始值）


// ROS
#include <ros/ros.h>
#include <std_msgs/Header.h>
#include <std_msgs/Float64MultiArray.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/NavSatFix.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <tf/LinearMath/Quaternion.h>
#include <tf/transform_listener.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
// OpenCV
#include <opencv2/imgproc.hpp>
// PCL
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/search/impl/search.hpp>
#include <pcl/range_image/range_image.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/registration/icp.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/crop_box.h>
#include <pcl_conversions/pcl_conversions.h>
// std
#include <vector>
#include <cmath>
#include <algorithm>
#include <queue>
#include <deque>
#include <iostream>
#include <fstream>
#include <ctime>
#include <cfloat>
#include <iterator>
#include <sstream>
#include <string>
#include <limits>
#include <iomanip>
#include <array>
#include <thread>
#include <mutex>
#include <chrono>
// GTSAM
#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/navigation/GPSFactor.h>
#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/ISAM2.h>
// custom utility
#include "aloam_velodyne/common.h"
#include "aloam_velodyne/tic_toc.h"
#include <lo_sam/cloud_info.h>

using namespace gtsam;

using symbol_shorthand::B; // Bias  (ax,ay,az,gx,gy,gz)
using symbol_shorthand::G; // GPS pose
using symbol_shorthand::V; // Vel   (xdot,ydot,zdot)
using symbol_shorthand::X; // Pose3 (x,y,z,r,p,y)

int numberOfCores = 4;
//  A point cloud type that has 6D pose info ([x,y,z,roll,pitch,yaw] intensity is time stamp)6D位姿点云结构定义
struct PointXYZIRPYT
{
    PCL_ADD_POINT4D
    PCL_ADD_INTENSITY; // preferred way of adding a XYZ+padding
    float roll;
    float pitch;
    float yaw;
    double time;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW // make sure our new allocators are aligned
} EIGEN_ALIGN16;                    // enforce SSE padding for correct memory alignment
POINT_CLOUD_REGISTER_POINT_STRUCT(PointXYZIRPYT,
                                  (float, x, x)(float, y, y)(float, z, z)(float, intensity, intensity)(float, roll, roll)(float, pitch, pitch)(float, yaw, yaw)(double, time, time))
typedef PointXYZIRPYT PointTypePose;

// typedef pcl::PointXYZI PointType;

using FactorParam = Eigen::Matrix<double, 3, 4>;

class Map_Optimization
{

public:
    // gtsam
    NonlinearFactorGraph gtSAMgraph;
    Values initialEstimate;
    Values optimizedEstimate;
    ISAM2 *isam;
    Values isamCurrentEstimate;
    Eigen::MatrixXd poseCovariance;

    ros::Publisher pubLaserCloudSurround;
    ros::Publisher pubLaserOdometryGlobal;
    ros::Publisher pubLaserOdometryIncremental;
    ros::Publisher pubKeyPoses;
    ros::Publisher pubPath;

    ros::Publisher pubHistoryKeyFrames;
    ros::Publisher pubIcpKeyFrames;
    ros::Publisher pubRecentKeyFrames;
    ros::Publisher pubRecentKeyFrame;
    ros::Publisher pubCloudRegisteredRaw;
    ros::Publisher pubLoopConstraintEdge;

    ros::Publisher pubRecentKeyFrame_corner;

    ros::Publisher pub_select_corner, pub_select_surf;

    ros::Subscriber subCloud;
    ros::Subscriber subGPS;
    ros::Subscriber subLoop;
    ros::ServiceServer srvSaveMap;

    /**************************************************************************************/
    // 配置参数
    bool loopClosureEnableFlag = true; // 是否开启回环检测
    double loopClosureFrequency = 1.0; // 回环检测频率
    bool savePCD = false;              // 是否保存点云地图
    int N_SCAN = 32;
    int Horizon_SCAN = 1024;
    double mappingProcessInterval = 0.15;
    double globalMapVisualizationSearchRadius = 1000;
    double globalMapVisualizationPoseDensity = 10;
    double globalMapVisualizationLeafSize = 1.0;
    std::string odometryFrame = "camera_init";
    int surroundingKeyframeSize = 50;
    int surroundingKeyframeSearchRadius = 50;
    int edgeFeatureMinValidNum = 10;
    int surfFeatureMinValidNum = 100;
    double z_tollerance = 1000;
    double rotation_tollerance = 1000;
    int historyKeyframeSearchNum = 25;
    double surroundingkeyframeAddingAngleThreshold = 0.2;
    double historyKeyframeSearchRadius = 10.0;
    double historyKeyframeSearchTimeDiff = 30.0;
    double surroundingkeyframeAddingDistThreshold = 1.0;
    double historyKeyframeFitnessScore = 0.3;
    double odometrySurfLeafSize = 0.5;
    double mappingCornerLeafSize = 0.25;
    double mappingSurfLeafSize = 0.5;
    double surroundingKeyframeDensity = 2;
  

    std::string save_map_dir;

    ros::NodeHandle nh_;

    ros::Subscriber edge_cloud_sub, surf_cloud_sub, full_cloud_sub; // 边缘点云、平面点云、所有点云订阅者
    ros::Subscriber odom_init_sub;                                  // 初始激光里程计订阅者

    ros::Subscriber odom_cloud_sub;


    std::deque<sensor_msgs::PointCloud2::ConstPtr> edge_cloud_queue; // 存储边缘点云
    std::deque<sensor_msgs::PointCloud2::ConstPtr> surf_cloud_queue; // 存储平面点云
    std::deque<sensor_msgs::PointCloud2::ConstPtr> full_cloud_queue; // 存储全部点云
    std::deque<nav_msgs::Odometry::ConstPtr> odom_init_queue;        // 存储初始里程计


    std::mutex mutex; // 线程锁

    // pcl存储点云
    // pcl::VoxelGrid<pcl::PointXYZI> downSizeFilterCorner; // 将采样

    pcl::PointCloud<PointType>::Ptr edge_cloud_curr; // 存储最新时刻的边缘点云
    pcl::PointCloud<PointType>::Ptr surf_cloud_curr; // 存储最新时刻的平面点云

    nav_msgs::Odometry odom_init_curr; // 当前时刻的初始里程计

    tf::Transform tf_odom_last;

    std::vector<gtsam::Symbol> X_node; // 用于存储里程计序号
 
    int graph_pose_num = 0;



    Map_Optimization(ros::NodeHandle &nh, ros::NodeHandle nh_private);
    ~Map_Optimization();

    void allocateMemory(); // 分配点云指针内存空间函数


    void odom_cloud_handler(const lo_sam::cloud_info::ConstPtr &odom_cloud_msg); // 回调函数存储初始激光里程计

    sensor_msgs::PointCloud2 publishCloud(ros::Publisher *thisPub, pcl::PointCloud<PointType>::Ptr thisCloud, ros::Time thisStamp, std::string thisFrame); // 发布点云

    void updateInitialGuess();

    void extractSurroundingKeyFrames();
    void extractNearby();
    void extractCloud(pcl::PointCloud<PointType>::Ptr cloudToExtract);
    void extractForLoopClosure();

    void downsampleCurrentScan();

    void scan2MapOptimization();
    void updatePointAssociateToMap();
    void cornerOptimization();
    void surfOptimization();
    void combineOptimizationCoeffs();
    inline Eigen::Matrix<float, 6, 6> transMatrix(Eigen::Matrix<float, 6, 6> &M);
    bool LMOptimization(int iterCount);
    void transformUpdate();

    void saveKeyFramesAndFactor();
    bool saveFrame();
    void addOdomFactor();
    void addLoopFactor();
    void updatePath(const PointTypePose &pose_in);

    void correctPoses();

    void publishOdometry();

    void publishFrames();

    void pointAssociateToMap(PointType const *const pi, PointType *const po);

    pcl::PointCloud<PointType>::Ptr transformPointCloud(pcl::PointCloud<PointType>::Ptr cloudIn, PointTypePose *transformIn);

    gtsam::Pose3 pclPointTogtsamPose3(PointTypePose thisPoint);

    gtsam::Pose3 trans2gtsamPose(float transformIn[]);

    Eigen::Affine3f pclPointToAffine3f(PointTypePose thisPoint);

    Eigen::Affine3f trans2Affine3f(float transformIn[]);

    PointTypePose trans2PointTypePose(float transformIn[]);

    void loopClosureThread();
    void performLoopClosure();
    void visualizeLoopClosure();
    bool detectLoopClosureDistance(int *latestID, int *closestID);
    bool detectLoopClosureExternal(int *latestID, int *closestID);
    void loopFindNearKeyframes(pcl::PointCloud<PointType>::Ptr &nearKeyframes, const int &key, const int &searchNum);

    void visualizeGlobalMapThread();
    void publishGlobalMap();

    void saveMap();

    /**************************************************************************************/

    std::deque<nav_msgs::Odometry> gpsQueue;
    // LO_SAM::cloud_info cloudInfo;

    vector<pcl::PointCloud<PointType>::Ptr> cornerCloudKeyFrames; // 历史所有关键帧的角点集合（降采样)
    vector<pcl::PointCloud<PointType>::Ptr> surfCloudKeyFrames;   // 历史所有关键帧的平面点集合（降采样）

    pcl::PointCloud<PointType>::Ptr cloudKeyPoses3D;     // 历史关键帧位姿（只有3D位置）
    pcl::PointCloud<PointTypePose>::Ptr cloudKeyPoses6D; // 历史关键帧位姿（6DOF）
    pcl::PointCloud<PointType>::Ptr copy_cloudKeyPoses3D;
    pcl::PointCloud<PointTypePose>::Ptr copy_cloudKeyPoses6D;

    pcl::PointCloud<PointType>::Ptr laserCloudCornerLast;   // corner feature set from odoOptimization // 当前激光帧角点集合
    pcl::PointCloud<PointType>::Ptr laserCloudSurfLast;     // surf feature set from odoOptimization  // 当前激光帧平面点集合
    pcl::PointCloud<PointType>::Ptr laserCloudCornerLastDS; // downsampled corner featuer set from odoOptimization // 当前激光帧角点集合，降采样
    pcl::PointCloud<PointType>::Ptr laserCloudSurfLastDS;   // downsampled surf featuer set from odoOptimization // 当前激光帧平面点集合，降采样

    // 当前帧与局部map匹配上了的角点、平面点，加入同一集合；后面是对应点的参数
    pcl::PointCloud<PointType>::Ptr laserCloudOri;
    pcl::PointCloud<PointType>::Ptr coeffSel;

    // 当前帧与局部map匹配上了的角点、参数、标记
    std::vector<PointType> laserCloudOriCornerVec; // corner point holder for parallel computation
    std::vector<PointType> coeffSelCornerVec;
    std::vector<bool> laserCloudOriCornerFlag;
    // 当前帧与局部map匹配上了的平面点、参数、标记
    std::vector<PointType> laserCloudOriSurfVec; // surf point holder for parallel computation
    std::vector<PointType> coeffSelSurfVec;
    std::vector<bool> laserCloudOriSurfFlag;

    map<int, pair<pcl::PointCloud<PointType>, pcl::PointCloud<PointType>>> laserCloudMapContainer;

    pcl::PointCloud<PointType>::Ptr laserCloudCornerFromMap;   // 局部map的角点集合
    pcl::PointCloud<PointType>::Ptr laserCloudSurfFromMap;     // 局部map的平面点集合
    pcl::PointCloud<PointType>::Ptr laserCloudCornerFromMapDS; // 局部map的角点集合，降采样
    pcl::PointCloud<PointType>::Ptr laserCloudSurfFromMapDS;   // 局部map的平面点集合，降采样

    // 局部关键帧构建的map点云，对应kdtree，用于scan-to-map找相邻点
    pcl::KdTreeFLANN<PointType>::Ptr kdtreeCornerFromMap;
    pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurfFromMap;

    pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurroundingKeyPoses;
    pcl::KdTreeFLANN<PointType>::Ptr kdtreeHistoryKeyPoses;

    // 降采样
    pcl::VoxelGrid<PointType> downSizeFilterCorner;
    pcl::VoxelGrid<PointType> downSizeFilterSurf;
    pcl::VoxelGrid<PointType> downSizeFilterICP;
    pcl::VoxelGrid<PointType> downSizeFilterSurroundingKeyPoses; // for surrounding key poses of scan-to-map optimization

    ros::Time timeLaserInfoStamp;
    double timeLaserInfoCur;

    float transformTobeMapped[6]; // r, p, y, x, y, z

    std::mutex mtx;
    std::mutex mtxLoopInfo;

    bool isDegenerate = false;
    cv::Mat matP;

    int laserCloudCornerFromMapDSNum = 0; // 局部map角点数量
    int laserCloudSurfFromMapDSNum = 0;   // 局部map平面点数量
    int laserCloudCornerLastDSNum = 0;    // 当前激光帧角点数量
    int laserCloudSurfLastDSNum = 0;      // 当前激光帧面点数量

    bool aLoopIsClosed = false;
    map<int, int> loopIndexContainer; // from new to old
    vector<pair<int, int>> loopIndexQueue;
    vector<gtsam::Pose3> loopPoseQueue;
    vector<gtsam::noiseModel::Diagonal::shared_ptr> loopNoiseQueue;
    deque<std_msgs::Float64MultiArray> loopInfoVec;

    nav_msgs::Path globalPath;

    Eigen::Affine3f transPointAssociateToMap;       // 当前帧位姿
    Eigen::Affine3f incrementalOdometryAffineFront; // 前一帧位姿
    Eigen::Affine3f incrementalOdometryAffineBack;  // 当前帧位姿

    std::vector<FactorParam> corner_factor, surf_factor;

    std::string ns_, ns_2;

    float constraintTransformation(float value, float limit)
    {
        if (value < -limit)
            value = -limit;
        if (value > limit)
            value = limit;

        return value;
    }

    void convert_factor_to_cloud(const std::vector<FactorParam> &params, pcl::PointCloud<PointType> &cloud, int color)
    {
        PointType p;
        Eigen::Vector3d p1, p2;
        p.intensity = color;
        for (auto factor : params)
        {
            p.x = factor.col(0).x();
            p.y = factor.col(0).y();
            p.z = factor.col(0).z();
            cloud.push_back(p);
            for (int i = 1; i <= 5; ++i)
            {
                p1 = factor.col(0) + factor.col(1) * 0.1 * i;
                p2 = factor.col(0) - factor.col(1) * 0.1 * i;
                p.x = p1.x(), p.y = p1.y(), p.z = p1.z();
                cloud.push_back(p);
                p.x = p2.x(), p.y = p2.y(), p.z = p2.z();
                cloud.push_back(p);
            }
        }
    }

    void visualize_factor(const std::vector<FactorParam> &param, const ros::Time &stamp, int color, const ros::Publisher &pub)
    {
        if (param.empty())
            return;
        static pcl::PointCloud<PointType> cloud;
        static sensor_msgs::PointCloud2 msg;
        cloud.clear();
        if (pub.getNumSubscribers() > 0)
        {
            convert_factor_to_cloud(param, cloud, color);
            pcl::toROSMsg(cloud, msg);
            msg.header.frame_id = odometryFrame;
            msg.header.stamp = stamp;
            pub.publish(msg);
        }
    }
};

Map_Optimization::Map_Optimization(ros::NodeHandle &nh, ros::NodeHandle nh_private)
{
    nh_ = nh;

    // 设置参数
    nh.param<bool>("loopClosureEnableFlag", loopClosureEnableFlag, false); // 是否开启回环检测
    nh.param<double>("loopClosureFrequency", loopClosureFrequency, 1.0);   //  回环检测频率
    nh.param<bool>("savePCD", savePCD, false);                             // 是否保存点云地图
    nh.param<int>("N_SCAN", N_SCAN, 16);
    nh.param<int>("Horizon_SCAN", Horizon_SCAN, 1800);
    nh.param<double>("mappingProcessInterval", mappingProcessInterval, 0.15);                         //
    nh.param<double>("globalMapVisualizationSearchRadius", globalMapVisualizationSearchRadius, 1000); //
    nh.param<double>("globalMapVisualizationPoseDensity", globalMapVisualizationPoseDensity, 15);     //
    nh.param<double>("globalMapVisualizationLeafSize", globalMapVisualizationLeafSize, 1.5);          //
    nh.param<std::string>("odometryFrame", odometryFrame, "camera_init");                             //
    nh.param<int>("surroundingKeyframeSize", surroundingKeyframeSize, 1800);
    nh.param<int>("surroundingKeyframeSearchRadius", surroundingKeyframeSearchRadius, 50);
    nh.param<int>("edgeFeatureMinValidNum", edgeFeatureMinValidNum, 10);
    nh.param<int>("surfFeatureMinValidNum", surfFeatureMinValidNum, 200);
    nh.param<double>("z_tollerance", z_tollerance, 100);                                                       //
    nh.param<double>("rotation_tollerance", rotation_tollerance, 100);                                         //
    nh.param<int>("historyKeyframeSearchNum", historyKeyframeSearchNum, 15);                                   //
    nh.param<double>("surroundingkeyframeAddingAngleThreshold", surroundingkeyframeAddingAngleThreshold, 1.5); //
    nh.param<double>("historyKeyframeSearchRadius", historyKeyframeSearchRadius, 15);                          //
    nh.param<double>("historyKeyframeSearchTimeDiff", historyKeyframeSearchTimeDiff, 1000);                    //
    nh.param<double>("surroundingkeyframeAddingDistThreshold", surroundingkeyframeAddingDistThreshold, 1.5);   //
    nh.param<double>("historyKeyframeFitnessScore", historyKeyframeFitnessScore, 0.5);                         //
    nh.param<double>("odometrySurfLeafSize", odometrySurfLeafSize, 0.5);                                       //
    nh.param<double>("mappingCornerLeafSize", mappingCornerLeafSize, 0.2);                                     //
    nh.param<double>("mappingSurfLeafSize", mappingSurfLeafSize, 0.4);                                         //
    nh.param<double>("surroundingKeyframeDensity", surroundingKeyframeDensity, 2);                             //


    ISAM2Params parameters;
    parameters.relinearizeThreshold = 0.1;
    parameters.relinearizeSkip = 1;
    isam = new ISAM2(parameters);

    // 订阅者初始化，消息均来源于laserOdometry
    odom_cloud_sub = nh.subscribe<lo_sam::cloud_info>("odom_init/odom_cloud", 100, &Map_Optimization::odom_cloud_handler, this); // 订阅初始激光里程计


    pubKeyPoses = nh.advertise<sensor_msgs::PointCloud2>("mapping/trajectory", 1);                     // 发布历史关键帧里程计
    pubLaserCloudSurround = nh.advertise<sensor_msgs::PointCloud2>("mapping/map_global", 1);           // 发布局部关键帧map的特征点云
    pubLaserOdometryGlobal = nh.advertise<nav_msgs::Odometry>("mapping/odometry", 1);                  // 发布激光里程计，rviz中表现为坐标轴
    pubLaserOdometryIncremental = nh.advertise<nav_msgs::Odometry>("mapping/odometry_incremental", 1); // 发布激光里程计，它与上面的激光里程计基本一样，只是roll、pitch用imu数据加权平均了一下，z做了限制
    pubPath = nh.advertise<nav_msgs::Path>("mapping/path", 1);                                         // 发布激光里程计路径，rviz中表现为载体的运行轨迹

    pubHistoryKeyFrames = nh.advertise<sensor_msgs::PointCloud2>("mapping/icp_loop_closure_history_cloud", 1);    // 发布闭环匹配关键帧局部map
    pubIcpKeyFrames = nh.advertise<sensor_msgs::PointCloud2>("mapping/icp_loop_closure_corrected_cloud", 1);      // 发布当前关键帧经过闭环优化后的位姿变换之后的特征点云
    pubLoopConstraintEdge = nh.advertise<visualization_msgs::MarkerArray>("mapping/loop_closure_constraints", 1); // 发布闭环边，rviz中表现为闭环帧之间的连线

    pubRecentKeyFrames = nh.advertise<sensor_msgs::PointCloud2>("mapping/map_local", 1);               // 发布局部map的降采样平面点集合
    pubRecentKeyFrame = nh.advertise<sensor_msgs::PointCloud2>("mapping/cloud_registered", 1);         // 发布历史帧（累加的）的角点、平面点降采样集合
    pubCloudRegisteredRaw = nh.advertise<sensor_msgs::PointCloud2>("mapping/cloud_registered_raw", 1); // 发布当前帧原始点云配准之后的点云

    pubRecentKeyFrame_corner = nh.advertise<sensor_msgs::PointCloud2>("mapping/cloud_registered_corner", 1); // 发布历史帧（累加的）的角点降采样集合

    pub_select_corner = nh.advertise<sensor_msgs::PointCloud2>("select_corner", 3);
    pub_select_surf = nh.advertise<sensor_msgs::PointCloud>("select_surf", 3);

    downSizeFilterCorner.setLeafSize(mappingCornerLeafSize, mappingCornerLeafSize, mappingCornerLeafSize);
    downSizeFilterSurf.setLeafSize(mappingSurfLeafSize, mappingSurfLeafSize, mappingSurfLeafSize);
    downSizeFilterICP.setLeafSize(mappingSurfLeafSize, mappingSurfLeafSize, mappingSurfLeafSize);
    downSizeFilterSurroundingKeyPoses.setLeafSize(surroundingKeyframeDensity, surroundingKeyframeDensity, surroundingKeyframeDensity); // for surrounding key poses of scan-to-map optimization

    allocateMemory();
}

Map_Optimization::~Map_Optimization()
{
    if (savePCD == true)
    {
        saveMap();
    }
}

// 初始化点云指针的内存空间
void Map_Optimization::allocateMemory()
{
    cloudKeyPoses3D.reset(new pcl::PointCloud<PointType>());
    cloudKeyPoses6D.reset(new pcl::PointCloud<PointTypePose>());
    copy_cloudKeyPoses3D.reset(new pcl::PointCloud<PointType>());
    copy_cloudKeyPoses6D.reset(new pcl::PointCloud<PointTypePose>());

    kdtreeSurroundingKeyPoses.reset(new pcl::KdTreeFLANN<PointType>());
    kdtreeHistoryKeyPoses.reset(new pcl::KdTreeFLANN<PointType>());

    laserCloudCornerLast.reset(new pcl::PointCloud<PointType>());   // corner feature set from odoOptimization
    laserCloudSurfLast.reset(new pcl::PointCloud<PointType>());     // surf feature set from odoOptimization
    laserCloudCornerLastDS.reset(new pcl::PointCloud<PointType>()); // downsampled corner featuer set from odoOptimization
    laserCloudSurfLastDS.reset(new pcl::PointCloud<PointType>());   // downsampled surf featuer set from odoOptimization

    laserCloudOri.reset(new pcl::PointCloud<PointType>());
    coeffSel.reset(new pcl::PointCloud<PointType>());

    laserCloudOriCornerVec.resize(N_SCAN * Horizon_SCAN);
    coeffSelCornerVec.resize(N_SCAN * Horizon_SCAN);
    laserCloudOriCornerFlag.resize(N_SCAN * Horizon_SCAN);
    laserCloudOriSurfVec.resize(N_SCAN * Horizon_SCAN);
    coeffSelSurfVec.resize(N_SCAN * Horizon_SCAN);
    laserCloudOriSurfFlag.resize(N_SCAN * Horizon_SCAN);

    std::fill(laserCloudOriCornerFlag.begin(), laserCloudOriCornerFlag.end(), false);
    std::fill(laserCloudOriSurfFlag.begin(), laserCloudOriSurfFlag.end(), false);

    laserCloudCornerFromMap.reset(new pcl::PointCloud<PointType>());
    laserCloudSurfFromMap.reset(new pcl::PointCloud<PointType>());
    laserCloudCornerFromMapDS.reset(new pcl::PointCloud<PointType>());
    laserCloudSurfFromMapDS.reset(new pcl::PointCloud<PointType>());

    kdtreeCornerFromMap.reset(new pcl::KdTreeFLANN<PointType>());
    kdtreeSurfFromMap.reset(new pcl::KdTreeFLANN<PointType>());

    for (int i = 0; i < 6; ++i)
    {
        transformTobeMapped[i] = 0;
    }

    matP = cv::Mat(6, 6, CV_32F, cv::Scalar::all(0));

    edge_cloud_curr.reset(new pcl::PointCloud<PointType>());
    surf_cloud_curr.reset(new pcl::PointCloud<PointType>());
}

void Map_Optimization::odom_cloud_handler(const lo_sam::cloud_info::ConstPtr &odom_cloud_msg)
{
    pcl::fromROSMsg(odom_cloud_msg->cloud_edge, *laserCloudCornerLast);
    pcl::fromROSMsg(odom_cloud_msg->cloud_edge, *laserCloudSurfLast);

    // laserCloudCornerLast = odom_cloud_msg.;
    // laserCloudSurfLast = surf_cloud_curr;

    timeLaserInfoStamp = odom_cloud_msg->header.stamp;
    timeLaserInfoCur = odom_cloud_msg->header.stamp.toSec();

    odom_init_curr = odom_cloud_msg->odom_init;

   

    std::lock_guard<std::mutex> lock(mtx);
    // mapping执行频率控制

    static double timeLastProcessing = -1;
    static int laser_num = 0;

    // std::cout << "time diff:" << timeLaserInfoCur - timeLastProcessing << std::endl;

    if (timeLaserInfoCur - timeLastProcessing >= mappingProcessInterval)
    {
        timeLastProcessing = timeLaserInfoCur;
        ros::Time t1 = ros::Time::now();
        // 1.当前帧位姿初始化： 采用来自laserOdometry的初始激光里程计来初始位姿
        updateInitialGuess();

        // std::cout << "init trans(x,y,z): " << transformTobeMapped[3] << ", " << transformTobeMapped[4] << ", " << transformTobeMapped[5] << std::endl;

        // 2.提取局部角点、平面点云集合，加入局部map
        // (1)对最近的一帧关键帧，搜索时空维度上相邻的关键帧集合，降采样一下
        // (2)对关键帧集合中的每一帧，提取对应的角点、平面点，加入局部map中
        extractSurroundingKeyFrames();

        // 3.当前激光帧角点、平面点集合降采样
        downsampleCurrentScan();

        // 4.scan-to-map优化当前帧位姿
        // (1)要求当前帧特征点数量足够多，且匹配的点数够多，才执行优化
        // (2)迭代30次（上限）优化
        //    1) 当前激光帧角点寻找局部map匹配点
        //       a.更新当前帧位姿，将当前帧角点坐标变换到map系下，在局部map中查找5个最近点，距离小于1m，且5个点构成直线（用距离中心点的协方差矩阵，特征值进行判断），则认为匹配上了
        //       b.计算当前帧角点到直线的距离、垂线的单位向量，存储为角点参数
        //    2) 当前激光帧平面点寻找局部map匹配点
        //       a.更新当前帧位姿，将当前帧平面点坐标变换到map系下，在局部map中查找5个最近点，距离小于1m，且5个点构成平面（最小二乘拟合平面），则认为匹配上了
        //       b.计算当前帧平面点到平面的距离、垂线的单位向量，存储为平面点参数
        //    3) 提取当前帧中与局部map匹配上了的角点、平面点，加入同一集合
        //    4) 对匹配特征点计算Jacobian矩阵，观测值为特征点到直线、平面的距离，构建高斯牛顿方程，迭代优化当前位姿，存transformTobeMapped
        scan2MapOptimization();

        // std::cout << "scan-to-map optimization trans(x,y,z): " << transformTobeMapped[3] << ", " << transformTobeMapped[4] << ", " << transformTobeMapped[5] << std::endl;

        // 5.设置当前帧为关键帧并执行因子图优化
        // (1)计算当前帧与前一帧位姿变换，如果变化太小，不设为关键帧，反之设为关键帧
        // (2)添加激光里程计因子、闭环因子、杆件路标因子
        // (3)执行因子图优化
        // (4)得到当前帧优化后位姿，位姿协方差
        // (5)添加cloudKeyPoses3D，cloudKeyPoses6D，更新transformTobeMapped，添加当前关键帧的角点、平面点集合
        saveKeyFramesAndFactor();

        // 6.更新因子图中所有变量节点的位姿，也就是所有历史关键帧的位姿，更新里程计轨迹
        correctPoses();

        // 7.发布激光里程计
        publishOdometry();

        // 8.发布里程计、点云、轨迹
        // (1)发布历史关键帧位姿集合
        // (2)发布局部map的降采样平面点集合
        // (3)发布历史帧（累加的）的角点、平面点降采样集合
        // (4)发布里程计轨迹
        publishFrames();
        ros::Time t2 = ros::Time::now();
        // std::cout << "mapping takes: " << (t2 - t1).toSec() * 1000 << "ms\n"
        //           << std::endl;
        //     laser_num++;
    }
}

sensor_msgs::PointCloud2 Map_Optimization::publishCloud(ros::Publisher *thisPub, pcl::PointCloud<PointType>::Ptr thisCloud, ros::Time thisStamp, std::string thisFrame)
{
    sensor_msgs::PointCloud2 tempCloud;
    pcl::toROSMsg(*thisCloud, tempCloud);
    tempCloud.header.stamp = thisStamp;
    tempCloud.header.frame_id = thisFrame;
    if (thisPub->getNumSubscribers() != 0)
        thisPub->publish(tempCloud);
    return tempCloud;
}

// 当前帧位姿初始化
void Map_Optimization::updateInitialGuess()
{

    Eigen::Vector3d odom_init_t = Eigen::Vector3d(odom_init_curr.pose.pose.position.x, odom_init_curr.pose.pose.position.y, odom_init_curr.pose.pose.position.z);
    Eigen::Quaterniond odom_init_q(odom_init_curr.pose.pose.orientation.w, odom_init_curr.pose.pose.orientation.x, odom_init_curr.pose.pose.orientation.y, odom_init_curr.pose.pose.orientation.z);

    // 这里都采用来自laserOdometry的初始激光里程计进行初始化
    if (cloudKeyPoses3D->points.empty())
    {

        Eigen::Vector3d eular = odom_init_q.matrix().eulerAngles(2, 1, 0); // Yaw, Pitch, Roll
        transformTobeMapped[3] = odom_init_t(0);                           // x
        transformTobeMapped[4] = odom_init_t(1);                           // y
        transformTobeMapped[5] = odom_init_t(2);                           // z
        transformTobeMapped[0] = eular(2);                                 // roll
        transformTobeMapped[1] = eular(1);                                 // pitch
        transformTobeMapped[2] = eular(0);                                 // yaw

        tf_odom_last.setOrigin(tf::Vector3(odom_init_t(0), odom_init_t(1), odom_init_t(2)));
        tf_odom_last.setRotation(tf::Quaternion(odom_init_q.x(), odom_init_q.y(), odom_init_q.z(), odom_init_q.w()));
        return;
    }

    tf::Transform tf_odom_curr;
    tf_odom_curr.setOrigin(tf::Vector3(odom_init_t(0), odom_init_t(1), odom_init_t(2)));
    tf_odom_curr.setRotation(tf::Quaternion(odom_init_q.x(), odom_init_q.y(), odom_init_q.z(), odom_init_q.w()));

    tf::Transform tf_odom_increment = tf_odom_last.inverse() * tf_odom_curr;

    tf::Transform tf_odom_mapped_last;
    tf_odom_mapped_last.setOrigin(tf::Vector3(transformTobeMapped[3], transformTobeMapped[4], transformTobeMapped[5]));
    tf::Quaternion q;
    q.setRPY(transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]);
    tf_odom_mapped_last.setRotation(q);

    tf::Transform tf_odom_mapped_curr = tf_odom_mapped_last * tf_odom_increment;
    Eigen::Quaterniond q_curr(tf_odom_mapped_curr.getRotation().w(), tf_odom_mapped_curr.getRotation().x(),
                              tf_odom_mapped_curr.getRotation().y(), tf_odom_mapped_curr.getRotation().z());

    Eigen::Vector3d eulerAngle = q_curr.matrix().eulerAngles(2, 1, 0);
    transformTobeMapped[0] = eulerAngle(2);                       // roll
    transformTobeMapped[1] = eulerAngle(1);                       // pitch
    transformTobeMapped[2] = eulerAngle(0);                       // yaw
    transformTobeMapped[3] = tf_odom_mapped_curr.getOrigin().x(); // x
    transformTobeMapped[4] = tf_odom_mapped_curr.getOrigin().y(); // y
    transformTobeMapped[5] = tf_odom_mapped_curr.getOrigin().z(); // z

    tf_odom_last = tf_odom_curr;

    // // save current transformation before any processing
    // // 前一帧的位姿，注：这里指lidar的位姿，后面都简写成位姿
    // incrementalOdometryAffineFront = trans2Affine3f(transformTobeMapped);
    // // 前一帧的初始化姿态角（来自原始imu数据），用于估计第一帧的位姿（旋转部分）
    // static Eigen::Affine3f lastImuTransformation;
    // // initialization
    // // 如果关键帧集合为空，继续进行初始化
    // if (cloudKeyPoses3D->points.empty())
    // {
    //     // 当前帧位姿的旋转部分，用激光帧信息中的RPY（来自imu原始数据）初始化
    //     transformTobeMapped[0] = cloudInfo.imuRollInit;
    //     transformTobeMapped[1] = cloudInfo.imuPitchInit;
    //     transformTobeMapped[2] = cloudInfo.imuYawInit;

    //     if (!useImuHeadingInitialization)
    //         transformTobeMapped[2] = 0;

    //     lastImuTransformation = pcl::getTransformation(0, 0, 0, cloudInfo.imuRollInit, cloudInfo.imuPitchInit, cloudInfo.imuYawInit); // save imu before return;
    //     return;
    // }

    // // use imu pre-integration estimation for pose guess
    // // 用当前帧和前一帧对应的imu里程计计算相对位姿变换，再用前一帧的位姿与相对变换，计算当前帧的位姿，存transformTobeMapped
    // static bool lastImuPreTransAvailable = false;
    // static Eigen::Affine3f lastImuPreTransformation;
    // //odomAvailable和imuAvailable均来源于imageProjection.cpp中赋值，
    // //imuAvailable是遍历激光帧前后起止时刻0.01s之内的imu数据，
    // //如果都没有那就是false，因为imu频率一般比激光帧快，因此这里应该是都有的。
    // //odomAvailable同理，是监听imu里程计的位姿，如果没有紧挨着激光帧的imu里程计数据，那么就是false；
    // //这俩应该一般都有
    // if (cloudInfo.odomAvailable == true)
    // {
    //     // cloudInfo来自featureExtraction.cpp发布的lio_sam/feature/cloud_info,
    //     //而其中的initialGuessX等信息本质上来源于ImageProjection.cpp发布的deskew/cloud_info信息，
    //     //而deskew/cloud_info中的initialGuessX则来源于ImageProjection.cpp中的回调函数odometryHandler，
    //     //odometryHandler订阅的是imuPreintegration.cpp发布的odometry/imu_incremental话题，
    //     //该话题发布的xyz是imu在前一帧雷达基础上的增量位姿
    //     //纠正一个观点：增量位姿，指的绝不是预积分位姿！！是在前一帧雷达的基础上(包括该基础!!)的（基础不是0）的位姿
    //     //当前帧的初始估计位姿（来自imu里程计），后面用来计算增量位姿变换
    //     Eigen::Affine3f transBack = pcl::getTransformation(cloudInfo.initialGuessX,    cloudInfo.initialGuessY,     cloudInfo.initialGuessZ,
    //                                                        cloudInfo.initialGuessRoll, cloudInfo.initialGuessPitch, cloudInfo.initialGuessYaw);
    //     if (lastImuPreTransAvailable == false)
    //     {
    //         // 赋值给前一帧
    //         //lastImuPreTransAvailable是一个静态变量，初始被设置为false,之后就变成了true
    //         //也就是说这段只调用一次，就是初始时，把imu位姿赋值给lastImuPreTransformation
    //         lastImuPreTransformation = transBack;
    //         lastImuPreTransAvailable = true;
    //     } else {
    //         // 当前帧相对于前一帧的位姿变换，imu里程计计算得到
    //         //lastImuPreTransformation就是上一帧激光时刻的imu位姿,transBack是这一帧时刻的imu位姿
    //         //求完逆相乘以后才是增量，绝不可把imu_incremental发布的当成是两激光间的增量
    //         Eigen::Affine3f transIncre = lastImuPreTransformation.inverse() * transBack;

    //         // 前一帧的位姿
    //         Eigen::Affine3f transTobe = trans2Affine3f(transformTobeMapped);

    //         // 当前帧的位姿
    //         Eigen::Affine3f transFinal = transTobe * transIncre;
    //         //将transFinal传入，结果输出至transformTobeMapped中
    //         pcl::getTranslationAndEulerAngles(transFinal, transformTobeMapped[3], transformTobeMapped[4], transformTobeMapped[5],
    //                                                       transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]);
    //         // 当前帧初始位姿赋值作为前一帧
    //         lastImuPreTransformation = transBack;

    //         lastImuTransformation = pcl::getTransformation(0, 0, 0, cloudInfo.imuRollInit, cloudInfo.imuPitchInit, cloudInfo.imuYawInit); // save imu before return;
    //         return;
    //     }
    // }

    // // use imu incremental estimation for pose guess (only rotation)
    // // 只在第一帧调用（注意上面的return），用imu数据初始化当前帧位姿，仅初始化旋转部分
    // if (cloudInfo.imuAvailable == true)
    // {
    //     //注：这一时刻的transBack和之前if (cloudInfo.odomAvailable == true)内部的transBack不同，
    //     //之前获得的是initialGuessRoll等，但是在这里是imuRollInit，它来源于imageProjection中的imuQueue，直接存储原始imu数据的。
    //     //那么对于第一帧数据，目前的lastImuTransformation是initialGuessX等，即imu里程计的数据；
    //     //而transBack是imuRollInit是imu的瞬时原始数据roll、pitch和yaw三个角。
    //     //那么imuRollInit和initialGuessRoll这两者有啥区别呢？
    //     //imuRollInit是imu姿态角，在imageProjection中一收到，就马上赋值给它要发布的cloud_info，
    //     //而initialGuessRoll是imu里程计发布的姿态角。
    //     //直观上来说，imu原始数据收到速度是应该快于imu里程计的数据的，因此感觉二者之间应该有一个增量，
    //     //那么lastImuTransformation.inverse() * transBack算出增量，增量再和原先的transformTobeMapped计算,
    //     //结果依旧以transformTobeMapped来保存
    //     //感觉这里写的非常奇怪
    //     Eigen::Affine3f transBack = pcl::getTransformation(0, 0, 0, cloudInfo.imuRollInit, cloudInfo.imuPitchInit, cloudInfo.imuYawInit);
    //     Eigen::Affine3f transIncre = lastImuTransformation.inverse() * transBack;

    //     Eigen::Affine3f transTobe = trans2Affine3f(transformTobeMapped);
    //     Eigen::Affine3f transFinal = transTobe * transIncre;
    //     pcl::getTranslationAndEulerAngles(transFinal, transformTobeMapped[3], transformTobeMapped[4], transformTobeMapped[5],
    //                                                   transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]);

    //     lastImuTransformation = pcl::getTransformation(0, 0, 0, cloudInfo.imuRollInit, cloudInfo.imuPitchInit, cloudInfo.imuYawInit); // save imu before return;
    //     return;
    // }
}

void Map_Optimization::extractSurroundingKeyFrames()
{
    if (cloudKeyPoses3D->points.empty() == true)
    {
        return;
    }

    extractNearby();
}

void Map_Optimization::extractNearby()
{
    pcl::PointCloud<PointType>::Ptr surroundingKeyPoses(new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr surroundingKeyPosesDS(new pcl::PointCloud<PointType>());
    std::vector<int> pointSearchInd;
    std::vector<float> pointSearchSqDis;

    // extract all the nearby key poses and downsample them
    // kdtree的输入，全局关键帧位姿集合（历史所有关键帧集合）
    kdtreeSurroundingKeyPoses->setInputCloud(cloudKeyPoses3D); // create kd-tree
    // 创建Kd树然后搜索  半径在配置文件中
    // 指定半径范围查找近邻
    // 球状固定距离半径近邻搜索
    //  surroundingKeyframeSearchRadius是搜索半径，pointSearchInd应该是返回的index，pointSearchSqDis应该是依次距离中心点的距离
    kdtreeSurroundingKeyPoses->radiusSearch(cloudKeyPoses3D->back(), (double)surroundingKeyframeSearchRadius, pointSearchInd, pointSearchSqDis);

    for (int i = 0; i < (int)pointSearchInd.size(); ++i)
    {
        int id = pointSearchInd[i];

        // 保存附近关键帧,加入相邻关键帧位姿集合中
        surroundingKeyPoses->push_back(cloudKeyPoses3D->points[id]);
    }
    // 降采样
    // 把相邻关键帧位姿集合，进行下采样，滤波后存入surroundingKeyPosesDS
    downSizeFilterSurroundingKeyPoses.setInputCloud(surroundingKeyPoses);
    downSizeFilterSurroundingKeyPoses.filter(*surroundingKeyPosesDS);
    for (auto &pt : surroundingKeyPosesDS->points)
    {
        // k近邻搜索,找出最近的k个节点（这里是1）
        kdtreeSurroundingKeyPoses->nearestKSearch(pt, 1, pointSearchInd, pointSearchSqDis);
        // 把强度替换掉，注意是从原始关键帧数据中替换的,相当于是把下采样以后的点的强度，换成是原始点强度
        // 注意，这里的intensity应该不是强度，因为在函数saveKeyFramesAndFactor中:
        //  thisPose3D.intensity = cloudKeyPoses3D->size();
        // 就是索引，只不过这里借用intensity结构来存放
        pt.intensity = cloudKeyPoses3D->points[pointSearchInd[0]].intensity;
    }

    // also extract some latest key frames in case the robot rotates in one position
    // 提取了一些最新的关键帧，以防机器人在一个位置原地旋转
    int numPoses = cloudKeyPoses3D->size();
    // 把10s内的关键帧也加到surroundingKeyPosesDS中,注意是“也”，原先已经装了下采样的位姿(位置)
    for (int i = numPoses - 1; i >= 0; --i)
    {
        if (timeLaserInfoCur - cloudKeyPoses6D->points[i].time < 10.0)
            surroundingKeyPosesDS->push_back(cloudKeyPoses3D->points[i]);
        else
            break;
    }
    // 对降采样后的点云进行提取出边缘点和平面点对应的localmap
    extractCloud(surroundingKeyPosesDS);
}

// 将相邻关键帧集合对应的角点、平面点，加入到局部map中，作为scan-to-map匹配的局部点云地图
void Map_Optimization::extractCloud(pcl::PointCloud<PointType>::Ptr cloudToExtract)
{
    // fuse the map
    laserCloudCornerFromMap->clear();
    laserCloudSurfFromMap->clear();
    // 遍历当前帧（实际是取最近的一个关键帧来找它相邻的关键帧集合）时空维度上相邻的关键帧集合
    for (int i = 0; i < (int)cloudToExtract->size(); ++i)
    {
        // 距离超过阈值，丢弃
        if (pointDistance(cloudToExtract->points[i], cloudKeyPoses3D->back()) > surroundingKeyframeSearchRadius)
            continue;

        // 相邻关键帧索引
        int thisKeyInd = (int)cloudToExtract->points[i].intensity;
        if (laserCloudMapContainer.find(thisKeyInd) != laserCloudMapContainer.end())
        {
            // transformed cloud available
            *laserCloudCornerFromMap += laserCloudMapContainer[thisKeyInd].first;
            *laserCloudSurfFromMap += laserCloudMapContainer[thisKeyInd].second;
        }
        else
        {
            // transformed cloud not available
            // 相邻关键帧对应的角点、平面点云，通过6D位姿变换到世界坐标系下
            // transformPointCloud输入的两个形参，分别为点云和变换，返回变换位姿后的点
            pcl::PointCloud<PointType> laserCloudCornerTemp = *transformPointCloud(cornerCloudKeyFrames[thisKeyInd], &cloudKeyPoses6D->points[thisKeyInd]);
            pcl::PointCloud<PointType> laserCloudSurfTemp = *transformPointCloud(surfCloudKeyFrames[thisKeyInd], &cloudKeyPoses6D->points[thisKeyInd]);
            // 加入局部map
            *laserCloudCornerFromMap += laserCloudCornerTemp;
            *laserCloudSurfFromMap += laserCloudSurfTemp;
            laserCloudMapContainer[thisKeyInd] = make_pair(laserCloudCornerTemp, laserCloudSurfTemp);
        }
    }

    // Downsample the surrounding corner key frames (or map)
    // 降采样局部角点map
    downSizeFilterCorner.setInputCloud(laserCloudCornerFromMap);
    downSizeFilterCorner.filter(*laserCloudCornerFromMapDS);
    laserCloudCornerFromMapDSNum = laserCloudCornerFromMapDS->size();
    // Downsample the surrounding surf key frames (or map)
    // 降采样局部平面点map
    downSizeFilterSurf.setInputCloud(laserCloudSurfFromMap);
    downSizeFilterSurf.filter(*laserCloudSurfFromMapDS);
    laserCloudSurfFromMapDSNum = laserCloudSurfFromMapDS->size();

    // clear map cache if too large
    // 太大了，清空一下内存
    if (laserCloudMapContainer.size() > 1000)
        laserCloudMapContainer.clear();
}

// 提取局部角点、平面点云集合，加入局部map
void Map_Optimization::extractForLoopClosure()
{
    // 1、对最近的一帧关键帧，搜索时空维度上相邻的关键帧集合，降采样一下
    // 2、对关键帧集合中的每一帧，提取对应的角点、平面点，加入局部map中
    pcl::PointCloud<PointType>::Ptr cloudToExtract(new pcl::PointCloud<PointType>());
    int numPoses = cloudKeyPoses3D->size();
    for (int i = numPoses - 1; i >= 0; --i)
    {
        if ((int)cloudToExtract->size() <= surroundingKeyframeSize)
            cloudToExtract->push_back(cloudKeyPoses3D->points[i]);
        else
            break;
    }

    extractCloud(cloudToExtract);
}

// 对激光帧角点和平面点集合进行降采样
void Map_Optimization::downsampleCurrentScan()
{

    // Downsample cloud from current scan
    // 对当前帧点云降采样  刚刚完成了周围关键帧的降采样
    // 大量的降采样工作无非是为了使点云稀疏化 加快匹配以及实时性要求
    laserCloudCornerLastDS->clear();
    downSizeFilterCorner.setInputCloud(laserCloudCornerLast);
    downSizeFilterCorner.filter(*laserCloudCornerLastDS);
    laserCloudCornerLastDSNum = laserCloudCornerLastDS->size();

    laserCloudSurfLastDS->clear();
    downSizeFilterSurf.setInputCloud(laserCloudSurfLast);
    downSizeFilterSurf.filter(*laserCloudSurfLastDS);
    laserCloudSurfLastDSNum = laserCloudSurfLastDS->size();
}

// scan-to-map优化当前帧位姿
void Map_Optimization::scan2MapOptimization()
{
    // 根据现有地图与最新点云数据进行配准从而更新机器人精确位姿与融合建图，
    // 它分为角点优化、平面点优化、配准与更新等部分。
    // 优化的过程与里程计的计算类似，是通过计算点到直线或平面的距离，构建优化公式再用LM法求解。
    if (cloudKeyPoses3D->points.empty())
        return;

    if (laserCloudCornerLastDSNum > edgeFeatureMinValidNum && laserCloudSurfLastDSNum > surfFeatureMinValidNum)
    {
        // 构建kdtree
        kdtreeCornerFromMap->setInputCloud(laserCloudCornerFromMapDS);
        kdtreeSurfFromMap->setInputCloud(laserCloudSurfFromMapDS);
        // 迭代30次
        int iterCount;
        for (iterCount = 0; iterCount < 30; iterCount++)
        {
            laserCloudOri->clear();
            coeffSel->clear();
            // 边缘点匹配优化
            cornerOptimization();
            // 平面点匹配优化
            surfOptimization();
            // 组合优化多项式系数
            combineOptimizationCoeffs();

            if (LMOptimization(iterCount) == true)
                break;
        }

        transformUpdate();
        // visualize_factor(surf_factor, timeLaserInfoStamp, 250, pub_select_surf);
        // visualize_factor(corner_factor, timeLaserInfoStamp, 100, pub_select_corner);
        // std::cout << "iter: " << iterCount << std::endl;
    }
    else
    {
        ROS_WARN("Not enough features! Only %d edge and %d planar features available.", laserCloudCornerLastDSNum, laserCloudSurfLastDSNum);
    }
}

// 将当前位姿transformTobeMapped存储为仿射矩阵
void Map_Optimization::updatePointAssociateToMap()
{
    transPointAssociateToMap = trans2Affine3f(transformTobeMapped);
}

// 边缘点匹配优化
void Map_Optimization::cornerOptimization()
{
    corner_factor.clear();
    // 实现transformTobeMapped的仿射矩阵形式转换 下面调用的函数就一行就不展开了  工具类函数
    //  把结果存入transPointAssociateToMap中
    updatePointAssociateToMap();

#pragma omp parallel for num_threads(numberOfCores)
    for (int i = 0; i < laserCloudCornerLastDSNum; i++)
    {
        PointType pointOri, pointSel, coeff;
        std::vector<int> pointSearchInd;
        std::vector<float> pointSearchSqDis;

        pointOri = laserCloudCornerLastDS->points[i];
        // 第i帧的点转换到第一帧坐标系下
        // 这里就调用了第一步中updatePointAssociateToMap中实现的transPointAssociateToMap，
        // 然后利用这个变量，把pointOri的点转换到pointSel下，pointSel作为输出
        pointAssociateToMap(&pointOri, &pointSel);
        // kd树的最近搜索
        kdtreeCornerFromMap->nearestKSearch(pointSel, 5, pointSearchInd, pointSearchSqDis);

        cv::Mat matA1(3, 3, CV_32F, cv::Scalar::all(0));
        cv::Mat matD1(1, 3, CV_32F, cv::Scalar::all(0));
        cv::Mat matV1(3, 3, CV_32F, cv::Scalar::all(0));

        if (pointSearchSqDis[4] < 1.0)
        {
            float cx = 0, cy = 0, cz = 0;
            // 先求5个样本的平均值
            for (int j = 0; j < 5; j++)
            {
                cx += laserCloudCornerFromMapDS->points[pointSearchInd[j]].x;
                cy += laserCloudCornerFromMapDS->points[pointSearchInd[j]].y;
                cz += laserCloudCornerFromMapDS->points[pointSearchInd[j]].z;
            }
            cx /= 5;
            cy /= 5;
            cz /= 5;

            // 下面求矩阵matA1=[ax,ay,az]^t*[ax,ay,az]
            // 更准确地说应该是在求协方差matA1
            float a11 = 0, a12 = 0, a13 = 0, a22 = 0, a23 = 0, a33 = 0;
            for (int j = 0; j < 5; j++)
            {
                float ax = laserCloudCornerFromMapDS->points[pointSearchInd[j]].x - cx;
                float ay = laserCloudCornerFromMapDS->points[pointSearchInd[j]].y - cy;
                float az = laserCloudCornerFromMapDS->points[pointSearchInd[j]].z - cz;

                a11 += ax * ax;
                a12 += ax * ay;
                a13 += ax * az;
                a22 += ay * ay;
                a23 += ay * az;
                a33 += az * az;
            }
            a11 /= 5;
            a12 /= 5;
            a13 /= 5;
            a22 /= 5;
            a23 /= 5;
            a33 /= 5;

            matA1.at<float>(0, 0) = a11;
            matA1.at<float>(0, 1) = a12;
            matA1.at<float>(0, 2) = a13;
            matA1.at<float>(1, 0) = a12;
            matA1.at<float>(1, 1) = a22;
            matA1.at<float>(1, 2) = a23;
            matA1.at<float>(2, 0) = a13;
            matA1.at<float>(2, 1) = a23;
            matA1.at<float>(2, 2) = a33;

            // 求正交阵的特征值和特征向量
            // 特征值：matD1，特征向量：matV1中  对应于LOAM论文里雷达建图 特征值与特征向量那块
            cv::eigen(matA1, matD1, matV1);
            // 边缘：与较大特征值相对应的特征向量代表边缘线的方向（一大两小，大方向）
            // 以下这一大块是在计算点到边缘的距离，最后通过系数s来判断是否距离很近
            // 如果距离很近就认为这个点在边缘上，需要放到laserCloudOri中
            // 如果最大的特征值相比次大特征值，大很多，认为构成了线，角点是合格的
            if (matD1.at<float>(0, 0) > 3 * matD1.at<float>(0, 1))
            {
                // 当前帧角点坐标（map系下）
                float x0 = pointSel.x;
                float y0 = pointSel.y;
                float z0 = pointSel.z;
                float x1 = cx + 0.1 * matV1.at<float>(0, 0);
                float y1 = cy + 0.1 * matV1.at<float>(0, 1);
                float z1 = cz + 0.1 * matV1.at<float>(0, 2);
                float x2 = cx - 0.1 * matV1.at<float>(0, 0);
                float y2 = cy - 0.1 * matV1.at<float>(0, 1);
                float z2 = cz - 0.1 * matV1.at<float>(0, 2);

                // 局部map对应中心角点，沿着特征向量（直线方向）方向，前后各取一个点
                float a012 = sqrt(((x0 - x1) * (y0 - y2) - (x0 - x2) * (y0 - y1)) * ((x0 - x1) * (y0 - y2) - (x0 - x2) * (y0 - y1)) + ((x0 - x1) * (z0 - z2) - (x0 - x2) * (z0 - z1)) * ((x0 - x1) * (z0 - z2) - (x0 - x2) * (z0 - z1)) + ((y0 - y1) * (z0 - z2) - (y0 - y2) * (z0 - z1)) * ((y0 - y1) * (z0 - z2) - (y0 - y2) * (z0 - z1)));
                // line_12，底边边长
                float l12 = sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2) + (z1 - z2) * (z1 - z2));
                // 两次叉积，得到点到直线的垂线段单位向量，x分量，下面同理
                // 求叉乘结果[la',lb',lc']=[(x1-x2),(y1-y2),(z1-z2)]x[XXX,YYY,ZZZ]
                // [la,lb,lc]=[la',lb',lc']/a012/l12
                // 得到底边上的高的方向向量[la,lb,lc]
                // LLL=[la,lb,lc]是V1[0]这条高上的单位法向量。||LLL||=1；

                // 如不理解则看图：
                //        A
                //   B        C
                // 这里ABxAC，代表垂直于ABC面的法向量，其模长为平行四边形面积
                // 因此BCx(ABxAC),代表了BC和（ABC平面的法向量）的叉乘，那么其实这个向量就是A到BC的垂线的方向向量
                // 那么(ABxAC)/|ABxAC|,代表着ABC平面的单位法向量
                // BCxABC平面单位法向量，即为一个长度为|BC|的（A到BC垂线的方向向量），因此再除以|BC|，得到A到BC垂线的单位方向向量

                float la = ((y1 - y2) * ((x0 - x1) * (y0 - y2) - (x0 - x2) * (y0 - y1)) + (z1 - z2) * ((x0 - x1) * (z0 - z2) - (x0 - x2) * (z0 - z1))) / a012 / l12;

                float lb = -((x1 - x2) * ((x0 - x1) * (y0 - y2) - (x0 - x2) * (y0 - y1)) - (z1 - z2) * ((y0 - y1) * (z0 - z2) - (y0 - y2) * (z0 - z1))) / a012 / l12;

                float lc = -((x1 - x2) * ((x0 - x1) * (z0 - z2) - (x0 - x2) * (z0 - z1)) + (y1 - y2) * ((y0 - y1) * (z0 - z2) - (y0 - y2) * (z0 - z1))) / a012 / l12;

                // 三角形的高，也就是点到直线距离
                // 计算点pointSel到直线的距离
                // 这里需要特别说明的是ld2代表的是点pointSel到过点[cx,cy,cz]的方向向量直线的距离
                float ld2 = a012 / l12;
                // 下面涉及到一个鲁棒核函数，作者简单地设计了这个核函数。
                // 距离越大，s越小，是个距离惩罚因子（权重
                float s = 1 - 0.9 * fabs(ld2);
                // coeff代表系数的意思
                // coff用于保存距离的方向向量
                coeff.x = s * la;
                coeff.y = s * lb;
                coeff.z = s * lc;
                // intensity本质上构成了一个核函数，ld2越接近于1，增长越慢
                // intensity=(1-0.9*ld2)*ld2=ld2-0.9*ld2*ld2
                coeff.intensity = s * ld2;
                // 程序末尾根据s的值来判断是否将点云点放入点云集合laserCloudOri以及coeffSel中。
                // 所以就应该认为这个点是边缘点
                // s>0.1 也就是要求点到直线的距离ld2要小于1m
                // s越大说明ld2越小(离边缘线越近)，这样就说明点pointOri在直线上
                if (s > 0.1)
                {
                    laserCloudOriCornerVec[i] = pointOri;
                    coeffSelCornerVec[i] = coeff;
                    laserCloudOriCornerFlag[i] = true;

                    // FactorParam factor;
                    // factor.col(0) << cx, cy, cz;
                    // factor.col(1) << matV1.at<float>(0, 0), matV1.at<float>(0, 1), matV1.at<float>(0, 2);
                    // factor.col(2) << pointOri.x, pointOri.y, pointOri.z;
                    // factor.col(3)[0] = s * ld2;

                    // corner_factor.push_back(factor);
                }
            }
        }
    }
}

// 平面点优化
void Map_Optimization::surfOptimization()
{
    updatePointAssociateToMap();

#pragma omp parallel for num_threads(numberOfCores)
    for (int i = 0; i < laserCloudSurfLastDSNum; i++)
    {
        PointType pointOri, pointSel, coeff;
        std::vector<int> pointSearchInd;
        std::vector<float> pointSearchSqDis;

        pointOri = laserCloudSurfLastDS->points[i];
        pointAssociateToMap(&pointOri, &pointSel);
        kdtreeSurfFromMap->nearestKSearch(pointSel, 5, pointSearchInd, pointSearchSqDis);

        Eigen::Matrix<float, 5, 3> matA0;
        Eigen::Matrix<float, 5, 1> matB0;
        Eigen::Vector3f matX0;

        matA0.setZero();
        matB0.fill(-1);
        matX0.setZero();

        if (pointSearchSqDis[4] < 1.0)
        {
            for (int j = 0; j < 5; j++)
            {
                matA0(j, 0) = laserCloudSurfFromMapDS->points[pointSearchInd[j]].x;
                matA0(j, 1) = laserCloudSurfFromMapDS->points[pointSearchInd[j]].y;
                matA0(j, 2) = laserCloudSurfFromMapDS->points[pointSearchInd[j]].z;
            }

            matX0 = matA0.colPivHouseholderQr().solve(matB0);

            float pa = matX0(0, 0);
            float pb = matX0(1, 0);
            float pc = matX0(2, 0);
            float pd = 1;

            float ps = sqrt(pa * pa + pb * pb + pc * pc);
            pa /= ps;
            pb /= ps;
            pc /= ps;
            pd /= ps;

            bool planeValid = true;
            for (int j = 0; j < 5; j++)
            {
                if (fabs(pa * laserCloudSurfFromMapDS->points[pointSearchInd[j]].x +
                         pb * laserCloudSurfFromMapDS->points[pointSearchInd[j]].y +
                         pc * laserCloudSurfFromMapDS->points[pointSearchInd[j]].z + pd) > 0.2)
                {
                    planeValid = false;
                    break;
                }
            }

            if (planeValid)
            {
                float pd2 = pa * pointSel.x + pb * pointSel.y + pc * pointSel.z + pd;

                float s = 1 - 0.9 * fabs(pd2) / sqrt(sqrt(pointSel.x * pointSel.x + pointSel.y * pointSel.y + pointSel.z * pointSel.z));

                coeff.x = s * pa;
                coeff.y = s * pb;
                coeff.z = s * pc;
                coeff.intensity = s * pd2;

                if (s > 0.1)
                {
                    laserCloudOriSurfVec[i] = pointOri;
                    coeffSelSurfVec[i] = coeff;
                    laserCloudOriSurfFlag[i] = true;
                }
            }
        }
    }
}

// 组合优化多项式系数
void Map_Optimization::combineOptimizationCoeffs()
{
    int num_edge = 0, num_surf = 0;
    // combine corner coeffs
    // 遍历当前帧角点集合，提取出与局部map匹配上了的角点
    for (int i = 0; i < laserCloudCornerLastDSNum; ++i)
    {
        if (laserCloudOriCornerFlag[i] == true)
        {
            laserCloudOri->push_back(laserCloudOriCornerVec[i]);
            coeffSel->push_back(coeffSelCornerVec[i]);
            num_edge++;
        }
    }
    // combine surf coeffs
    // 遍历当前帧平面点集合，提取出与局部map匹配上了的平面点
    for (int i = 0; i < laserCloudSurfLastDSNum; ++i)
    {
        if (laserCloudOriSurfFlag[i] == true)
        {
            laserCloudOri->push_back(laserCloudOriSurfVec[i]);
            coeffSel->push_back(coeffSelSurfVec[i]);
            num_surf++;
        }
    }
    // std::cout << "select " << num_edge << " corner and " << num_surf << " surf points" << std::endl;
    // reset flag for next iteration
    // 清空标记
    std::fill(laserCloudOriCornerFlag.begin(), laserCloudOriCornerFlag.end(), false);
    std::fill(laserCloudOriSurfFlag.begin(), laserCloudOriSurfFlag.end(), false);
}

inline Eigen::Matrix<float, 6, 6> Map_Optimization::transMatrix(Eigen::Matrix<float, 6, 6> &M)
{
    Eigen::Matrix<float, 6, 6> out;
    out.setZero();
    for (int i = 5; i >= 0; i--)
        for (int j = 0; j <= 5; j++)
            out(5 - i, j) = M(j, i);
    return out;
}

// #define USE_CV_MAT_ 1

// scan-to-map优化
bool Map_Optimization::LMOptimization(int iterCount)
{
    // 对匹配特征点计算Jacobian矩阵，观测值为特征点到直线、平面的距离，构建高斯牛顿方程，迭代优化当前位姿，存transformTobeMapped

    // 由于LOAM里雷达的特殊坐标系 所以这里也转了一次
    // This optimization is from the original loam_velodyne by Ji Zhang, need to cope with coordinate transformation
    // lidar <- camera      ---     camera <- lidar
    // x = z                ---     x = y
    // y = x                ---     y = z
    // z = y                ---     z = x
    // roll = yaw           ---     roll = pitch
    // pitch = roll         ---     pitch = yaw
    // yaw = pitch          ---     yaw = roll

    // lidar -> camera
    float srx = sin(transformTobeMapped[1]);
    float crx = cos(transformTobeMapped[1]);
    float sry = sin(transformTobeMapped[2]);
    float cry = cos(transformTobeMapped[2]);
    float srz = sin(transformTobeMapped[0]);
    float crz = cos(transformTobeMapped[0]);

    // 当前帧匹配特征点数太少
    int laserCloudSelNum = laserCloudOri->size();
    if (laserCloudSelNum < 50)
    {
        return false;
    }

    Eigen::Matrix<float, Eigen::Dynamic, 6> mat_A(laserCloudSelNum, 6);
    Eigen::Matrix<float, 6, Eigen::Dynamic> mat_A_transpose(6, laserCloudSelNum);
    Eigen::Matrix<float, 6, 6> mat_A_transpose_A;
    Eigen::VectorXf mat_B(laserCloudSelNum);
    Eigen::Matrix<float, 6, 1> mat_A_transpose_B;
    Eigen::Matrix<float, 6, 1> mat_X;
    Eigen::Matrix<float, 6, 6> mat_P_;
    mat_P_.setZero();

    PointType pointOri, coeff;
    // 遍历匹配特征点，构建Jacobian矩阵
    for (int i = 0; i < laserCloudSelNum; i++)
    {
        // lidar -> camera
        pointOri.x = laserCloudOri->points[i].y;
        pointOri.y = laserCloudOri->points[i].z;
        pointOri.z = laserCloudOri->points[i].x;
        // lidar -> camera
        coeff.x = coeffSel->points[i].y;
        coeff.y = coeffSel->points[i].z;
        coeff.z = coeffSel->points[i].x;
        coeff.intensity = coeffSel->points[i].intensity;
        // in camera
        // https://wykxwyc.github.io/2019/08/01/The-Math-Formula-in-LeGO-LOAM/
        // 求雅克比矩阵中的元素，距离d对roll角度的偏导量即d(d)/d(roll)
        // 各种cos sin的是旋转矩阵对roll求导，pointOri.x是点的坐标，coeff.x等是距离到局部点的偏导，也就是法向量（建议看链接）
        // 注意：链接当中的R0-5公式中，ex和ey是反的
        // 另一个链接https://blog.csdn.net/weixin_37835423/article/details/111587379#commentBox当中写的更好
        float arx = (crx * sry * srz * pointOri.x + crx * crz * sry * pointOri.y - srx * sry * pointOri.z) * coeff.x + (-srx * srz * pointOri.x - crz * srx * pointOri.y - crx * pointOri.z) * coeff.y + (crx * cry * srz * pointOri.x + crx * cry * crz * pointOri.y - cry * srx * pointOri.z) * coeff.z;
        // 同上，求解的是对pitch的偏导量
        float ary = ((cry * srx * srz - crz * sry) * pointOri.x + (sry * srz + cry * crz * srx) * pointOri.y + crx * cry * pointOri.z) * coeff.x + ((-cry * crz - srx * sry * srz) * pointOri.x + (cry * srz - crz * srx * sry) * pointOri.y - crx * sry * pointOri.z) * coeff.z;

        float arz = ((crz * srx * sry - cry * srz) * pointOri.x + (-cry * crz - srx * sry * srz) * pointOri.y) * coeff.x + (crx * crz * pointOri.x - crx * srz * pointOri.y) * coeff.y + ((sry * srz + cry * crz * srx) * pointOri.x + (crz * sry - cry * srx * srz) * pointOri.y) * coeff.z;

        // lidar -> camera
        mat_A(i, 0) = arz;
        mat_A(i, 1) = arx;
        mat_A(i, 2) = ary;
        mat_A(i, 3) = coeff.z;
        mat_A(i, 4) = coeff.x;
        mat_A(i, 5) = coeff.y;
        mat_B(i, 0) = -coeff.intensity;
    }
    mat_A_transpose = mat_A.transpose();
    mat_A_transpose_A = mat_A_transpose * mat_A;
    mat_A_transpose_B = mat_A_transpose * mat_B;
    mat_X = mat_A_transpose_A.colPivHouseholderQr().solve(mat_A_transpose_B);

    if (iterCount == 0)
    {
        Eigen::Matrix<float, 1, 6> mat_E;
        Eigen::Matrix<float, 6, 6> mat_V;
        Eigen::Matrix<float, 6, 6> mat_V_temp;

        Eigen::SelfAdjointEigenSolver<Eigen::Matrix<float, 6, 6>> esolver(mat_A_transpose_A);
        mat_E = esolver.eigenvalues().real(); //  eigenvalue from small to large here
        mat_V = esolver.eigenvectors().real();

        mat_V_temp = mat_V;

        isDegenerate = false;
        float eignThre[6] = {100, 100, 100, 100, 100, 100};
        int i;
        for (int i = 50; i <= 5; i++)
        {
            if (mat_E(0, i) < eignThre[i])
            {
                for (int j = 0; j < 6; j++)
                {
                    mat_V_temp(j, i) = 0; //  cols
                }
                isDegenerate = true;
            }
            else
            {
                break;
            }
        }
        if (isDegenerate)
            std::cout << "Degenerate here at " << i << std::endl;
        // matP = matV.inv() * matV2;
        mat_P_ = transMatrix(mat_V).inverse() * transMatrix(mat_V_temp);
    }

    if (isDegenerate)
    {
        Eigen::Matrix<float, 6, 1> matX2(mat_X);
        matX2 = mat_X;
        mat_X = mat_P_ * matX2;
    }

    // std::cout << "mat_X(delta r,p,y,x,y,z):\n" << mat_X(0, 0) << ", " << mat_X(1, 0) << ", " << mat_X(2, 0) << ", " << mat_X(3, 0) << ", " << mat_X(4, 0)
    //         << ", "  << mat_X(5, 0)  << std::endl;

    transformTobeMapped[0] += mat_X(0, 0);
    transformTobeMapped[1] += mat_X(1, 0);
    transformTobeMapped[2] += mat_X(2, 0);
    transformTobeMapped[3] += mat_X(3, 0);
    transformTobeMapped[4] += mat_X(4, 0);
    transformTobeMapped[5] += mat_X(5, 0);

    float deltaR = sqrt(
        pow(pcl::rad2deg(mat_X(0, 0)), 2) +
        pow(pcl::rad2deg(mat_X(1, 0)), 2) +
        pow(pcl::rad2deg(mat_X(2, 0)), 2));
    float deltaT = sqrt(
        pow(mat_X(3, 0) * 100, 2) +
        pow(mat_X(4, 0) * 100, 2) +
        pow(mat_X(5, 0) * 100, 2));

    if (deltaR < 0.05 && deltaT < 0.05)
    {
        return true; // converged
    }
    return false; // keep optimizing
}

void Map_Optimization::transformUpdate()
{
    // 当前帧位姿
    incrementalOdometryAffineBack = trans2Affine3f(transformTobeMapped);
}

// 设置当前帧为关键帧并执行因子图优化
void Map_Optimization::saveKeyFramesAndFactor()
{
    // 1、计算当前帧与前一帧位姿变换，如果变化太小，不设为关键帧，反之设为关键帧
    // 2、添加激光里程计因子、GPS因子、闭环因子
    // 3、执行因子图优化
    // 4、得到当前帧优化后位姿，位姿协方差
    // 5、添加cloudKeyPoses3D，cloudKeyPoses6D，更新transformTobeMapped，添加当前关键帧的角点、平面点集合

    // 计算当前帧与前一帧位姿变换，如果变化太小，不设为关键帧，反之设为关键帧
    if (saveFrame() == false)
        return;

    // 添加激光里程计因子
    addOdomFactor();

    // 添加回环因子
    addLoopFactor();

    // cout << "****************************************************" << endl;
    // gtSAMgraph.print("GTSAM Graph:\n");

    // std::cout << "init error: " << gtSAMgraph.error(initialEstimate) << std::endl;

    // update iSAM
    // 执行优化
    isam->update(gtSAMgraph, initialEstimate);
    isam->update();

    // Values result = LevenbergMarquardtOptimizer(gtSAMgraph, initialEstimate).optimize();
    // std::cout << "final error: " << gtSAMgraph.error(result) << std::endl;

    if (aLoopIsClosed == true)
    {
        isam->update();
        isam->update();
        isam->update();
        isam->update();
        isam->update();
    }
    // update之后要清空一下保存的因子图，注：历史数据不会清掉，ISAM保存起来了
    gtSAMgraph.resize(0);
    initialEstimate.clear();

    // save key poses
    PointType thisPose3D;
    PointTypePose thisPose6D;
    Pose3 latestEstimate;

    // 优化结果
    isamCurrentEstimate = isam->calculateEstimate();

    // std::cout << "isamCurrentEstimate.size: " << isamCurrentEstimate.size() << std::endl;
    // 当前帧位姿结果
    // latestEstimate = isamCurrentEstimate.at<Pose3>(isamCurrentEstimate.size() - 1);
    latestEstimate = isamCurrentEstimate.at<Pose3>(X_node[graph_pose_num - 1]);

   
    // cout << "****************************************************" << endl;
    // isamCurrentEstimate.print("Current estimate: ");

    // std::cout << "latestEstimate.x " << latestEstimate.translation().x() << std::endl;

    // cloudKeyPoses3D加入当前帧位姿
    thisPose3D.x = latestEstimate.translation().x();
    thisPose3D.y = latestEstimate.translation().y();
    thisPose3D.z = latestEstimate.translation().z();
    thisPose3D.intensity = cloudKeyPoses3D->size(); // this can be used as index
    cloudKeyPoses3D->push_back(thisPose3D);

    // cloudKeyPoses6D加入当前帧位姿
    thisPose6D.x = thisPose3D.x;
    thisPose6D.y = thisPose3D.y;
    thisPose6D.z = thisPose3D.z;
    thisPose6D.intensity = thisPose3D.intensity; // this can be used as index
    thisPose6D.roll = latestEstimate.rotation().roll();
    thisPose6D.pitch = latestEstimate.rotation().pitch();
    thisPose6D.yaw = latestEstimate.rotation().yaw();
    thisPose6D.time = timeLaserInfoCur;
    cloudKeyPoses6D->push_back(thisPose6D);

    // cout << "****************************************************" << endl;
    // cout << "Pose covariance:" << endl;
    // cout << isam->marginalCovariance(isamCurrentEstimate.size()-1) << endl << endl;
    // 位姿协方差
    // poseCovariance = isam->marginalCovariance(isamCurrentEstimate.size() - 1);
    poseCovariance = isam->marginalCovariance(X_node[graph_pose_num - 1]);

    // save updated transform
    transformTobeMapped[0] = latestEstimate.rotation().roll();
    transformTobeMapped[1] = latestEstimate.rotation().pitch();
    transformTobeMapped[2] = latestEstimate.rotation().yaw();
    transformTobeMapped[3] = latestEstimate.translation().x();
    transformTobeMapped[4] = latestEstimate.translation().y();
    transformTobeMapped[5] = latestEstimate.translation().z();

    // save all the received edge and surf points
    // 当前帧激光角点、平面点，降采样集合
    pcl::PointCloud<PointType>::Ptr thisCornerKeyFrame(new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr thisSurfKeyFrame(new pcl::PointCloud<PointType>());
    pcl::copyPointCloud(*laserCloudCornerLastDS, *thisCornerKeyFrame);
    pcl::copyPointCloud(*laserCloudSurfLastDS, *thisSurfKeyFrame);

    // save key frame cloud
    cornerCloudKeyFrames.push_back(thisCornerKeyFrame);
    surfCloudKeyFrames.push_back(thisSurfKeyFrame);

    // save path for visualization
    // 更新里程计轨迹
    updatePath(thisPose6D);

    // std::cout << std::endl;
}

// 计算当前帧与前一帧位姿变换，如果变化太小，不设为关键帧，反之设为关键帧
bool Map_Optimization::saveFrame()
{
    if (cloudKeyPoses3D->points.empty())
        return true;

    // 前一帧位姿
    // 注：最开始没有的时候，在函数extractCloud里面有
    Eigen::Affine3f transStart = pclPointToAffine3f(cloudKeyPoses6D->back());
    // 当前帧位姿
    Eigen::Affine3f transFinal = pcl::getTransformation(transformTobeMapped[3], transformTobeMapped[4], transformTobeMapped[5],
                                                        transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]);
    // 位姿变换增量
    Eigen::Affine3f transBetween = transStart.inverse() * transFinal;
    float x, y, z, roll, pitch, yaw;
    pcl::getTranslationAndEulerAngles(transBetween, x, y, z, roll, pitch, yaw);
    // 旋转和平移量都较小，当前帧不设为关键帧
    if (abs(roll) < surroundingkeyframeAddingAngleThreshold &&
        abs(pitch) < surroundingkeyframeAddingAngleThreshold &&
        abs(yaw) < surroundingkeyframeAddingAngleThreshold &&
        sqrt(x * x + y * y + z * z) < surroundingkeyframeAddingDistThreshold)
        return false;

    return true;
}

// 添加激光里程计因子
void Map_Optimization::addOdomFactor()
{
    if (cloudKeyPoses3D->points.empty())
    {
        // 第一帧初始化先验因子
        noiseModel::Diagonal::shared_ptr priorNoise = noiseModel::Diagonal::Variances((Vector(6) << 1e-2, 1e-2, M_PI * M_PI, 1e8, 1e8, 1e8).finished()); // rad*rad, meter*meter
        X_node.push_back(gtsam::Symbol('x', 0));
        // gtSAMgraph.add(PriorFactor<Pose3>(0, trans2gtsamPose(transformTobeMapped), priorNoise));
        gtSAMgraph.add(PriorFactor<Pose3>(X_node[0], trans2gtsamPose(transformTobeMapped), priorNoise));
        // 变量节点设置初始值
        initialEstimate.insert(X_node[0], trans2gtsamPose(transformTobeMapped));
        graph_pose_num++;
    }
    else
    {
        // 添加激光里程计因子
        noiseModel::Diagonal::shared_ptr odometryNoise = noiseModel::Diagonal::Variances((Vector(6) << 1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4).finished());
        gtsam::Pose3 poseFrom = pclPointTogtsamPose3(cloudKeyPoses6D->points.back());
        gtsam::Pose3 poseTo = trans2gtsamPose(transformTobeMapped);

        while (X_node.size() <= cloudKeyPoses3D->size())
        {
            X_node.push_back(gtsam::Symbol('x', X_node.size()));
        }

        gtSAMgraph.add(BetweenFactor<Pose3>(X_node[cloudKeyPoses3D->size() - 1], X_node[cloudKeyPoses3D->size()], poseFrom.between(poseTo), odometryNoise));
        initialEstimate.insert(X_node[cloudKeyPoses3D->size()], poseTo);
        graph_pose_num++;
        // // 参数：前一帧id，当前帧id，前一帧与当前帧的位姿变换（作为观测值），噪声协方差
        // gtSAMgraph.add(BetweenFactor<Pose3>(cloudKeyPoses3D->size() - 1, cloudKeyPoses3D->size(), poseFrom.between(poseTo), odometryNoise));
        // // 变量节点设置初始值
        // initialEstimate.insert(cloudKeyPoses3D->size(), poseTo);
    }
}

// 添加闭环因子
void Map_Optimization::addLoopFactor()
{
    if (loopIndexQueue.empty())
        return;

    // 闭环队列
    for (int i = 0; i < (int)loopIndexQueue.size(); ++i)
    {
        // 闭环边对应两帧的索引
        int indexFrom = loopIndexQueue[i].first;
        int indexTo = loopIndexQueue[i].second;
        // 闭环边的位姿变换
        gtsam::Pose3 poseBetween = loopPoseQueue[i];
        gtsam::noiseModel::Diagonal::shared_ptr noiseBetween = loopNoiseQueue[i];
        while (X_node.size() <= cloudKeyPoses3D->size())
        {
            X_node.push_back(gtsam::Symbol('x', X_node.size()));
        }
        gtSAMgraph.add(BetweenFactor<Pose3>(X_node[indexFrom], X_node[indexTo], poseBetween, noiseBetween));
        // gtSAMgraph.add(BetweenFactor<Pose3>(indexFrom, indexTo, poseBetween, noiseBetween));
    }

    loopIndexQueue.clear();
    loopPoseQueue.clear();
    loopNoiseQueue.clear();
    aLoopIsClosed = true;
}

// 更新里程计轨迹
void Map_Optimization::updatePath(const PointTypePose &pose_in)
{

    geometry_msgs::PoseStamped pose_stamped;
    pose_stamped.header.stamp = ros::Time().fromSec(pose_in.time);

    pose_stamped.header.frame_id = odometryFrame;
    pose_stamped.pose.position.x = pose_in.x;
    pose_stamped.pose.position.y = pose_in.y;
    pose_stamped.pose.position.z = pose_in.z;
    tf::Quaternion q = tf::createQuaternionFromRPY(pose_in.roll, pose_in.pitch, pose_in.yaw);
    pose_stamped.pose.orientation.x = q.x();
    pose_stamped.pose.orientation.y = q.y();
    pose_stamped.pose.orientation.z = q.z();
    pose_stamped.pose.orientation.w = q.w();

    globalPath.poses.push_back(pose_stamped);
}

// 更新因子图中所有变量节点的位姿，也就是所有历史关键帧的位姿，更新里程计轨迹
void Map_Optimization::correctPoses()
{
    if (cloudKeyPoses3D->points.empty())
        return;

    if (aLoopIsClosed == true)
    {
        // clear map cache
        // 清空局部map
        laserCloudMapContainer.clear();
        // clear path
        // 清空里程计轨迹
        globalPath.poses.clear();
        // update key poses
        // 更新因子图中所有变量节点的位姿，也就是所有历史关键帧的位姿
        // int numPoses = isamCurrentEstimate.size();
        int numPoses = graph_pose_num;
        for (int i = 0; i < numPoses; ++i)
        {
            cloudKeyPoses3D->points[i].x = isamCurrentEstimate.at<Pose3>(X_node[i]).translation().x();
            cloudKeyPoses3D->points[i].y = isamCurrentEstimate.at<Pose3>(X_node[i]).translation().y();
            cloudKeyPoses3D->points[i].z = isamCurrentEstimate.at<Pose3>(X_node[i]).translation().z();

            cloudKeyPoses6D->points[i].x = cloudKeyPoses3D->points[i].x;
            cloudKeyPoses6D->points[i].y = cloudKeyPoses3D->points[i].y;
            cloudKeyPoses6D->points[i].z = cloudKeyPoses3D->points[i].z;
            cloudKeyPoses6D->points[i].roll = isamCurrentEstimate.at<Pose3>(X_node[i]).rotation().roll();
            cloudKeyPoses6D->points[i].pitch = isamCurrentEstimate.at<Pose3>(X_node[i]).rotation().pitch();
            cloudKeyPoses6D->points[i].yaw = isamCurrentEstimate.at<Pose3>(X_node[i]).rotation().yaw();

            updatePath(cloudKeyPoses6D->points[i]);
        }

        aLoopIsClosed = false;
    }
}

// 发布激光里程计
void Map_Optimization::publishOdometry()
{
    // Publish odometry for ROS (global)
    // 发布激光里程计，odom等价map
    nav_msgs::Odometry laserOdometryROS;
    laserOdometryROS.header.stamp = timeLaserInfoStamp;
    laserOdometryROS.header.frame_id = odometryFrame;
    laserOdometryROS.child_frame_id = "odom_mapping";
    laserOdometryROS.pose.pose.position.x = transformTobeMapped[3];
    laserOdometryROS.pose.pose.position.y = transformTobeMapped[4];
    laserOdometryROS.pose.pose.position.z = transformTobeMapped[5];
    laserOdometryROS.pose.pose.orientation = tf::createQuaternionMsgFromRollPitchYaw(transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]);
    pubLaserOdometryGlobal.publish(laserOdometryROS);

    // Publish TF
    // 发布TF，odom->lidar
    static tf::TransformBroadcaster br;
    tf::Transform t_odom_to_lidar = tf::Transform(tf::createQuaternionFromRPY(transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]),
                                                  tf::Vector3(transformTobeMapped[3], transformTobeMapped[4], transformTobeMapped[5]));
    tf::StampedTransform trans_odom_to_lidar = tf::StampedTransform(t_odom_to_lidar, timeLaserInfoStamp, odometryFrame, "base_link");
    br.sendTransform(trans_odom_to_lidar);

    // Publish odometry for ROS (incremental)
    static bool lastIncreOdomPubFlag = false;
    static nav_msgs::Odometry laserOdomIncremental; // incremental odometry msg
    static Eigen::Affine3f increOdomAffine;         // incremental odometry in affine
    // 第一次数据直接用全局里程计初始化
    if (lastIncreOdomPubFlag == false)
    {
        lastIncreOdomPubFlag = true;
        laserOdomIncremental = laserOdometryROS;
        increOdomAffine = trans2Affine3f(transformTobeMapped);
    }
    else
    {
        // 当前帧与前一帧之间的位姿变换
        Eigen::Affine3f affineIncre = incrementalOdometryAffineFront.inverse() * incrementalOdometryAffineBack;
        increOdomAffine = increOdomAffine * affineIncre;
        float x, y, z, roll, pitch, yaw;
        pcl::getTranslationAndEulerAngles(increOdomAffine, x, y, z, roll, pitch, yaw);
        /* if (cloudInfo.imuAvailable == true)
         {
             if (std::abs(cloudInfo.imuPitchInit) < 1.4)
             {
                 double imuWeight = 0.1;
                 tf::Quaternion imuQuaternion;
                 tf::Quaternion transformQuaternion;
                 double rollMid, pitchMid, yawMid;

                 // slerp roll
                 transformQuaternion.setRPY(roll, 0, 0);
                 imuQuaternion.setRPY(cloudInfo.imuRollInit, 0, 0);
                 tf::Matrix3x3(transformQuaternion.slerp(imuQuaternion, imuWeight)).getRPY(rollMid, pitchMid, yawMid);
                 roll = rollMid;

                 // slerp pitch
                 transformQuaternion.setRPY(0, pitch, 0);
                 imuQuaternion.setRPY(0, cloudInfo.imuPitchInit, 0);
                 tf::Matrix3x3(transformQuaternion.slerp(imuQuaternion, imuWeight)).getRPY(rollMid, pitchMid, yawMid);
                 pitch = pitchMid;
             }
         }*/
        laserOdomIncremental.header.stamp = timeLaserInfoStamp;
        laserOdomIncremental.header.frame_id = odometryFrame;
        laserOdomIncremental.child_frame_id = "odom_mapping";
        laserOdomIncremental.pose.pose.position.x = x;
        laserOdomIncremental.pose.pose.position.y = y;
        laserOdomIncremental.pose.pose.position.z = z;
        laserOdomIncremental.pose.pose.orientation = tf::createQuaternionMsgFromRollPitchYaw(roll, pitch, yaw);
        if (isDegenerate)
            laserOdomIncremental.pose.covariance[0] = 1;
        else
            laserOdomIncremental.pose.covariance[0] = 0;
    }
    pubLaserOdometryIncremental.publish(laserOdomIncremental);
}

// 发布里程计、点云、轨迹
void Map_Optimization::publishFrames()
{
    // 1、发布历史关键帧位姿集合
    // 2、发布局部map的降采样平面点集合
    // 3、发布历史帧（累加的）的角点、平面点降采样集合
    // 4、发布里程计轨迹

    if (cloudKeyPoses3D->points.empty())
        return;
    // publish key poses
    publishCloud(&pubKeyPoses, cloudKeyPoses3D, timeLaserInfoStamp, odometryFrame);
    // Publish surrounding key frames
    publishCloud(&pubRecentKeyFrames, laserCloudSurfFromMapDS, timeLaserInfoStamp, odometryFrame);
    // publish registered key frame

    if (pubRecentKeyFrame.getNumSubscribers() != 0)
    {
        pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());
        PointTypePose thisPose6D = trans2PointTypePose(transformTobeMapped);
        *cloudOut += *transformPointCloud(laserCloudCornerLastDS, &thisPose6D);
        *cloudOut += *transformPointCloud(laserCloudSurfLastDS, &thisPose6D);
        publishCloud(&pubRecentKeyFrame, cloudOut, timeLaserInfoStamp, odometryFrame);
    }
    if (pubRecentKeyFrame_corner.getNumSubscribers() != 0)
    {
        PointTypePose thisPose6D = trans2PointTypePose(transformTobeMapped);
        pcl::PointCloud<PointType>::Ptr corner_cloudOut(new pcl::PointCloud<PointType>());
        corner_cloudOut = transformPointCloud(laserCloudCornerLastDS, &thisPose6D);
        // 发布边缘点集合（累积）
        publishCloud(&pubRecentKeyFrame_corner, corner_cloudOut, timeLaserInfoStamp, odometryFrame);
    }
    // publish registered high-res raw cloud
    if (pubCloudRegisteredRaw.getNumSubscribers() != 0)
    {
        // pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());
        // pcl::fromROSMsg(cloudInfo.cloud_deskewed, *cloudOut);
        // PointTypePose thisPose6D = trans2PointTypePose(transformTobeMapped);
        // *cloudOut = *transformPointCloud(cloudOut, &thisPose6D);
        // publishCloud(&pubCloudRegisteredRaw, cloudOut, timeLaserInfoStamp, odometryFrame);
    }
    // publish path
    if (pubPath.getNumSubscribers() != 0)
    {
        globalPath.header.stamp = timeLaserInfoStamp;
        globalPath.header.frame_id = odometryFrame;
        pubPath.publish(globalPath);
    }
}

void Map_Optimization::pointAssociateToMap(PointType const *const pi, PointType *const po)
{
    po->x = transPointAssociateToMap(0, 0) * pi->x + transPointAssociateToMap(0, 1) * pi->y + transPointAssociateToMap(0, 2) * pi->z + transPointAssociateToMap(0, 3);
    po->y = transPointAssociateToMap(1, 0) * pi->x + transPointAssociateToMap(1, 1) * pi->y + transPointAssociateToMap(1, 2) * pi->z + transPointAssociateToMap(1, 3);
    po->z = transPointAssociateToMap(2, 0) * pi->x + transPointAssociateToMap(2, 1) * pi->y + transPointAssociateToMap(2, 2) * pi->z + transPointAssociateToMap(2, 3);
    po->intensity = pi->intensity;
}

pcl::PointCloud<PointType>::Ptr Map_Optimization::transformPointCloud(pcl::PointCloud<PointType>::Ptr cloudIn, PointTypePose *transformIn)
{
    pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());

    int cloudSize = cloudIn->size();
    cloudOut->resize(cloudSize);

    Eigen::Affine3f transCur = pcl::getTransformation(transformIn->x, transformIn->y, transformIn->z, transformIn->roll, transformIn->pitch, transformIn->yaw);

#pragma omp parallel for num_threads(numberOfCores)
    for (int i = 0; i < cloudSize; ++i)
    {
        const auto &pointFrom = cloudIn->points[i];
        cloudOut->points[i].x = transCur(0, 0) * pointFrom.x + transCur(0, 1) * pointFrom.y + transCur(0, 2) * pointFrom.z + transCur(0, 3);
        cloudOut->points[i].y = transCur(1, 0) * pointFrom.x + transCur(1, 1) * pointFrom.y + transCur(1, 2) * pointFrom.z + transCur(1, 3);
        cloudOut->points[i].z = transCur(2, 0) * pointFrom.x + transCur(2, 1) * pointFrom.y + transCur(2, 2) * pointFrom.z + transCur(2, 3);
        cloudOut->points[i].intensity = pointFrom.intensity;
    }
    return cloudOut;
}

gtsam::Pose3 Map_Optimization::pclPointTogtsamPose3(PointTypePose thisPoint)
{
    return gtsam::Pose3(gtsam::Rot3::RzRyRx(double(thisPoint.roll), double(thisPoint.pitch), double(thisPoint.yaw)),
                        gtsam::Point3(double(thisPoint.x), double(thisPoint.y), double(thisPoint.z)));
}

gtsam::Pose3 Map_Optimization::trans2gtsamPose(float transformIn[])
{
    return gtsam::Pose3(gtsam::Rot3::RzRyRx(transformIn[0], transformIn[1], transformIn[2]),
                        gtsam::Point3(transformIn[3], transformIn[4], transformIn[5]));
}

Eigen::Affine3f Map_Optimization::pclPointToAffine3f(PointTypePose thisPoint)
{
    return pcl::getTransformation(thisPoint.x, thisPoint.y, thisPoint.z, thisPoint.roll, thisPoint.pitch, thisPoint.yaw);
}

Eigen::Affine3f Map_Optimization::trans2Affine3f(float transformIn[])
{
    return pcl::getTransformation(transformIn[3], transformIn[4], transformIn[5], transformIn[0], transformIn[1], transformIn[2]);
}

PointTypePose Map_Optimization::trans2PointTypePose(float transformIn[])
{
    PointTypePose thisPose6D;
    thisPose6D.x = transformIn[3];
    thisPose6D.y = transformIn[4];
    thisPose6D.z = transformIn[5];
    thisPose6D.roll = transformIn[0];
    thisPose6D.pitch = transformIn[1];
    thisPose6D.yaw = transformIn[2];
    return thisPose6D;
}

// 闭环线程
void Map_Optimization::loopClosureThread()
{
    //    1、闭环scan-to-map，icp优化位姿
    //  *   1) 在历史关键帧中查找与当前关键帧距离最近的关键帧集合，选择时间相隔较远的一帧作为候选闭环帧
    //  *   2) 提取当前关键帧特征点集合，降采样；提取闭环匹配关键帧前后相邻若干帧的关键帧特征点集合，降采样
    //  *   3) 执行scan-to-map优化，调用icp方法，得到优化后位姿，构造闭环因子需要的数据，在因子图优化中一并加入更新位姿
    //  * 2、rviz展示闭环边

    if (loopClosureEnableFlag == false)
        return;

    ros::Rate rate(loopClosureFrequency);
    while (ros::ok())
    {
        rate.sleep();
        performLoopClosure();
        visualizeLoopClosure();
    }
}

// 闭环scan-to-map，icp优化位姿
void Map_Optimization::performLoopClosure()
{
    //    1、在历史关键帧中查找与当前关键帧距离最近的关键帧集合，选择时间相隔较远的一帧作为候选闭环帧
    //  * 2、提取当前关键帧特征点集合，降采样；提取闭环匹配关键帧前后相邻若干帧的关键帧特征点集合，降采样
    //  * 3、执行scan-to-map优化，调用icp方法，得到优化后位姿，构造闭环因子需要的数据，在因子图优化中一并加入更新位姿
    //  * 注：闭环的时候没有立即更新当前帧的位姿，而是添加闭环因子，让图优化去更新位姿

    if (cloudKeyPoses3D->points.empty() == true)
        return;

    mtx.lock();
    *copy_cloudKeyPoses3D = *cloudKeyPoses3D;
    *copy_cloudKeyPoses6D = *cloudKeyPoses6D;
    mtx.unlock();

    // find keys
    // 当前关键帧索引，候选闭环匹配帧索引
    int loopKeyCur;
    int loopKeyPre;
    if (detectLoopClosureExternal(&loopKeyCur, &loopKeyPre) == false)
        if (detectLoopClosureDistance(&loopKeyCur, &loopKeyPre) == false)
            return;

    // extract cloud
    // 提取
    pcl::PointCloud<PointType>::Ptr cureKeyframeCloud(new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr prevKeyframeCloud(new pcl::PointCloud<PointType>());
    {
        loopFindNearKeyframes(cureKeyframeCloud, loopKeyCur, 0);
        loopFindNearKeyframes(prevKeyframeCloud, loopKeyPre, historyKeyframeSearchNum);
        if (cureKeyframeCloud->size() < 300 || prevKeyframeCloud->size() < 1000)
            return;
        if (pubHistoryKeyFrames.getNumSubscribers() != 0)
            publishCloud(&pubHistoryKeyFrames, prevKeyframeCloud, timeLaserInfoStamp, odometryFrame);
    }

    // ICP Settings
    // ICP参数设置
    static pcl::IterativeClosestPoint<PointType, PointType> icp;
    icp.setMaxCorrespondenceDistance(historyKeyframeSearchRadius * 2);
    icp.setMaximumIterations(100);
    icp.setTransformationEpsilon(1e-6);
    icp.setEuclideanFitnessEpsilon(1e-6);
    icp.setRANSACIterations(0);

    // Align clouds
    icp.setInputSource(cureKeyframeCloud);
    icp.setInputTarget(prevKeyframeCloud);
    pcl::PointCloud<PointType>::Ptr unused_result(new pcl::PointCloud<PointType>());
    icp.align(*unused_result);

    // 未收敛，或者匹配不够好
    if (icp.hasConverged() == false || icp.getFitnessScore() > historyKeyframeFitnessScore)
        return;

    // publish corrected cloud
    // 发布当前关键帧经过闭环优化后的位姿变换之后的特征点云
    if (pubIcpKeyFrames.getNumSubscribers() != 0)
    {
        pcl::PointCloud<PointType>::Ptr closed_cloud(new pcl::PointCloud<PointType>());
        pcl::transformPointCloud(*cureKeyframeCloud, *closed_cloud, icp.getFinalTransformation());
        publishCloud(&pubIcpKeyFrames, closed_cloud, timeLaserInfoStamp, odometryFrame);
    }

    // Get pose transformation
    // 闭环优化得到的当前关键帧与闭环关键帧之间的位姿变换
    float x, y, z, roll, pitch, yaw;
    Eigen::Affine3f correctionLidarFrame;
    correctionLidarFrame = icp.getFinalTransformation();
    // transform from world origin to wrong pose
    // 闭环优化前当前帧位姿
    Eigen::Affine3f tWrong = pclPointToAffine3f(copy_cloudKeyPoses6D->points[loopKeyCur]);
    // transform from world origin to corrected pose
    // 闭环优化后当前帧位姿
    Eigen::Affine3f tCorrect = correctionLidarFrame * tWrong; // pre-multiplying -> successive rotation about a fixed frame
    pcl::getTranslationAndEulerAngles(tCorrect, x, y, z, roll, pitch, yaw);
    gtsam::Pose3 poseFrom = Pose3(Rot3::RzRyRx(roll, pitch, yaw), Point3(x, y, z));
    // 闭环匹配帧的位姿
    gtsam::Pose3 poseTo = pclPointTogtsamPose3(copy_cloudKeyPoses6D->points[loopKeyPre]);
    gtsam::Vector Vector6(6);
    float noiseScore = icp.getFitnessScore();
    Vector6 << noiseScore, noiseScore, noiseScore, noiseScore, noiseScore, noiseScore;
    noiseModel::Diagonal::shared_ptr constraintNoise = noiseModel::Diagonal::Variances(Vector6);

    // std::cout << "icp Score:" << icp.getFitnessScore() << std::endl;
    // std::cout << "x: " << x << ", y: " << y << ", z: " << z << ", roll:" << roll << ", pitch: "  << pitch << ", yaw: " << yaw  << std::endl;

    // Add pose constraint
    // 添加闭环因子需要的数据
    // 这些内容会在函数addLoopFactor中用到
    mtx.lock();
    loopIndexQueue.push_back(make_pair(loopKeyCur, loopKeyPre));
    loopPoseQueue.push_back(poseFrom.between(poseTo));
    loopNoiseQueue.push_back(constraintNoise);
    mtx.unlock();

    // add loop constriant
    loopIndexContainer[loopKeyCur] = loopKeyPre;
}

// rviz展示闭环边
void Map_Optimization::visualizeLoopClosure()
{
    if (loopIndexContainer.empty())
        return;

    visualization_msgs::MarkerArray markerArray;
    // loop nodes
    visualization_msgs::Marker markerNode;
    markerNode.header.frame_id = odometryFrame;
    markerNode.header.stamp = timeLaserInfoStamp;
    markerNode.action = visualization_msgs::Marker::ADD;
    markerNode.type = visualization_msgs::Marker::SPHERE_LIST;
    markerNode.ns = "loop_nodes";
    markerNode.id = 0;
    markerNode.pose.orientation.w = 1;
    markerNode.scale.x = 0.3;
    markerNode.scale.y = 0.3;
    markerNode.scale.z = 0.3;
    markerNode.color.r = 0;
    markerNode.color.g = 0.8;
    markerNode.color.b = 1;
    markerNode.color.a = 1;
    // loop edges
    visualization_msgs::Marker markerEdge;
    markerEdge.header.frame_id = odometryFrame;
    markerEdge.header.stamp = timeLaserInfoStamp;
    markerEdge.action = visualization_msgs::Marker::ADD;
    markerEdge.type = visualization_msgs::Marker::LINE_LIST;
    markerEdge.ns = "loop_edges";
    markerEdge.id = 1;
    markerEdge.pose.orientation.w = 1;
    markerEdge.scale.x = 0.1;
    markerEdge.color.r = 0.9;
    markerEdge.color.g = 0.9;
    markerEdge.color.b = 0;
    markerEdge.color.a = 1;
    // 遍历闭环
    for (auto it = loopIndexContainer.begin(); it != loopIndexContainer.end(); ++it)
    {
        int key_cur = it->first;
        int key_pre = it->second;
        geometry_msgs::Point p;
        p.x = copy_cloudKeyPoses6D->points[key_cur].x;
        p.y = copy_cloudKeyPoses6D->points[key_cur].y;
        p.z = copy_cloudKeyPoses6D->points[key_cur].z;
        markerNode.points.push_back(p);
        markerEdge.points.push_back(p);
        p.x = copy_cloudKeyPoses6D->points[key_pre].x;
        p.y = copy_cloudKeyPoses6D->points[key_pre].y;
        p.z = copy_cloudKeyPoses6D->points[key_pre].z;
        markerNode.points.push_back(p);
        markerEdge.points.push_back(p);
    }

    markerArray.markers.push_back(markerNode);
    markerArray.markers.push_back(markerEdge);
    pubLoopConstraintEdge.publish(markerArray);
}

// 在历史关键帧中查找与当前关键帧距离最近的关键帧集合，选择时间相隔较远的一帧作为候选闭环帧
bool Map_Optimization::detectLoopClosureDistance(int *latestID, int *closestID)
{
    int loopKeyCur = copy_cloudKeyPoses3D->size() - 1;
    int loopKeyPre = -1;

    // check loop constraint added before
    // 当前帧已经添加过闭环对应关系，不再继续添加
    auto it = loopIndexContainer.find(loopKeyCur);
    if (it != loopIndexContainer.end())
        return false;

    // find the closest history key frame
    // 在历史关键帧中查找与当前关键帧距离最近的关键帧集合
    // 配置文件中默认historyKeyframeSearchRadius=15m
    std::vector<int> pointSearchIndLoop;
    std::vector<float> pointSearchSqDisLoop;
    kdtreeHistoryKeyPoses->setInputCloud(copy_cloudKeyPoses3D);
    kdtreeHistoryKeyPoses->radiusSearch(copy_cloudKeyPoses3D->back(), historyKeyframeSearchRadius, pointSearchIndLoop, pointSearchSqDisLoop, 0);

    // 在候选关键帧集合中，找到与当前帧时间相隔较远的帧，设为候选匹配帧
    // 配置文件中默认30s
    for (int i = 0; i < (int)pointSearchIndLoop.size(); ++i)
    {
        int id = pointSearchIndLoop[i];
        if (abs(copy_cloudKeyPoses6D->points[id].time - timeLaserInfoCur) > historyKeyframeSearchTimeDiff)
        {
            loopKeyPre = id;
            break;
        }
    }

    if (loopKeyPre == -1 || loopKeyCur == loopKeyPre)
        return false;

    *latestID = loopKeyCur;
    *closestID = loopKeyPre;

    return true;
}

// 来自外部闭环检测程序提供的闭环匹配索引对
bool Map_Optimization::detectLoopClosureExternal(int *latestID, int *closestID)
{
    // this function is not used yet, please ignore it
    int loopKeyCur = -1;
    int loopKeyPre = -1;

    std::lock_guard<std::mutex> lock(mtxLoopInfo);
    if (loopInfoVec.empty())
        return false;

    double loopTimeCur = loopInfoVec.front().data[0];
    double loopTimePre = loopInfoVec.front().data[1];
    loopInfoVec.pop_front();

    if (abs(loopTimeCur - loopTimePre) < historyKeyframeSearchTimeDiff)
        return false;

    int cloudSize = copy_cloudKeyPoses6D->size();
    if (cloudSize < 2)
        return false;

    // latest key
    loopKeyCur = cloudSize - 1;
    for (int i = cloudSize - 1; i >= 0; --i)
    {
        if (copy_cloudKeyPoses6D->points[i].time >= loopTimeCur)
            loopKeyCur = round(copy_cloudKeyPoses6D->points[i].intensity);
        else
            break;
    }

    // previous key
    loopKeyPre = 0;
    for (int i = 0; i < cloudSize; ++i)
    {
        if (copy_cloudKeyPoses6D->points[i].time <= loopTimePre)
            loopKeyPre = round(copy_cloudKeyPoses6D->points[i].intensity);
        else
            break;
    }

    if (loopKeyCur == loopKeyPre)
        return false;

    auto it = loopIndexContainer.find(loopKeyCur);
    if (it != loopIndexContainer.end())
        return false;

    *latestID = loopKeyCur;
    *closestID = loopKeyPre;

    return true;
}

// 提取key索引的关键帧前后相邻若干帧的关键帧特征点集合，降采样
void Map_Optimization::loopFindNearKeyframes(pcl::PointCloud<PointType>::Ptr &nearKeyframes, const int &key, const int &searchNum)
{
    // extract near keyframes
    // 提取key索引的关键帧前后相邻若干帧的关键帧特征点集合
    nearKeyframes->clear();
    int cloudSize = copy_cloudKeyPoses6D->size();
    // 通过-searchNum 到 +searchNum，搜索key两侧内容
    for (int i = -searchNum; i <= searchNum; ++i)
    {
        int keyNear = key + i;
        if (keyNear < 0 || keyNear >= cloudSize)
            continue;
        *nearKeyframes += *transformPointCloud(cornerCloudKeyFrames[keyNear], &copy_cloudKeyPoses6D->points[keyNear]);
        *nearKeyframes += *transformPointCloud(surfCloudKeyFrames[keyNear], &copy_cloudKeyPoses6D->points[keyNear]);
    }

    if (nearKeyframes->empty())
        return;

    // downsample near keyframes
    pcl::PointCloud<PointType>::Ptr cloud_temp(new pcl::PointCloud<PointType>());
    downSizeFilterICP.setInputCloud(nearKeyframes);
    downSizeFilterICP.filter(*cloud_temp);
    *nearKeyframes = *cloud_temp;
}

// 展示线程
void Map_Optimization::visualizeGlobalMapThread()
{
    // 1、发布局部关键帧map的特征点云
    // 2、保存全局关键帧特征点集合

    ros::Rate rate(0.2);
    while (ros::ok())
    {
        rate.sleep();
        publishGlobalMap();
    }

    if (savePCD == false)
        return;

    // LO_SAM::save_mapRequest req;
    // LO_SAM::save_mapResponse res;

    // if (!saveMapService(req, res))
    // {
    //     cout << "Fail to save map" << endl;
    // }
}

// 发布局部关键帧map的特征点云
void Map_Optimization::publishGlobalMap()
{
    if (pubLaserCloudSurround.getNumSubscribers() == 0)
        return;

    if (cloudKeyPoses3D->points.empty() == true)
        return;

    pcl::KdTreeFLANN<PointType>::Ptr kdtreeGlobalMap(new pcl::KdTreeFLANN<PointType>());
    pcl::PointCloud<PointType>::Ptr globalMapKeyPoses(new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr globalMapKeyPosesDS(new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr globalMapKeyFrames(new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr globalMapKeyFramesDS(new pcl::PointCloud<PointType>());

    // kd-tree to find near key frames to visualize
    // kdtree查找最近一帧关键帧相邻的关键帧集合
    std::vector<int> pointSearchIndGlobalMap;
    std::vector<float> pointSearchSqDisGlobalMap;
    // search near key frames to visualize
    mtx.lock();
    kdtreeGlobalMap->setInputCloud(cloudKeyPoses3D);
    kdtreeGlobalMap->radiusSearch(cloudKeyPoses3D->back(), globalMapVisualizationSearchRadius, pointSearchIndGlobalMap, pointSearchSqDisGlobalMap, 0);
    mtx.unlock();

    for (int i = 0; i < (int)pointSearchIndGlobalMap.size(); ++i)
        globalMapKeyPoses->push_back(cloudKeyPoses3D->points[pointSearchIndGlobalMap[i]]);
    // downsample near selected key frames
    // 降采样
    pcl::VoxelGrid<PointType> downSizeFilterGlobalMapKeyPoses;                                                                                            // for global map visualization
    downSizeFilterGlobalMapKeyPoses.setLeafSize(globalMapVisualizationPoseDensity, globalMapVisualizationPoseDensity, globalMapVisualizationPoseDensity); // for global map visualization
    downSizeFilterGlobalMapKeyPoses.setInputCloud(globalMapKeyPoses);
    downSizeFilterGlobalMapKeyPoses.filter(*globalMapKeyPosesDS);
    for (auto &pt : globalMapKeyPosesDS->points)
    {
        kdtreeGlobalMap->nearestKSearch(pt, 1, pointSearchIndGlobalMap, pointSearchSqDisGlobalMap);
        pt.intensity = cloudKeyPoses3D->points[pointSearchIndGlobalMap[0]].intensity;
    }

    // extract visualized and downsampled key frames
    // 提取局部相邻关键帧对应的特征点云
    for (int i = 0; i < (int)globalMapKeyPosesDS->size(); ++i)
    {
        if (pointDistance(globalMapKeyPosesDS->points[i], cloudKeyPoses3D->back()) > globalMapVisualizationSearchRadius)
            continue;
        int thisKeyInd = (int)globalMapKeyPosesDS->points[i].intensity;
        *globalMapKeyFrames += *transformPointCloud(cornerCloudKeyFrames[thisKeyInd], &cloudKeyPoses6D->points[thisKeyInd]);
        *globalMapKeyFrames += *transformPointCloud(surfCloudKeyFrames[thisKeyInd], &cloudKeyPoses6D->points[thisKeyInd]);
    }
    // downsample visualized points
    pcl::VoxelGrid<PointType> downSizeFilterGlobalMapKeyFrames;                                                                                   // for global map visualization
    downSizeFilterGlobalMapKeyFrames.setLeafSize(globalMapVisualizationLeafSize, globalMapVisualizationLeafSize, globalMapVisualizationLeafSize); // for global map visualization
    downSizeFilterGlobalMapKeyFrames.setInputCloud(globalMapKeyFrames);
    downSizeFilterGlobalMapKeyFrames.filter(*globalMapKeyFramesDS);
    publishCloud(&pubLaserCloudSurround, globalMapKeyFramesDS, timeLaserInfoStamp, odometryFrame);
}

void Map_Optimization::saveMap()
{
    // 保存全局关键帧特征点集合

    std::cout << "****************************************************" << std::endl;
    std::cout << "Saving map to pcd files ..." << std::endl;

    std::cout << "Save destination: " << (std::string(ROOT_DIR) + "PCD/") << std::endl;
    // create directory and remove old files;

    // save key frame transformations
    pcl::io::savePCDFileBinary(std::string(ROOT_DIR) + "PCD/trajectory.pcd", *cloudKeyPoses3D);
    pcl::io::savePCDFileBinary(std::string(ROOT_DIR) + "PCD/transformations.pcd", *cloudKeyPoses6D);
    // extract global point cloud map
    pcl::PointCloud<PointType>::Ptr globalCornerCloud(new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr globalCornerCloudDS(new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr globalSurfCloud(new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr globalSurfCloudDS(new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr globalMapCloud(new pcl::PointCloud<PointType>());
    for (int i = 0; i < (int)cloudKeyPoses3D->size(); i++)
    {
        *globalCornerCloud += *transformPointCloud(cornerCloudKeyFrames[i], &cloudKeyPoses6D->points[i]);
        *globalSurfCloud += *transformPointCloud(surfCloudKeyFrames[i], &cloudKeyPoses6D->points[i]);
        cout << "\r" << std::flush << "Processing feature cloud " << i << " of " << cloudKeyPoses6D->size() << " ...";
    }
    std::cout << std::endl;

    // save corner cloud
    pcl::io::savePCDFileBinary(std::string(ROOT_DIR) + "PCD/CornerMap.pcd", *globalCornerCloud);
    // save surf cloud
    pcl::io::savePCDFileBinary(std::string(ROOT_DIR) + "PCD/SurfMap.pcd", *globalSurfCloud);

    std::cout << "Saving map to pcd files completed" << std::endl;
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "lo_sam");

    ros::NodeHandle nh;
    ros::NodeHandle nh_private("~");

    Map_Optimization MO(nh, nh_private);

    ROS_INFO("\033[1;32m----> Map Optimization Started.\033[0m");

    std::thread loopthread(&Map_Optimization::loopClosureThread, &MO);
    std::thread visualizeMapThread(&Map_Optimization::visualizeGlobalMapThread, &MO);

    ros::spin();

    // odom_estimate_thread.join();
    loopthread.join();
    visualizeMapThread.join();

    return 0;
}
