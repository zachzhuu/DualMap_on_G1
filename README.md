## 代码记录
| 2025.11.18 | 创建 |  | Zach |
| --- | --- | --- | --- |
| 2025.11.21 | 代码暂存 | [https://github.com/zachzhuu/DualMap_on_G1/commit/c4cefd5fd1d704674f361dd9e07ea52083028972](https://github.com/zachzhuu/DualMap_on_G1/commit/c4cefd5fd1d704674f361dd9e07ea52083028972) |  |
| 2025.11.22 | 文档整理 |  |  |


## Prerequisite
| Device | System |
| --- | --- |
| PC | Ubuntu 22.04, ROS2-Humble |
| Nx on Unitree G1 | Ubuntu 20.04, ROS2-Foxy |


基本的setting是用网线将PC与宇树G1上的Nx板连接，保证在同一局域网内（包括机器人上搭载的Mid360雷达），然后通过ROS2共享topic以传输信息。

为实现PC与G1搭载Nx板的ROS2通讯，需要在本地安装unitree_ros2，并按照要求编译，配置网络IP，并source与cyclonedds相关的环境变量（我把`source ~/unitree_ros2/setup.sh`写在了~/.bashrc）。过程详见这个文档[https://support.unitree.com/home/zh/G1_developer/ros2_communication_routine](https://support.unitree.com/home/zh/G1_developer/ros2_communication_routine)。

_**<u>NOTE:</u>**_ 必须配置unitree_ros2。在本地PC上`ros2 topic list`可以成功看到宇树相关的topic，这并不标志着成功配置通信。即使没有安装untree_ros2库（[https://github.com/unitreerobotics/unitree_ros2](https://github.com/unitreerobotics/unitree_ros2)）并配置，只是更改PC在有线连接下的IP也是可以看到topic的，但是当你launch相机或者雷达后，去`ros2 topic echo/hz`这些topic却是空的。

_**<u>NOTE:</u>**__** **_据说unitree SDK的功能需要将机器人固件升级到最新才可以使用。起初想要使用自带的SLAM SDK（[https://support.unitree.com/home/zh/G1_developer/slam_navigation_services_interface](https://support.unitree.com/home/zh/G1_developer/slam_navigation_services_interface)），从输出中提取Odometry信息用于建图，但是始终无法得到topic内容。

## Installation
LiDAR数据用于得到实时的Pose，RGB-D数据用于语义建图。

### LiDAR Driver
**Step 1.** 在成功配置ROS2通讯后，可以通过`ping 192.168.123.120`测试雷达的连接。

**Step 2.** 在本地PC上配置livox_ros_driver2（[https://github.com/Livox-SDK/livox_ros_driver2](https://github.com/Livox-SDK/livox_ros_driver2)）。

+ `git clone`ROS Driver 2仓库
+ 按照仓库中的README文件安装并build LIVOX SDK2（[https://github.com/Livox-SDK/Livox-SDK2/blob/master/README.md](https://github.com/Livox-SDK/Livox-SDK2/blob/master/README.md)）
+ 立即参照[https://support.unitree.com/home/zh/G1_developer/lidar_Instructions#heading-6](https://support.unitree.com/home/zh/G1_developer/lidar_Instructions#heading-6)修改`MID360_config.json`
+ 最后才能build Driver 仓库（[https://github.com/Livox-SDK/livox_ros_driver2?tab=readme-ov-file#23-build-the-livox-ros-driver-2](https://github.com/Livox-SDK/livox_ros_driver2?tab=readme-ov-file#23-build-the-livox-ros-driver-2)）

如果没有修改`MID360_config.json`则需要重新编译。完成后可以运行以下命令验证安装，安装成功时可以看到有雷达实时点云的rviz窗口：

```bash
ros2 launch livox_ros_driver2 rviz_MID360_launch.py
```

###  FAST LIO ROS2
参照[https://github.com/Ericsii/FAST_LIO_ROS2](https://github.com/Ericsii/FAST_LIO_ROS2)。在自己工作目录下建立`/ws_fast_lio_ros2/src`，在当中进行`git clone`和`build`即可。

### Realsense Camera
Realsense相机是直接通过USB3.0连接在机器人的Nx板上的，要得到相机信息，需要在Nx板上安装realsense sdk2.0和realsense-ros。

**Step 1.** 安装过程需要连接WIFI，机器人联网方式详见[https://support.unitree.com/home/zh/G1_developer/FAQ#heading-2](https://support.unitree.com/home/zh/G1_developer/FAQ#heading-2)当中的<font style="color:rgb(63, 74, 84);">nmcli配置WIFI方式</font>

**<font style="color:rgb(63, 74, 84);">Step 2.</font>**<font style="color:rgb(63, 74, 84);"> 按照</font>[https://github.com/IntelRealSense/realsense-ros?tab=readme-ov-file#installation-on-ubuntu](https://github.com/IntelRealSense/realsense-ros?tab=readme-ov-file#installation-on-ubuntu)的三个步骤安装sdk和ROS wrapper（第二步按照Jetson教程安装librealsense2，第三步直接使用`sudo apt install 'ros-foxy-realsense2-*'`）

_**<u>NOTE:</u>**__** **_尽管realsense的安装步骤中声称：<font style="color:rgb(31, 35, 40);">Foxy EOL distro is not supported by this option (install by running </font>`<font style="color:rgb(31, 35, 40);">sudo apt install</font>`<font style="color:rgb(31, 35, 40);">)，但在本次实验中暂未出现毁灭性的bug。</font>

<font style="color:rgb(31, 35, 40);">安装完成后，即使在本地PC也安装了realsense sdk，还是使用</font>`<font style="color:rgb(31, 35, 40);">realsense-viewer</font>`<font style="color:rgb(31, 35, 40);">来查看相机图像，因为相机并未与本机相连。但是可以通过以下方法验证，先打开RGB-D streaming：</font>

```bash
ssh unitree@192.168.123.168

ros2 launch realsense2_camera rs_launch.py \
  depth_module.profile:=640x480x30 \
  rgb_camera.profile:=640x480x30 \
  align_depth.enable:=true
```

然后在本地PC的另一个terminal检查数据：

```bash
ros2 topic echo /camera/depth/camera_info
OR
ros2 topic hz /camera/color/image_raw
```

## LiDAR-Cam Calibration
目的是标定雷达到相机的坐标变换矩阵。

_**<u>NOTE:</u>**__** **_宇树G1自带雷达和相机的安装方式和角度决定了他们的共视区域非常的有限，详见真机或者装配图。机器人的前脸挡住了很多雷达本可以和相机交叉的区域，因此难以获得很好的标定结果（见TODO）。也许可以不用雷达，而是使用RGB-D SLAM等其他方式进行定位。另外，标定结果不佳还有其他的原因，在此不过多讨论。

_**<u>NOTE:</u>**__** **_适用于本次实验使用的标定板，整体是红色方形，中间一个有高反射率的白色圆形区域。

**Step 1.** 采集标定数据

+ 在terminal打开相机

```bash
ssh unitree@192.168.123.164
ros2 launch realsense2_camera rs_launch.py \
  depth_module.profile:=640x480x30 \
  rgb_camera.profile:=640x480x30 \
  align_depth.enable:=true
```

+ 在另一个terminal中打开雷达。**但是**不建议用这个LiDAR开启方式，而是`ros2 launch livox_ros_driver2 rviz_MID360_launch.py`，这样便于实时检查点云的效果，而不会盲录导致TODO的效果。

```bash
source /home/zach/repos/ws_livox/install/setup.bash
ros2 launch livox_ros_driver2 msg_MID360_launch.py
```

+ 运行`calibration/cali.py`（注意按实际topic名称修改其中的topic名称），按照提示采集标定数据，数据示例如下。

| Captured LiDAR Point Cloud | Captured RGB Image |
| --- | --- |
| ![](https://cdn.nlark.com/yuque/0/2025/png/46827146/1763647731830-61bad343-6342-4470-91a1-30902d5890b3.png) | ![](https://cdn.nlark.com/yuque/0/2025/png/46827146/1763651053392-1c5c56f6-0352-482b-9939-2e2a3323b37a.png) |


**Step 2.** 运行`calibraation/solve_pnp_calibration_v2.py`求解PnP，标定结果见TODO。

**Step 3.** 将求解出的矩阵做一个逆变换，详见`calibration/compute_inverse.py`。因为PnP求解的是$ sC_{cam}=K[R|t]C_{world} $中的$ [R|t] $，即LiDAR的3D坐标到RGB图像2D坐标的pose transformation。而实验中我们的pose是以雷达odometry为准，因此我们需要的是把RGB-D图像投影成相机坐标的点云，随后再变换到以雷达的3D坐标为基准。求解出来的变换填入`config/data_config/ros/self_collected.yaml`中。

## During Experiment
+ **Terminal 1**

```bash
ssh unitree@192.168.123.164

ros2 launch realsense2_camera rs_launch.py \
  depth_module.profile:=640x480x30 \
  rgb_camera.profile:=640x480x30 \
  align_depth.enable:=true
```

+ **Terminal 2** 启动FAST-LIO后`ros2 topic list`看到名为`/Odometry`的topic，这就是我们所订阅的。

```bash
source /home/zach/repos/ws_fast_lio_ros2/install/setup.bash
ros2 launch fast_lio mapping.launch.py config_file:=mid360.yaml
```

+ **Terminal 3**

```bash
source /home/zach/repos/ws_livox/install/setup.bash
ros2 launch livox_ros_driver2 msg_MID360_launch.py
```

+ **Terminal 4 **需要给相机topic重打时间戳，详见TODO，并且该script在给定`bag-name`参数后可以用于录制rosbag。

```bash
cd ~/repos/DualMap
python restamp.py --bag-name experiment01
```

+ **Terminal 5** 运行DualMap主程序

```bash
cd ~/repos/DualMap
python -m applications.runner_ros
```

_**<u>NOTE:</u>**__** **_停止时记得先不要关闭DualMap主程序！检测到ROS topic停止后会进入end_process存储地图。

## Other Issues
### Calibration Precision
由于标定时没开雷达点云的可视化，只是将标定板尽量布满相机视野。但是标定板位置较低时的数据基本全都不可用，如下图这种。这是由于宇树G1的雷达与共视区域非常的有限，如图的角度雷达基本被机器人的机械部分遮挡，点云中的高反射率点很少或者残缺。

![](https://cdn.nlark.com/yuque/0/2025/png/46827146/1763716964715-1b712b01-6007-4fe5-a06e-f0b2f1167aa0.png)

另外，不论是否打开可视化，都可以手动筛选数据，见`calibration/debug_reflectivity_filter.py`和`calibration/extract_circle_from_rgb.py`，最终筛选出的高反射率点云如下图：  
![](https://cdn.nlark.com/yuque/0/2025/png/46827146/1763716356603-d33f06d0-423f-42a7-ac3b-9764ff16a848.png)

标定结果：TODO

### Nx board Timestamp Problem
#### Problem Description
在`runner_ros2_g1.py`中会调用一系列`callback`函数，打印已经获取数据的信息，如：

```bash
# Sync messages
self.sync = ApproximateTimeSynchronizer(
    [self.rgb_sub, self.depth_sub, self.odom_sub],
    queue_size=10,
    slop=self.cfg.sync_threshold,
)
self.sync.registerCallback(self.synced_callback)

def synced_callback(self, rgb_msg, depth_msg, odom_msg):
    """Callback for synced RGB-D-Odom input."""
    if not self._sync_received:
        self._sync_received = True
        self.logger.warning("[Main] ✓ Synchronized messages received! Starting processing...")
```

但是从未收到synchronizer的输出：

```bash
2025-11-21 21:33:25,956 - WARNING - [Main] ✓ Odom topic is receiving data
2025-11-21 21:33:27,301 - WARNING - [Main] ✓ Depth topic is receiving data
2025-11-21 21:33:27,849 - WARNING - [Main] ✓ RGB topic is receiving data
```

经检查：

```bash
2025-11-21 21:37:39,155 - WARNING - [Main] Timestamp diagnosis:
2025-11-21 21:37:39,156 - WARNING -   RGB ts:   263729.689 (age: 1763468529.47s)
2025-11-21 21:37:39,156 - WARNING -   Depth ts: 263729.689 (age: 1763468529.47s)
2025-11-21 21:37:39,156 - WARNING -   Odom ts:  1763732259.026 (age: 0.13s)
```

因为realsense启动程序是在G1的Nx板上跑的，而FAST-LIO在我本地的PC，二者的时间不同步。

#### Solution
比较朴实的解决方法是给相机的topic重新打时间戳，见`restamp.py`。也可以尝试将Nx板联网，随后在terminal检查`date`，查看联网是否能同步时间。或者在本地和Nx板上使用chrony（同样需要网络）。
