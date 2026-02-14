import numpy as np

def invert_extrinsics(T_cam_lidar: np.ndarray) -> np.ndarray:
    """
    Invert a 4x4 extrinsic matrix.

    Input:
        T_cam_lidar: 4x4 matrix from lidar frame to camera frame:
                     X_cam = R * X_lidar + t

    Output:
        T_lidar_camera: 4x4 matrix from camera frame to lidar frame:
                        X_lidar = R_inv * X_cam + t_inv
    """
    assert T_cam_lidar.shape == (4, 4)
    R = T_cam_lidar[:3, :3]
    t = T_cam_lidar[:3, 3]

    R_inv = R.T
    t_inv = -R_inv @ t

    T_lidar_camera = np.eye(4)
    T_lidar_camera[:3, :3] = R_inv
    T_lidar_camera[:3, 3] = t_inv

    return T_lidar_camera


if __name__ == "__main__":
    T_cam_lidar = np.array([
        [0.019422186628740068,  0.9991918412120033, -0.035191520596943415,  0.0538674537685072],
        [-0.6547524391913779,   0.039311771684021835, 0.7548203945177978, -0.12438417168231966],
        [0.7555938208055298,    0.008381471376198324, 0.6549868158200366, -0.048231555223464966],
        [0.0,                    0.0,                  0.0,                 1.0]
    ], dtype=np.float64)

    T_lidar_camera = invert_extrinsics(T_cam_lidar)
    print("T_lidar_camera =\n", T_lidar_camera)
