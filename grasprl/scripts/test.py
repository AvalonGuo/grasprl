from grasprl.trainer.dqn_baseline import DQN_Trainer
from grasprl.envs.grasp import GraspRobot
from grasprl.utils.transform_utils import quat2mat
import numpy as np


workspace_limits = np.asarray([[-0.224, 0.224], [-0.224, 0.224], [0.95,1.6]]) # Cols: min max, Rows: x y z (define workspace limits in robot coordinates)


def test_DQN(model_path):
    model = DQN_Trainer(render_mode="human")
    model.load(filename=model_path)
    state = model.env.reset()
    state = model.transform_state(state)
    for i in range(10):
        
        action = model.predict(state,show=True)
        next_state,reward,terminated,info  = model.env.step(action)
        state = model.transform_state(next_state)
        print(info)
        if terminated:
            state = model.env.reset()
            state = model.transform_state(state)

def get_pointcloud(color_img, depth_img, camera_intrinsics):

    # Get depth image size
    im_h = depth_img.shape[0]
    im_w = depth_img.shape[1]

    # Project depth into 3D point cloud in camera coordinates
    pix_x,pix_y = np.meshgrid(np.linspace(0,im_w-1,im_w), np.linspace(0,im_h-1,im_h))
    cam_pts_x = np.multiply(pix_x-camera_intrinsics[0][2],depth_img/camera_intrinsics[0][0])
    cam_pts_y = np.multiply(pix_y-camera_intrinsics[1][2],depth_img/camera_intrinsics[1][1])
    cam_pts_z = depth_img.copy()
    cam_pts_x.shape = (im_h*im_w,1)
    cam_pts_y.shape = (im_h*im_w,1)
    cam_pts_z.shape = (im_h*im_w,1)

    # Reshape image into colors for 3D point cloud
    rgb_pts_r = color_img[:,:,0]
    rgb_pts_g = color_img[:,:,1]
    rgb_pts_b = color_img[:,:,2]
    rgb_pts_r.shape = (im_h*im_w,1)
    rgb_pts_g.shape = (im_h*im_w,1)
    rgb_pts_b.shape = (im_h*im_w,1)

    cam_pts = np.concatenate((cam_pts_x, cam_pts_y, cam_pts_z), axis=1)
    rgb_pts = np.concatenate((rgb_pts_r, rgb_pts_g, rgb_pts_b), axis=1)

    return cam_pts, rgb_pts


def get_heightmap(color_img, depth_img, cam_intrinsics, cam_pose, workspace_limits, heightmap_resolution):
    # Compute heightmap size
    heightmap_size = np.round(((workspace_limits[1][1] - workspace_limits[1][0])/heightmap_resolution, (workspace_limits[0][1] - workspace_limits[0][0])/heightmap_resolution)).astype(int)    
    surface_pts, color_pts = get_pointcloud(color_img, depth_img, cam_intrinsics)
    print(surface_pts[:40],color_pts[:40])
    surface_pts = np.transpose(np.dot(cam_pose[0:3,0:3],np.transpose(surface_pts)) + np.tile(cam_pose[0:3,3:],(1,surface_pts.shape[0])))
    # Transform 3D point cloud from camera coordinates to robot coordinates
    surface_pts = np.transpose(np.dot(cam_pose[0:3,0:3],np.transpose(surface_pts)) + np.tile(cam_pose[0:3,3:],(1,surface_pts.shape[0])))
    # Sort surface points by z value
    sort_z_ind = np.argsort(surface_pts[:,2])
    surface_pts = surface_pts[sort_z_ind]
    color_pts = color_pts[sort_z_ind]
    # Filter out surface points outside heightmap boundaries
    heightmap_valid_ind = np.logical_and(np.logical_and(np.logical_and(np.logical_and(surface_pts[:,0] >= workspace_limits[0][0], surface_pts[:,0] < workspace_limits[0][1]), surface_pts[:,1] >= workspace_limits[1][0]), surface_pts[:,1] < workspace_limits[1][1]), surface_pts[:,2] < workspace_limits[2][1])
    surface_pts = surface_pts[heightmap_valid_ind]
    color_pts = color_pts[heightmap_valid_ind]
    # Create orthographic top-down-view RGB-D heightmaps
    color_heightmap_r = np.zeros((heightmap_size[0], heightmap_size[1], 1), dtype=np.uint8)
    color_heightmap_g = np.zeros((heightmap_size[0], heightmap_size[1], 1), dtype=np.uint8)
    color_heightmap_b = np.zeros((heightmap_size[0], heightmap_size[1], 1), dtype=np.uint8)
    depth_heightmap = np.zeros(heightmap_size)
    heightmap_pix_x = np.floor((surface_pts[:,0] - workspace_limits[0][0])/heightmap_resolution).astype(int)
    heightmap_pix_y = np.floor((surface_pts[:,1] - workspace_limits[1][0])/heightmap_resolution).astype(int)
    color_heightmap_r[heightmap_pix_y,heightmap_pix_x] = color_pts[:,[0]]
    color_heightmap_g[heightmap_pix_y,heightmap_pix_x] = color_pts[:,[1]]
    color_heightmap_b[heightmap_pix_y,heightmap_pix_x] = color_pts[:,[2]]
    color_heightmap = np.concatenate((color_heightmap_r, color_heightmap_g, color_heightmap_b), axis=2)
    depth_heightmap[heightmap_pix_y,heightmap_pix_x] = surface_pts[:,2]
    z_bottom = workspace_limits[2][0]
    depth_heightmap = depth_heightmap - z_bottom
    depth_heightmap[depth_heightmap < 0] = 0
    depth_heightmap[depth_heightmap == -z_bottom] = np.nan

    return color_heightmap, depth_heightmap
def get_camera_intrinsic(model,cam_id):
    focal_length = model.cam_intrinsic[cam_id][0]
    cx = model.cam_intrinsic[cam_id][2]
    cy = model.cam_intrinsic[cam_id][3]
    cam_intrinsic =  np.array([[focal_length,0,cx],
                               [0,focal_length,cy],
                               [0,0,1]])
    return cam_intrinsic

def get_cam_pose(model,data,cam_id):
    cam_trans = np.eye(4,4)
    cam_trans[0:3,3] = np.asarray(data.cam_xpos[cam_id])
    cam_rotm = np.eye(4,4)
    cam_rotm[0:3,0:3] = np.linalg.inv(quat2mat(model.cam_quat[cam_id]))
    cam_pose = np.dot(cam_trans, cam_rotm) 
    return cam_pose
def test_reset_object():
    import cv2
    env = GraspRobot(render_mode = "rgb")
    env.reset()
    env.before_grasp()
    color_img = env.observation["rgb"]
    depth_img = env.observation["depth"]
    cam_intrinic = get_camera_intrinsic(env.model,1)
    cam_pose = get_cam_pose(env.model,env.data,1)
    color_heightmap, depth_heightmap = get_heightmap(
                                        color_img=color_img,
                                        depth_img=depth_img,
                                        cam_intrinsics=cam_intrinic,
                                        cam_pose=cam_pose,
                                        workspace_limits=workspace_limits,
                                        heightmap_resolution=0.002)
    # cv2.imshow("color_heightmap",color_heightmap)
    # cv2.imshow("depth_heightmap",depth_heightmap)
    # print(color_heightmap.shape,depth_heightmap.shape)

    cv2.imshow("rgb",color_img)
    cv2.imshow("depth",depth_img)

    cv2.waitKey(0)


def test_cooconvert():
    import cv2
    env = GraspRobot(render_mode = "human")
    env.reset_without_random()
    env.before_grasp()
    color_img = env.observation["rgb"]
    depth_img = np.asarray(env.observation["depth"])
    pi_x,pi_y = env.world2pixel(cam_id=1,x=0.1,y=0,z=0.98)
    print(type(pi_x))
    world_coor = env.pixel_2_worldXY(cam_id=1,pixel_x=pi_x,pixel_y=pi_y,depth=depth_img[pi_x][pi_y])
    print(pi_x,pi_y,world_coor)
    cv2.imshow("rgb",color_img)
    cv2.imshow("depth",depth_img)
    cv2.waitKey(0)

if __name__ == "__main__":

    # test_DQN(model_path="grasprl/trained/resnet/2p5k")
    # test_reset_object()
    # test_cooconvert()
    test_DQN(model_path="grasprl/trained/resnet/pexresnet")
    # test_reset_object()

