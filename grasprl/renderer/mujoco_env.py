from os import path
import time
from typing import Any, Dict, Optional
import numpy as np
import cv2
import gymnasium as gym
from dm_control import mjcf
from dm_control.mujoco import engine
import mujoco
import mujoco.viewer
from grasprl.utils.mujoco_utils import MujocoModelNames
from grasprl.utils.transform_utils import quat2mat
import random




class MujocoPhyEnv(gym.Env):
    """Superclass for MuJoCo environments."""

    def __init__(
        self,
        model_path,
        frame_skip,
        render_mode: Optional[str] = None,
    ):
        self.TABLE_HEIGHT = 0.95
        self.IMAGE_WIDTH = 224
        self.IMAGE_HEIGHT = 224
        if model_path.startswith(".") or model_path.startswith("/"):
            self.fullpath = model_path
        self.mjcf_model = mjcf.from_path(self.fullpath)
        self.physics = mjcf.Physics.from_mjcf_model(self.mjcf_model)
        self.viewer = None
        self.image_renderer = mujoco.Renderer(self.physics.model.ptr,width=self.IMAGE_WIDTH,height=self.IMAGE_HEIGHT)
        self._step_start = None
        self._timestep = self.physics.model.opt.timestep
        self.arm_joints = self.mjcf_model.find_all('joint')[:6]
        self.eef_site = self.mjcf_model.find('site','eef_site')
        self.gripper = self.mjcf_model.find('joint','left_outer_knuckle_joint')
        self.model = self.physics.model.ptr
        self.data = self.physics.data.ptr
        self.frame_skip = frame_skip
        self.render_mode = render_mode
        self.model_names = MujocoModelNames(self.model)
        self.cam_matrix = None
        self.cam_init = False
 

    def _reset_simulation(self):
        mujoco.mj_resetData(self.model, self.data)


    def step_mujoco_simulation(self, n_frames=1):
        self.physics.step(n_frames)
        self.render()

    def render(self):
        """
        Renders the current frame and updates the viewer if the render mode is set to "human".
        """
        if self.viewer is None and self.render_mode == "human":
            self.viewer = mujoco.viewer.launch_passive(
                self.physics.model.ptr,
                self.physics.data.ptr,
            )
        if self._step_start is None and self.render_mode == "human":
            # initialize step timer
            self._step_start = time.time()

        if self.render_mode == "human":
            self.viewer.sync()
            time_until_next_step = self._timestep - (time.time() - self._step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
            self._step_start = time.time()

    def reset(self):
        
        with self.physics.reset_context():
                self._reset_simulation()
                self.physics.bind(self.arm_joints).qpos = [
                    1.5707,
                    -1.5707,
                    1.5707,
                    -1.5707,
                    -1.5707,
                    0.0,
                ]


    def close(self):
        if self.viewer is not None:
            self.viewer.close()

    def get_body_com(self, body_name):
        return self.data.body(body_name).xpos
    
    def create_camera_data(self, width, height, camera):
        """
        Initializes all camera parameters that only need to be calculated once.
        """
        cam_id = self.model_names.camera_name2id[camera]
        # Get field of view
        fovy = self.model.cam_fovy[cam_id]
        # Calculate focal length
        f = 0.5 * height / np.tan(fovy * np.pi / 360)
        # Construct camera matrix
        self.cam_matrix = np.array(((f, 0, width / 2), (0, f, height / 2), (0, 0, 1)))
        # Rotation of camera in world coordinates
        self.cam_rot_mat = self.model.cam_mat0[cam_id]
        self.cam_rot_mat = np.reshape(self.cam_rot_mat, (3, 3))
        # Position of camera in world coordinates
        self.cam_pos = self.model.cam_pos0[cam_id]
        self.cam_init = True

    def get_image_data(
        self,
        cam_name="top_down",
        depth = False,
        show = False
        ):
        self.image_renderer.enable_depth_rendering()
        self.image_renderer.update_scene(self.physics.data.ptr,camera=cam_name)
        depth_data = self.image_renderer.render()
        self.image_renderer.disable_depth_rendering()
        rgb_data = self.image_renderer.render()
        rgb_data = cv2.cvtColor(rgb_data,cv2.COLOR_BGR2RGB)
        if show:
            cv2.imshow(cam_name,rgb_data)
            cv2.imshow("depth",depth_data)
            cv2.waitKey(1)
        if depth:
            return np.array(rgb_data), np.array(depth_data)



    def depth_2_meters(self, depth):
        """
        Converts the depth array delivered by MuJoCo (values between 0 and 1) into actual m values.

        Args:
            depth: The depth array to be converted.
        """
        extend = self.model.stat.extent
        near = self.model.vis.map.znear * extend
        far = self.model.vis.map.zfar * extend
        return near / (1 - depth * (1 - near / far))
    
    def get_pos_w(self,x,y):

        depth = self.observation["depth"][y][x]
        coordinates = self.pixel_2_world(
            pixel_x=x, pixel_y=y, depth=depth, height=self.IMAGE_HEIGHT, width=self.IMAGE_WIDTH
        )
        return coordinates
    


    def pixel2world(self,cam_id,pixel_x,pixel_y,depth):
        depth = -depth
        camera = engine.Camera(self.physics,width=self.IMAGE_WIDTH,height=self.IMAGE_HEIGHT,
                               camera_id=cam_id)
        camera_matrix = np.linalg.pinv(camera.matrix)
        world_position = camera_matrix.dot(np.array([pixel_x,pixel_y,1.0]))*depth
        wx,wy = world_position[:2]
        return [wx,wy,1.16]

    def world2pixel(self,cam_id,x,y,z):
        camera = engine.Camera(self.physics, width=self.IMAGE_WIDTH, height=self.IMAGE_HEIGHT,
                            camera_id=cam_id)
        camera_matrix = camera.matrix  # Get camera matrix.
        xs, ys, s = camera_matrix.dot(np.array([x, y, z, 1.0]))
        return [int(np.round(xs/s)), int(np.round(ys/s))]
    def get_ee_pos(self):
        return  self.physics.bind(self.eef_site).xpos
    

    def set_body_pos(self,body_name):
        body = self.mjcf_model.find('body',body_name)
        new_pos = [random.uniform(a=-0.224,b=0.224) for i in range(2)]
        new_pos.append(1.1)
        self.physics.bind(body).pos = new_pos
