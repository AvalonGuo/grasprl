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


def test_cooconvert():
    import cv2
    env = GraspRobot(render_mode = "human")
    env.reset_without_random()
    env.before_grasp()
    color_img = env.observation["rgb"]
    depth_img = np.asarray(env.observation["depth"])
    pi_x,pi_y = env.world2pixel(cam_id=1,x=0.1,y=0,z=0.98)
    world_coor = env.pixel2world(cam_id=1,pixel_x=pi_x,pixel_y=pi_y,depth=depth_img[pi_x][pi_y])
    print(pi_x,pi_y,world_coor)
    cv2.imshow("rgb",color_img)
    cv2.imshow("depth",depth_img)
    cv2.waitKey(0)

if __name__ == "__main__":

    # test_DQN(model_path="grasprl/trained/resnet/2p5k")
    # test_reset_object()
    test_cooconvert()
    # test_DQN(model_path="grasprl/trained/resnet/pexresnet")
    # test_reset_object()

