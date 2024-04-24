from grasprl.trainer.dqn_baseline import DQN_Trainer
from grasprl.envs.grasp import GraspRobot
from grasprl.utils.transform_utils import quat2mat
import numpy as np
from tqdm import tqdm


workspace_limits = np.asarray([[-0.224, 0.224], [-0.224, 0.224], [0.95,1.6]]) # Cols: min max, Rows: x y z (define workspace limits in robot coordinates)


def test_DQN(model_path):
    max_iter = 120
    grasp_success = 0
    grasp_failed = 0
    loop = tqdm(range(1,max_iter+1))
    model = DQN_Trainer(render_mode="human",seed=20)
    model.load(filename=model_path)
    state = model.env.reset_without_random()
    state = model.transform_state(state)
    for i_iter in loop:
        action = model.predict(state,show=False)
        # next_state,reward,terminated,info  = model.env.step(action)
        next_state,reward,terminated,info  = model.env.step_test(action,grasp_failed)
        state = model.transform_state(next_state)
        if info['grasp'] == "Success":
            grasp_success+=1
            grasp_failed=0
        else:
            grasp_failed+=1
        if terminated:
            state = model.env.reset()
            state = model.transform_state(state)
        loop.set_postfix(grasp_info=info['grasp'])
    print("success rate:{}%".format((grasp_success/max_iter)*100))


def test_grasp_policy(model_path):
    scence = 1
    total_grasp = 1
    grasp_success = 0
    completed_scence = 0
    model = DQN_Trainer(render_mode="human",seed=10)
    model.load(filename=model_path)
    state = model.env.reset_without_random()
    state = model.transform_state(state)
    while scence != 13:
        action = model.predict(state,show=False)
        # next_state,reward,terminated,info  = model.env.step(action)
        next_state,reward,terminated,info  = model.env.step_test(action,total_grasp)
        print(info)
        total_grasp += 1
        state = model.transform_state(next_state)
        if info['grasp'] == "Success":
            grasp_success+=1
        if info['completion'] == "Success":
            completed_scence+=1
        if terminated:
            state = model.env.reset()
            scence+=1
            state = model.transform_state(state)
    print("success rate:{}%".format((grasp_success/total_grasp)*100))
    print("Completion rate:{}%".format((completed_scence/(scence-1))*100))
    print(total_grasp,completed_scence)


def test_cooconvert():
    import cv2
    env = GraspRobot(render_mode = "rgb")
    env.reset_without_random()
    env.before_grasp()
    color_img = env.observation["rgb"]
    depth_img = np.asarray(env.observation["depth"])
    targets = env.target_objects
    for obj_name in targets:
        pos = env.get_body_com(obj_name)
        wx,wy,wz = pos
        if -0.224<=wx<=0.224:
            print("True")
        px,py = env.world2pixel(cam_id=1,x=wx,y=wy,z=wz)
        depth = depth_img[px][py]
        world_coor = env.pixel2world(cam_id=1,pixel_x=px,pixel_y=py,depth=depth)
        print("target:{},pos:{},pixel:{}/{},world_coor:{}".format(obj_name,pos,px,py,world_coor))
    cv2.imshow("rgb",color_img)
    cv2.imshow("depth",depth_img)
    cv2.waitKey(0)
if __name__ == "__main__":

    # test_reset_object()
    # test_cooconvert()
    # test_DQN(model_path="grasprl/trained/resnet/resnetins")
    test_grasp_policy(model_path="grasprl/trained/resnet/pexresnet")
    # test_reset_object()

