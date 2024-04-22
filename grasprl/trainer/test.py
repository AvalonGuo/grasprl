from grasprl.envs.grasp import GraspRobot
import cv2
# env = GraspRobot(render_mode="human")
# env.reset()
# env.before_grasp()
# cam_id = env.model_names.camera_name2id["eyeinhand"]
# rgb = env.observation["rgb"]
# depth = env.observation["depth"]
# rgb = cv2.cvtColor(rgb,cv2.COLOR_BGR2RGB)
# cv2.imshow("rgb",rgb)
# cv2.imshow("depth",depth)
# depthxy = depth[112][112]
# print(depthxy)
# env.pixel_2_worldXY(112,112,depthxy)

# # cv2.imwrite("grasprl/demo/rgb.jpg",rgb)
# cv2.waitKey(0)
    
import numpy as np
action= [1,2]
action3 = np.clip(action,[0,0],[0,3])
print(action3,type(action))


