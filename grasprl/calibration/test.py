import mujoco
from grasprl.utils.mujoco_utils import MujocoModelNames
import random
model = mujoco.MjModel.from_xml_path("grasprl/worlds/grasp.xml")
data = mujoco.MjData(model)
model_names = MujocoModelNames(model)
box_id = model_names.body_name2id["box_1"]
data.set_body_xpos("box_1", [random.uniform(a=2.5,b=2.5) for i in range(3)])
mujoco.mj_step(model, data)
print(data.xpos)