from grasprl.trainer.dqn_baseline import DQN_Trainer
from grasprl.envs.grasp import GraspRobot
from grasprl.utils.transform_utils import quat2mat
import numpy as np
from tqdm import tqdm


def count():
    max_iter = 2500
    random = greedy = 0
    loop = tqdm(range(1,max_iter+1))
    trainer = DQN_Trainer(render_mode="rgb")
    for i_iter in loop:
        greedy,random  = trainer.count(greedy=greedy,random_num=random)
    print(greedy,random)

if __name__ == "__main__":
    count()
