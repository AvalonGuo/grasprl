import gymnasium as gym
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.tensorboard import SummaryWriter
from collections import deque
import math
import random
import copy
import os
from grasprl.envs.grasp import GraspRobot
from grasprl.modules.ddpg import ReplayBuffer,Transition
from grasprl.modules.qnet import MULTIDISCRETE_RESNET

from tqdm import tqdm
import cv2

MAX_POSSIBLE_SAMPLES = (
    12  # Number of transitions that fits on GPU memory for one backward-call (12 for RGB-D)
)
NUMBER_ACCUMULATIONS_BEFORE_UPDATE = 1  # How often to accumulate gradients before updating
BATCH_SIZE = MAX_POSSIBLE_SAMPLES * NUMBER_ACCUMULATIONS_BEFORE_UPDATE  # Effective batch size


class DQN_Trainer(object):
    def __init__(
        self,
        learning_rate = 0.001,
        mem_size=2000,
        eps_start=1.0,
        eps_end = 0.,
        eps_decay = 200,
        seed =20,
        log_dir = "test",
        render_mode="rgb"):

        self.writer = SummaryWriter(f"grasprl/log/DQN/{log_dir}")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.env = GraspRobot(render_mode=render_mode)
        self.memory = ReplayBuffer(mem_size,simple=False)
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.steps_done = 0

        self.q_net = MULTIDISCRETE_RESNET(1).to(self.device)
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=learning_rate, weight_decay=0.00002)
        self.criterion = torch.nn.SmoothL1Loss(reduce=False).to(device=self.device)

        #seeding
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    def transform_state(self,state):
        self.depth_before = state["depth"]
        depth_img = np.asarray(state["depth"])
        depth_img = depth_img.max()-depth_img
        img_trans =  T.ToTensor()
        img_normalize = T.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
        tensor_rgb = img_trans(state["rgb"]).unsqueeze(0)
        tensor_rgb = img_normalize(tensor_rgb)
        tensor_depth = img_trans(depth_img).unsqueeze(0)
        tensor_obs =  torch.cat([tensor_rgb,tensor_depth],1)
        return tensor_obs.to(self.device)
        

    def transform_action(self,max_idx):
        max_idx = max_idx.item()
        pixel_x = max_idx % self.env.IMAGE_WIDTH
        pixel_y = max_idx // self.env.IMAGE_HEIGHT
        depth =   self.depth_before[pixel_x][pixel_y]
        action = self.env.pixel2world(1,pixel_x,pixel_y,depth)
        # print("pixelx:{},pixely:{},action:{}".format(pixel_x,pixel_y,action))
        return action
    
    def limit_action(self,action):
        action = np.clip(action,[-0.25,-0.25,self.env.TABLE_HEIGHT+0.05],[0.25,0.25,2])
        return list(action)
    
    def select_action_by_eps_random(self,state):
        self.eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(
            -1.0 * self.steps_done / self.eps_decay
        )     
        self.steps_done+=1
        self.writer.add_scalar("Epslion",self.eps_threshold,self.steps_done)
        if random.random() > self.eps_threshold:
            self.last_action = "greedy"
            with torch.no_grad():
                q_max =  self.q_net(state).argmax()
                q_max = torch.tensor([[q_max]], dtype=torch.long)
                return q_max
        else:
            self.last_action = "random"
            action = np.random.randint(low=0,high=(self.env.IMAGE_WIDTH-1)*(self.env.IMAGE_HEIGHT-1))
            return torch.tensor([[action]], dtype=torch.long)
    def select_action_by_eps(self,state):
        self.eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(
            -1.0 * self.steps_done / self.eps_decay
        )
        self.steps_done+=1
        self.writer.add_scalar("Epslion",self.eps_threshold,self.steps_done)
        if random.random() > self.eps_threshold:
            self.last_action = "greedy"
            with torch.no_grad():
                q_max =  self.q_net(state).argmax()
                q_max = torch.tensor([[q_max]], dtype=torch.long)
                return q_max
        else:
            self.last_action = "random"
            r=g=b=0
            threshold = 0
            while r==g and g==b:
                action = np.random.randint(low=0,high=(self.env.IMAGE_WIDTH-1)*(self.env.IMAGE_HEIGHT-1))
                pixel_x = action % self.env.IMAGE_WIDTH
                pixel_y = action // self.env.IMAGE_HEIGHT
                r,g,b = self.env.observation["rgb"][pixel_x][pixel_y]
                threshold+=1
                if(threshold==10):
                    break
            return torch.tensor([[action]], dtype=torch.long)
    
    def select_action_by_instruction(self,state):
        self.eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(
            -1.0 * self.steps_done / self.eps_decay
        )
        self.steps_done+=1
        self.writer.add_scalar("Epslion",self.eps_threshold,self.steps_done)
        if random.random() > self.eps_threshold:
            self.last_action = "greedy"
            with torch.no_grad():
                q_max =  self.q_net(state).argmax()
                q_max = torch.tensor([[q_max]], dtype=torch.long)
                return q_max
        else:
            self.last_action = "instruction"
            action = None
            for obj_name in self.env.target_objects:    
                wx,wy,wz = self.env.get_body_com(obj_name)
                # print("obj_name:{},pos:{}".format(obj_name,[wx,wy,wz]))
                if -0.224<=wx<=0.224 and -0.224<=wy<=0.224 and wz>=0.9:
                    px,py = self.env.world2pixel(cam_id=1,x=wx,y=wy,z=wz)
                    action = py*self.env.IMAGE_WIDTH + px
                    # print("action:{},px/py:{}/{},pos:[{},{},{}]".format(action,px,py,wx,wy,wz))
                    break 

            return torch.tensor([[action]], dtype=torch.long)
    def learn(self):
        if len(self.memory) < 2 * BATCH_SIZE:
            print("Filling the replay buffer ...")
            return
        transitions = self.memory.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))
        # Gradient accumulation to bypass GPU memory restrictions
        for i in range(NUMBER_ACCUMULATIONS_BEFORE_UPDATE):
            start_idx = i * MAX_POSSIBLE_SAMPLES
            end_idx = (i + 1) * MAX_POSSIBLE_SAMPLES
            state_batch = torch.cat(batch.state[start_idx:end_idx]).to(self.device)
            action_batch = torch.cat(batch.action[start_idx:end_idx]).to(self.device)
            reward_batch = torch.cat(batch.reward[start_idx:end_idx]).to(self.device)
            # next_state_batch = torch.cat(batch.next_state[start_idx:end_idx]).to(self.device)
            # Compute the target Q value
            q_pred = self.q_net(state_batch).view(MAX_POSSIBLE_SAMPLES, -1).gather(1, action_batch)
            q_expected = reward_batch.float()

            loss = F.binary_cross_entropy(q_pred, q_expected) / NUMBER_ACCUMULATIONS_BEFORE_UPDATE
            loss.backward()

            self.optimizer.step()
            self.optimizer.zero_grad()

            self.writer.add_scalar("losses",loss,self.steps_done)


    def predict(self,state,show=False):
        with torch.no_grad():
            affordance = self.q_net(state)     
            q_max =  affordance.argmax()
            action = self.transform_action(q_max)
            action = self.limit_action(action)
            if show:
                cv2.imshow("affordance",affordance.cpu().numpy())
                cv2.waitKey(0)
        return action

    def save(self,path_name,filename):
        if not os.path.exists(path_name):
            os.mkdir(path_name)
        torch.save(self.q_net.state_dict(), path_name+filename + "_qnet")

    def load(self, filename):
        self.q_net.load_state_dict(torch.load(f=filename + "_qnet",map_location=self.device))


    def count(self,greedy,random_num):
        self.eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(
            -1.0 * self.steps_done / self.eps_decay
        )
        self.steps_done+=1
        if random.random() > self.eps_threshold:
            greedy+=1
        else:
            self.last_action = "random"
            random_num+=1
        return greedy,random_num
def main():
    max_iter = 2500
    grasp_success = 0
    loop = tqdm(range(1,max_iter+1))

    trainer = DQN_Trainer(log_dir="resnet_dqn_insne",render_mode="rgb")

    state = trainer.env.reset_without_random()
    state = trainer.transform_state(state)
    for i_iter in loop:
        max_idx = trainer.select_action_by_instruction(state)
        action = trainer.transform_action(max_idx)
        action = trainer.limit_action(action)
        next_state, reward, done,info = trainer.env.step(action)
        loop.set_description(f"iter [{i_iter}]/[{max_iter}]")
        loop.set_postfix(grasp_info=info['grasp'],reward=reward,action=trainer.last_action)
        if info["grasp"] == "Success":
            grasp_success+=1
        if done:
            trainer.env.reset_without_random()
        trainer.writer.add_scalar("Grasping performance(Success rate)",grasp_success/max_iter,trainer.steps_done)
        reward = torch.tensor([[reward]],dtype=torch.float32)
        next_state = trainer.transform_state(next_state)
        trainer.memory.push(state,max_idx, next_state,reward)
        state = next_state
        trainer.learn()


    trainer.save(path_name="grasprl/trained/resnet/resnet",filename="insne")
    trainer.env.close()
    trainer.writer.close()
if __name__ == "__main__":
    main()