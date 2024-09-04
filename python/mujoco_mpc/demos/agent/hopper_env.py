from mujoco_mpc import agent as agent_lib

from buffer import Buffer
import os

class Hopper:
    def __init__(self):
        model_path = (
            pathlib.Path(__file__).parent.parent.parent
            / "../../build/mjpc/tasks/hopper/task.xml"
        )
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(model)
        self.agent = agent_lib.Agent(task_id="Hopper", model=model)

        self._max_episode_steps = 1000

    def reset(self):
        mujoco.mj_resetData(model, data)
    
    def step(self):
        self.agent.set_state(
            time=self.data.time, # time
            qpos=self.data.qpos, # position
            qvel=self.data.qvel, # velocity
            act=self.data.act, # control
            mocap_pos=self.data.mocap_pos, # mocap position
            mocap_quat=self.data.mocap_quat, # mocap quaternion
            userdata=self.data.userdata, # user data
        )

        # run planner for num_steps
        num_steps = 10
        for _ in range(num_steps):
            self.agent.planner_step()
        
        # set ctrl from agent policy
        self.data.ctrl = agent.get_action()

        


        


