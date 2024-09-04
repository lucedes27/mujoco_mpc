# Copyright 2022 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# %%
import matplotlib.pyplot as plt
import mediapy as media
import mujoco
import numpy as np
import pathlib
from mujoco.viewer import launch_passive

# set current directory: mujoco_mpc/python/mujoco_mpc
from mujoco_mpc import agent as agent_lib

from buffer import Buffer
import os

# %matplotlib inline

# %%
# model
model_path = (
    pathlib.Path(__file__).parent.parent.parent
    / "../../build/mjpc/tasks/ant/task.xml"
)
model = mujoco.MjModel.from_xml_path(str(model_path))

# data
data = mujoco.MjData(model)

# renderer
# renderer = mujoco.Renderer(model)

# %%
# agent
agent = agent_lib.Agent(task_id="Ant", model=model)

# weights
print("Cost weights:", agent.get_cost_weights())

# # parameters
print("Parameters:", agent.get_task_parameters())

# %%
# rollout horizon
T = 10001

# trajectories
qpos = np.zeros((model.nq, T))
qvel = np.zeros((model.nv, T))
ctrl = np.zeros((model.nu, T - 1))
comPosX = np.zeros(T)
time = np.zeros(T)

# costs
cost_total = np.zeros(T - 1)
cost_terms = np.zeros((len(agent.get_cost_term_values()), T - 1))

# rollout
mujoco.mj_resetData(model, data)

# cache initial state
qpos[:, 0] = data.qpos
qvel[:, 0] = data.qvel

# get com pos
mujoco.mj_comPos(model, data)
comPosX[0] = data.subtree_com[0][0]

# frames
frames = []
FPS = 1.0 / model.opt.timestep

# init buffer (the minus 2 is since root x and root y pos is ignored in gymnasium)
BUFFER_SIZE=10000
buffer = Buffer(BUFFER_SIZE, [qpos.shape[0] + qvel.shape[0] - 2], [ctrl.shape[0]], 'cpu')

# for reward stuff
# reward_flags = np.ones(100000, dtype=bool)
# max_level = 0

total_reward = 0
episode_reward = 0
terminated = False
timestep_count = 0
num_episodes = 0

done_flag = False

# with launch_passive(model, data) as viewer:

# simulate
for t in range(T - 1):
  print("t = ", t)

  # set planner state
  agent.set_state(
      time=data.time, # time
      qpos=data.qpos, # position
      qvel=data.qvel, # velocity
      act=data.act, # control
      mocap_pos=data.mocap_pos, # mocap position
      mocap_quat=data.mocap_quat, # mocap quaternion
      userdata=data.userdata, # user data
  )

  # run planner for num_steps
  num_steps = 10
  for _ in range(num_steps):
    agent.planner_step()

  # set ctrl from agent policy
  data.ctrl = agent.get_action()
  ctrl[:, t] = data.ctrl

  # get costs
  cost_total[t] = agent.get_total_cost()
  for i, c in enumerate(agent.get_cost_term_values().items()):
    cost_terms[i, t] = c[1]

  # step
  mujoco.mj_step(model, data)

  # cache
  qpos[:, t + 1] = data.qpos
  qvel[:, t + 1] = data.qvel
  mujoco.mj_comPos(model, data)
  comPosX[t + 1] = data.subtree_com[0][0]

  # Compute dense rewards
  posbefore = comPosX[t]
  posafter = comPosX[t + 1]
  alive_bonus = 1.0
  reward = (posafter - posbefore) / 0.01
  reward += alive_bonus if not terminated else 0
  reward -= 0.5 * np.square(ctrl[:, t]).sum()
  done = (not np.isfinite(np.concatenate([qpos[2:, t + 1], qvel[:, t + 1]])).all()) or (qpos[2, t + 1] < 0.2 or qpos[2, t + 1] > 1.0)
  episode_reward += reward
  print("Terminated: ", done)
  print("Dense Reward: ", reward)

  buffer.append(np.concatenate([qpos[2:, t], qvel[:, t]]), ctrl[:, t], reward, done, np.concatenate([qpos[2:, t + 1], qvel[:, t + 1]]))

  if done:
    num_episodes += 1
    total_reward += episode_reward
    mujoco.mj_resetData(model, data)
    qpos[:, t + 1] = data.qpos
    qvel[:, t + 1] = data.qvel
    comPosX[t + 1] = data.subtree_com[0][0]
    episode_reward = 0

  # Compute rewards using RL reward function
  # new sparse reward
  # sparse_threshold = 1
  # level = int((comPosX[t + 1] - comPosX[0]) / sparse_threshold)
  # if level >= 1 and reward_flags[level]:
  #   reward = 1.
  #   reward_flags[level] = False
  # else:
  #   reward = 0.
  # if level > max_level:
  #   max_level = level
  # notdone = np.isfinite(np.concatenate([qpos[:, t + 1], qvel[:, t + 1]])).all() and qpos[2, t + 1] >= 0.2 and qpos[2, t + 1] <= 1.0
  # done = not notdone
  # print("Terminated: ", done)
  # print("x pos: ", qpos[0, t + 1])
  # buffer.append(np.concatenate([qpos[2:, t], qvel[:, t]]), ctrl[:, t], reward, done, np.concatenate([qpos[2:, t + 1], qvel[:, t + 1]]))

  timestep_count += 1
# render and save frames
# renderer.update_scene(data)
# pixels = renderer.render()
# frames.append(pixels)
    # viewer.sync()

print("Episodic Reward:", total_reward / num_episodes if num_episodes != 0 else episode_reward)
print("Num Episodes: ", num_episodes)

# save buffer
buffer.save(os.path.join(
  'buffers',
  'ant.pth'
))

# reset
agent.reset()

  

# display video
# print("Displaying video")
# SLOWDOWN = 0.5
# media.show_video(frames, fps=SLOWDOWN * FPS)

# %%
# plot position
# fig = plt.figure()

# plt.plot(time, qpos[0, :], label="q0", color="blue")
# plt.plot(time, qpos[1, :], label="q1", color="orange")

# plt.legend()
# plt.xlabel("Time (s)")
# plt.ylabel("Configuration")

# # %%
# # plot velocity
# fig = plt.figure()

# plt.plot(time, qvel[0, :], label="v0", color="blue")
# plt.plot(time, qvel[1, :], label="v1", color="orange")

# plt.legend()
# plt.xlabel("Time (s)")
# plt.ylabel("Velocity")

# # %%
# # plot control
# fig = plt.figure()

# plt.plot(time[:-1], ctrl[0, :], color="blue")

# plt.xlabel("Time (s)")
# plt.ylabel("Control")

# # %%
# # plot costs
# fig = plt.figure()

# for i, c in enumerate(agent.get_cost_term_values().items()):
#   plt.plot(time[:-1], cost_terms[i, :], label=c[0])

# plt.plot(time[:-1], cost_total, label="Total (weighted)", color="black")

# plt.legend()
# plt.xlabel("Time (s)")
# plt.ylabel("Costs")

# plt.savefig('graph.png')