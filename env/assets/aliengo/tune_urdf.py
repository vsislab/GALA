import pybullet as p
import pybullet_data as pd
import time
from math import pi, sin
import numpy as np

p.connect(p.GUI)
p.setAdditionalSearchPath(pd.getDataPath())
# p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
# p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
plane = p.loadURDF("plane.urdf")
p.setGravity(0, 0, -9.8)
p.setTimeStep(1. / 1000)
quadruped = p.loadURDF("urdf/aliengo.urdf", [0, 0, 0.55], [0, 0, 0, 1], flags=p.URDF_USE_SELF_COLLISION, useFixedBase=True)

LEG_NAMES = ['RF', 'LF', 'RR', 'LR']
LIMB_NAMES = ['hip_joint', 'thigh_joint', 'calf_joint']
MOTOR_NAMES = [leg + '_' + limb for leg in LEG_NAMES for limb in LIMB_NAMES]
MOTOR_NUM = len(MOTOR_NAMES)
LEG_NUM = FOOT_NUM = len(LEG_NAMES)

MOTOR_POSITION_HIGH = np.array([1.222, pi, -0.646]).repeat(4)
MOTOR_POSITION_LOW = np.array([-1.222, -pi, -2.775]).repeat(4)

STAND_MOTOR_POSITION_REFERENCE = np.array([0., 0.8, -1.5]).repeat(4)
MOTOR_FORCE_LIMIT = np.array([44.4] * MOTOR_NUM)
MOTOR_VELOCITY_LIMIT = np.array([10.6] * MOTOR_NUM)


def record_id_from_urdf():
    motor_dict = {}
    numJoints = p.getNumJoints(quadruped)
    print('joint number:', numJoints)
    for i in range(numJoints):
        joint_info = p.getJointInfo(quadruped, i)
        motor_dict[joint_info[1].decode('UTF-8')] = joint_info[0]
    motor_id_list = [motor_dict[motor_name] for motor_name in MOTOR_NAMES]
    return motor_id_list


pos = 0
motor_i = 0
dt = 1. / 1000.
p.setTimeStep(dt)
p.stepSimulation()
p.getCameraImage(480, 320)
p.setRealTimeSimulation(0)
p.resetDebugVisualizerCamera(0.8, 180, -30, [0, 0, 0.8])
i = 0
force = 10
count = 0
motor_id_list = record_id_from_urdf()
while True:
    # pos += 0.01
    #pos = 0.5 * sin(2 * pi * dt * count)
    count += 1
    for j in range(3):
        for i in range(4):
            p.setJointMotorControl2(quadruped, motor_id_list[3 * i + j], p.POSITION_CONTROL, STAND_MOTOR_POSITION_REFERENCE[4 * j] + pos, force=force)
            print(3 * i + j, p.getJointState(quadruped, motor_id_list[3 * i + j])[:2])
        print()
    p.stepSimulation()
    time.sleep(0.01)
    # p.setJointMotorControl2(quadruped, motor_id_list[i], p.POSITION_CONTROL, pos, force=force)
    # p.setJointMotorControl2(quadruped, motor_id_list[i], p.POSITION_CONTROL, pos, force=force)
    # p.stepSimulation()
    # time.sleep(0.01)
    # position, orientation = p.getBasePositionAndOrientation(quadruped)
    # if pos >= pi:
    #     if i == MOTOR_NUM:
    #         break
    #     pos = 0
    #     i += 1
    #     motor_i = motor_id_list[i]
    # p.resetDebugVisualizerCamera(0.8, 180 * (i // 3) + 180, -30, [0, 0, 0.8])
