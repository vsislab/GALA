# -*- coding: utf-8 -*-
import enum
import numpy as np
from env.tasks.common import register
from env.tasks import RecoveryTask, RollingTask
from env.tasks import ForwardLocomotionTask, HighSpeedLocomotionTask, LateralLocomotionTask, HeightLocomotionTask, WidthLocomotionTask

from .mc_locomotion_task import MCForwardLocomotionTask, MCHighSpeedLocomotionTask, MCRecoveryTask, MCRollingTask, \
    MCLateralLocomotionTask, MCHeightLocomotionTask, MCWidthLocomotionTask, MCMixTask


class MCMultiTask:
    task_cls = [RollingTask, RecoveryTask, ForwardLocomotionTask, HighSpeedLocomotionTask, LateralLocomotionTask,
                HeightLocomotionTask, WidthLocomotionTask]
    task_adv_coef = [1.5, 1.5, 1., 1., 1., 1., 1.]  # mc


@register
class MCMultiRollingTask(MCMultiTask, MCRollingTask):
    net_id = 0


@register
class MCMultiRecoveryTask(MCMultiTask, MCRecoveryTask):
    net_id = 1


@register
class MCMultiForwardLocomotionTask(MCMultiTask, MCForwardLocomotionTask):
    net_id = 2


@register
class MCMultiHighSpeedLocomotionTask(MCMultiTask, MCHighSpeedLocomotionTask):
    net_id = 3


@register
class MCMultiLateralLocomotionTask(MCMultiTask, MCLateralLocomotionTask):
    net_id = 4


@register
class MCMultiHeightLocomotionTask(MCMultiTask, MCHeightLocomotionTask):
    net_id = 5


@register
class MCMultiWidthLocomotionTask(MCMultiTask, MCWidthLocomotionTask):
    net_id = 6


@register
class MCMultiMixTask(MCMultiTask, MCMixTask):
    net_id = 7
    p = np.random.uniform()
    env_type = len(MCMultiTask.task_adv_coef) - 1
    if p < 1 / env_type:
        net_id = 0
    elif p < 2 / env_type:
        net_id = 1
    elif p < 3 / env_type:
        net_id = 2
    elif p < 4 / env_type:
        net_id = 3
    elif p < 5 / env_type:
        net_id = 4
    elif p < 6 / env_type:
        net_id = 5
    else:
        net_id = 6


TaskGroupColor = {
    'RecoveryTask': (1., 0., 0., 1.),
    'StandingTask': (0., 1., 0., 1.),
    'LocomotionTask': (0., 0., 1., 1.),
    'JumpTask': (1., 1., 0., 1.),
}
