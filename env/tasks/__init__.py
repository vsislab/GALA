# -*- coding: utf-8 -*-
from .common import load_task_cls, is_valid_task
from .base_task import BaseTask, BaseMCTask
from .null_task import NullTask
from .locomotion_task import ForwardLocomotionTask, HighSpeedLocomotionTask, LateralLocomotionTask, LocomotionTask, HeightLocomotionTask, WidthLocomotionTask
from .recovery_task import RecoveryTask, RollingTask

from .mc import *
