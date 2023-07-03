from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .ctdet import CtdetTrainer
from .ctdet_gaze import Ctdet_GazeTrainer
from .ddd import DddTrainer
from .exdet import ExdetTrainer
from .multi_pose import MultiPoseTrainer

train_factory = {
  'exdet': ExdetTrainer, 
  'ddd': DddTrainer,
  'ctdet': CtdetTrainer,
  'ctdet_gaze': Ctdet_GazeTrainer,
  'multi_pose': MultiPoseTrainer, 
}
