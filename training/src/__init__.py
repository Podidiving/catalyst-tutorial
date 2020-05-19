# flake8: noqa
# isort:skip_file

from catalyst.dl import registry, SupervisedRunner as Runner
from .experiment import Experiment
from .model import get_model_classification

registry.Model(get_model_classification)
