# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.parallel import is_module_wrapper
from mmcv.runner import HOOKS, Hook


@HOOKS.register_module()
class SetIterInfoHook(Hook):
    """Set runner's epoch information to the model."""

    def before_train_iter(self, runner):
        iter = runner.iter
        model = runner.model
        if is_module_wrapper(model):
            model = model.module
        model.set_iter(iter)