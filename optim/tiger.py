import torch
import torch.optim
from pytorch_optimizer import Tiger
from .optimizer import OptimizerWithWeightStashing

# 1. SafeTiger
class SafeTiger(Tiger):
    def __init__(self, params, **kwargs):
        allowed_keys = {'lr', 'betas', 'weight_decay'}
        clean_kwargs = {k: v for k, v in kwargs.items() if k in allowed_keys}
        super().__init__(params, **clean_kwargs)

# 2. Monkey Patch
if not hasattr(torch.optim, 'Tiger'):
    torch.optim.Tiger = SafeTiger

# 3. AsyncPP Wrapper
class TigerWithWeightStashing(OptimizerWithWeightStashing):
    def __init__(self, modules, master_parameters, model_parameters,
                 loss_scale, num_versions, lr=1e-3, 
                 weight_decay=0.01, verbose_freq=0, macrobatch=False,
                 use_use_weight=False, save_dir=None, stash_to_cpu=False,
                 clip_grad=None):
        
        super(TigerWithWeightStashing, self).__init__(
            optim_name='Tiger',
            modules=modules, master_parameters=master_parameters,
            model_parameters=model_parameters, loss_scale=loss_scale,
            num_versions=num_versions, lr=lr, weight_decay=weight_decay,
            verbose_freq=verbose_freq, macrobatch=macrobatch,
            use_use_weight=use_use_weight, save_dir=save_dir,
            stash_to_cpu=stash_to_cpu
        )
        
        # Tiger 설정 (Lion과 비슷하게 LR을 낮게 쓰는 경향이 있음)
        self.optim = SafeTiger(
            self.master_parameters,
            lr=lr,
            betas=(0.965, 0.99),
            weight_decay=weight_decay
        )