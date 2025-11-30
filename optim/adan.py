import torch
import torch.optim
from pytorch_optimizer import Adan
from .optimizer import OptimizerWithWeightStashing

# 1. SafeAdan: 불필요한 인자 필터링 (필수!)
class SafeAdan(Adan):
    def __init__(self, params, **kwargs):
        # Adan이 허용하는 인자만 남깁니다 (Whitelist 방식)
        allowed_keys = {
            'lr', 'betas', 'weight_decay', 'eps', 
            'weight_decay', 'max_grad_norm', 'no_prox'
        }
        clean_kwargs = {k: v for k, v in kwargs.items() if k in allowed_keys}
        super().__init__(params, **clean_kwargs)

# 2. Monkey Patch (필수!)
if not hasattr(torch.optim, 'Adan'):
    torch.optim.Adan = SafeAdan

# 3. AsyncPP Wrapper
class AdanWithWeightStashing(OptimizerWithWeightStashing):
    def __init__(self, modules, master_parameters, model_parameters,
                 loss_scale, num_versions, lr=1e-3, 
                 weight_decay=0.02, verbose_freq=0, macrobatch=False,
                 use_use_weight=False, save_dir=None, stash_to_cpu=False,
                 clip_grad=None):
        
        super(AdanWithWeightStashing, self).__init__(
            optim_name='Adan',
            modules=modules, master_parameters=master_parameters,
            model_parameters=model_parameters, loss_scale=loss_scale,
            num_versions=num_versions, lr=lr, weight_decay=weight_decay,
            verbose_freq=verbose_freq, macrobatch=macrobatch,
            use_use_weight=use_use_weight, save_dir=save_dir,
            stash_to_cpu=stash_to_cpu
        )
        
        # Adan 권장 설정 (Betas: 0.98, 0.92, 0.99)
        self.optim = SafeAdan(
            self.master_parameters,
            lr=lr,
            betas=(0.98, 0.92, 0.99),
            weight_decay=weight_decay
        )