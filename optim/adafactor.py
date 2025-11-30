import torch
import torch.optim
from transformers import Adafactor
from .optimizer import OptimizerWithWeightStashing

# ---------------------------------------------------------------------------
# 💡 [핵심 해결책] SafeAdafactor (Whitelist 방식)
# Adafactor가 확실히 아는 인자만 골라서 전달하고, 나머지는 과감히 버립니다.
# ---------------------------------------------------------------------------
class SafeAdafactor(Adafactor):
    def __init__(self, params, **kwargs):
        # 1. Adafactor가 허용하는 인자 목록 (Whitelist)
        allowed_keys = {
            'lr', 'weight_decay', 'eps', 'relative_step', 
            'warmup_init', 'scale_parameter', 'clip_threshold', 'beta1'
        }
        
        # 2. 들어온 인자 중에서 Whitelist에 있는 것만 골라냅니다.
        clean_kwargs = {k: v for k, v in kwargs.items() if k in allowed_keys}

        # 3. 필수 설정 강제 (LR 스케줄러 충돌 방지)
        # (AsyncPP가 LR을 제어하므로 자동 기능 끄기)
        clean_kwargs['relative_step'] = False
        clean_kwargs['warmup_init'] = False
        clean_kwargs['scale_parameter'] = False

        # 4. 깨끗해진 인자로 진짜 Adafactor 초기화
        super().__init__(params, **clean_kwargs)

# ---------------------------------------------------------------------------
# 💡 Monkey Patching: torch.optim 안에 SafeAdafactor를 주입
# ---------------------------------------------------------------------------
# 항상 덮어쓰도록 강제하여 확실하게 적용합니다.
torch.optim.Adafactor = SafeAdafactor


# ---------------------------------------------------------------------------
# AsyncPP용 래퍼 클래스
# ---------------------------------------------------------------------------
class AdafactorWithWeightStashing(OptimizerWithWeightStashing):
    def __init__(self, modules, master_parameters, model_parameters,
                 loss_scale, num_versions, lr=1e-3, 
                 weight_decay=0, verbose_freq=0, macrobatch=False,
                 use_use_weight=False, save_dir=None, stash_to_cpu=False,
                 clip_grad=None):
        
        # 부모 클래스 초기화
        # 여기서 내부적으로 torch.optim.Adafactor(즉, SafeAdafactor)가 호출됩니다.
        # 부모 클래스가 어떤 잡동사니 인자를 넘기든 SafeAdafactor가 다 걸러낼 것입니다.
        super(AdafactorWithWeightStashing, self).__init__(
            optim_name='Adafactor',
            modules=modules, master_parameters=master_parameters,
            model_parameters=model_parameters, loss_scale=loss_scale,
            num_versions=num_versions, lr=lr, weight_decay=weight_decay,
            verbose_freq=verbose_freq, macrobatch=macrobatch,
            use_use_weight=use_use_weight, save_dir=save_dir,
            stash_to_cpu=stash_to_cpu
        )