import torch
import torch.optim  # torch.optim을 임포트해야 주입할 수 있습니다.
from pytorch_optimizer import Lion
from .optimizer import OptimizerWithWeightStashing

# 💡 [핵심 해결책]: torch.optim 안에 Lion을 강제로 집어넣습니다.
# 이렇게 하면 부모 클래스가 getattr(torch.optim, 'Lion')을 호출할 때 성공합니다.
if not hasattr(torch.optim, 'Lion'):
    torch.optim.Lion = Lion

class LionWithWeightStashing(OptimizerWithWeightStashing):
    def __init__(self, modules, master_parameters, model_parameters,
                 loss_scale, num_versions, lr=1e-4, betas=(0.9, 0.99),
                 weight_decay=0, verbose_freq=0, macrobatch=False,
                 use_use_weight=False, save_dir=None, stash_to_cpu=False,
                 clip_grad=None):
        
        # 부모 클래스 초기화
        # 이제 torch.optim.Lion이 존재하므로 에러가 나지 않습니다.
        super(LionWithWeightStashing, self).__init__(
            optim_name='Lion',
            modules=modules, master_parameters=master_parameters,
            model_parameters=model_parameters, loss_scale=loss_scale,
            num_versions=num_versions, lr=lr, weight_decay=weight_decay,
            verbose_freq=verbose_freq, macrobatch=macrobatch,
            use_use_weight=use_use_weight, save_dir=save_dir,
            stash_to_cpu=stash_to_cpu
        )
        
        # (선택 사항) 만약 부모 클래스에서 생성된 옵티마이저 설정을 덮어쓰고 싶다면
        # 아래처럼 다시 할당할 수 있습니다. (보통은 위의 super() 호출로 충분합니다)
        # self.optim = torch.optim.Lion(...)    