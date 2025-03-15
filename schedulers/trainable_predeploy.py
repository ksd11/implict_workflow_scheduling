from trainers import make_trainer,model_path
from trainers.trainer import Trainer,CfgType
from cfg_loader import load
from .xanadu import XanaduScheduler

# 只调度到edge，并且选择下载时间最短的那台机器
class TrainablePredeployScheduler(XanaduScheduler):
    def __init__(self, config_path: str, env, predeploy_degree = 1):
        super(TrainablePredeployScheduler, self).__init__(env, predeploy_degree)
        
        cfg = load(config_path)
        trainer: Trainer = make_trainer(cfg)
        path = model_path(cfg['trainer']['trainer_cls'], cfg['env']['id'])
        # path = trainer.get_best_model_path() + "/best_model.zip"
        self.model = trainer.load(path, device=cfg['trainer']['device']).get_model()

    def schedule(self, obs: list)  -> tuple[int, dict]:
        return self.model.predict(obs, deterministic=True)[0], self.after_deploy_hook_func
    