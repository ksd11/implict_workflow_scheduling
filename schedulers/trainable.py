from trainers import make_trainer,model_path
from trainers.trainer import Trainer,CfgType


# 只调度到edge，并且选择下载时间最短的那台机器
class TrainableScheduler:
    def __init__(self, cfg):
        path = model_path(cfg['trainer']['trainer_cls'], cfg['env']['id'])
        trainer: Trainer = make_trainer(cfg)
        self.model = trainer.load(path).get_model()

    def schedule(self, obs: list)  -> tuple[int, dict]:
        return self.model.predict(obs, deterministic=True)
    