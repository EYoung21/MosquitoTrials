import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf

OmegaConf.register_new_resolver("len", lambda x: len(x), replace=True)

def build_model(cfg):
    model = instantiate(cfg)
    return model

if __name__ == "__main__":
    build_model()