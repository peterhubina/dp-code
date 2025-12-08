import os
import hydra

from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="config", config_name="default")
def do_main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))



if __name__ == "__main__":
    do_main()
