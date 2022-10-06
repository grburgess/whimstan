from dataclasses import dataclass
from pathlib import Path

from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from rich.tree import Tree

# Path to configuration

_config_path = Path("~/.config/whim_stan/").expanduser()

_config_name = Path("whim_stan_config.yml")

_config_file = _config_path / _config_name

# Define structure of configuration with dataclasses


@dataclass
class Logging:

    on: bool = True
    level: str = "WARNING"


@dataclass
class Colors:
    green: str = "#00D584"
    purple: str = "#985CFC"
    dark_purple: str = "#381C66"
    yellow: str = "#EDE966"
    grey: str = "#385656"
    lightgrey: str = "#839393"
    black: str = "#1F2222"

@dataclass
class Plotting:
    colors: Colors = Colors()


@dataclass
class WHIMStanConfig:

    logging: Logging = Logging()
    plotting: Plotting = Plotting()


# Read the default config
whim_stan_config: WHIMStanConfig = OmegaConf.structured(WHIMStanConfig)

# Merge with local config if it exists
if _config_file.is_file():

    _local_config = OmegaConf.load(_config_file)

    whim_stan_config: WHIMStanConfig = OmegaConf.merge(
        whim_stan_config, _local_config
    )

# Write defaults if not
else:

    # Make directory if needed
    _config_path.mkdir(parents=True, exist_ok=True)

    with _config_file.open("w") as f:

        OmegaConf.save(config=whim_stan_config, f=f.name)


def recurse_dict(d, tree) -> None:

    for k, v in d.items():

        if (type(v) == dict) or isinstance(v, DictConfig):

            branch = tree.add(
                k, guide_style="bold medium_orchid", style="bold medium_orchid"
            )

            recurse_dict(v, branch)

        else:

            tree.add(
                f"{k}: [blink cornflower_blue]{v}",
                guide_style="medium_spring_green",
                style="medium_spring_green",
            )

    return


def show_configuration() -> Tree:

    tree = Tree(
        "config", guide_style="bold medium_orchid", style="bold medium_orchid"
    )

    recurse_dict(whim_stan_config, tree)

    return tree
