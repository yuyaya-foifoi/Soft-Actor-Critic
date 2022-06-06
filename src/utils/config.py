import yaml

with open("./config/config.yml") as file:
    cfg = yaml.safe_load(file)


def get_config():
    return cfg
