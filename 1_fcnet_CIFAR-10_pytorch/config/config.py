"""
    原先的方式：文件A.py和文件B.py都各自加载了该配置文件，且文件B.py导入了文件A.py这个包。
    那么这种方式存在一定的隐患。
    因此建议将配置加载逻辑放在一个单独的模块中。
"""

from omegaconf import OmegaConf

# load config.yaml
conf = OmegaConf.load('config.yaml')