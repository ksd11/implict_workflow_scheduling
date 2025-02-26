import yaml
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import re

def convert_scientific_notation(cfg):
    """递归转换字典中的科学计数法字符串为float"""
    if isinstance(cfg, dict):
        return {k: convert_scientific_notation(v) for k, v in cfg.items()}
    elif isinstance(cfg, list):
        return [convert_scientific_notation(v) for v in cfg]
    elif isinstance(cfg, str):
        # 匹配科学计数法格式
        if re.match(r'^[-+]?[\d.]+[eE][-+]?\d+$', cfg):
            return float(cfg)
    return cfg

def load(filename=None):
    if not filename:
        args = make_parser().parse_args()
        filename = args.filename

    with open(filename, "r") as stream:
        cfg = yaml.safe_load(stream)
        cfg = convert_scientific_notation(cfg)

    return cfg

def make_parser():
    parser = ArgumentParser(
        description=__doc__, formatter_class=ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "-f",
        "--file",
        dest="filename",
        help="experiment definition file",
        metavar="FILE",
        required=True,
    )

    # 添加测试运行参数
    parser.add_argument(
        "-t",
        "--test",
        dest="test_mode",
        help="run in test mode",
        action="store_true",  # 布尔标志，不需要额外参数
        default=False
    )

    return parser

def parse():
    args = make_parser().parse_args()
    return {
        "cfg": load(args.filename),
        "test_mode": args.test_mode
    }