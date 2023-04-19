import os

import typing

import yaml  # type: ignore
from .utils import open

codedir = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + "/data"

xemc3dir = "~/.local/xemc3/"

configfile = xemc3dir + "config.yaml"

defaults = dict(
    filenames="default",
)

config = defaults.copy()

try:
    with open(configfile) as f:
        read = yaml.safe_load(f)
    config.update(read)
except FileNotFoundError:
    pass


def get(key):
    return config[key]


class context:
    def __init__(self, config):
        self.old = config.copy()
        self.state = config.copy()

    def __enter__(self):
        global config
        config = self.state

    def __exit__(self, *args):
        global config
        config = self.old

    def set(self, *args, **kwargs):
        if len(args) % 2 == 0:
            for key, value in zip(args[::2], args[1::2]):
                self.state[key] = value
        self.state.update(kwargs)
        return self


def set(*args, **kwargs):
    return context(config).set(*args, **kwargs)


class Files:
    def _update(self):
        if self.source != config["filenames"]:
            print("loading")
            self._load(config["filenames"])
        assert self.cache

    def __init__(self):
        self.caches = dict()
        self.source = None

    def __getitem__(self, key: str) -> typing.Dict[str, typing.Any]:
        self._update()
        return self.cache[key]

    def __setitem__(self, key: str, value: typing.Dict[str, typing.Any]):
        self._update()
        self.cache[key] = value
        return self

    def __iter__(self):
        self._update()
        return self.cache.__iter__()

    def _load(self, filename):
        if self.source:
            self.caches[self.source] = self.cache
        try:
            self.cache: typing.Dict[str, typing.Dict[str, typing.Any]] = self.caches[
                filename
            ]
            self.source = filename
            return
        except KeyError:
            pass
        filenames = [
            (f"{codedir}/{filename}.yaml", True),
            (f"{xemc3dir}/files.yaml", False),
            (f"{xemc3dir}/{filename}.yaml", True),
        ]
        found = False
        self.cache = dict()
        for fn, specific in filenames:
            try:
                with open(fn) as f:
                    self.cache.update(yaml.safe_load(f))
                if specific:
                    found = True
            except FileNotFoundError as e:
                print(e)
                pass
        if not found:
            raise FileNotFoundError(
                f"Failed to find a config file for {filename} - the following locations have been tried:"
                + ("".join([f"\n * '{fn}'" for fn, s in filenames if s]))
            )
        for fn, data in self.cache.items():
            if data.get("dtype", None) == "int":
                self.cache[fn]["dtype"] = int

        self.source = filename

    def items(self):
        self._update()
        return self.cache.items()


files = Files()
