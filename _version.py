def __load_version():
    try:
        import tomllib
    except ImportError:
        import toml as tomllib
    from os.path import dirname, join

    with open(join(dirname(__file__), "pyproject.toml"), "r") as f:
        return tomllib.loads(f.read())["tool"]["poetry"]["version"]


__version__ = __load_version()
