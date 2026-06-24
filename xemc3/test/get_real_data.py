import os


def get_data(force=False):
    basedir = "./example-data/"
    if not os.path.isdir(basedir) and not force:
        import pytest

        pytest.skip("create {basedir} to enable testing on real data")
        return
    if not os.path.isdir(basedir + ".git"):
        os.system(
            f"git clone https://oauth2:glpat-8Xqp1-UMwqV4rVuoSb-QfG86MQp1OjZrYgk.01.0z0bc32ro@gitlab.mpcdf.mpg.de/dave/xemc3-data/ {basedir} --depth 1"
        )
    else:
        os.system(f"cd {basedir}; git fetch origin main --depth 1")
        os.system(f"cd {basedir}; git checkout origin main")
    return basedir + "emc3_example"


if __name__ == "__main__":
    import sys

    get_data(force=True)
    sys.exit(0)
