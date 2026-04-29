#!/usr/bin/env python3


print("""CLI reference
=============

""")

path = __file__.split("/")
if len(path) < 2:
    path = ["."]
else:
    path = path[:-1]
path += ["..", "setup.cfg"]
rec = False
found = []
with open("/".join(path)) as f:
    for line in f:
        if rec:
            if line:
                found.append(line)
            else:
                rec = False
        if line.startswith("console_scripts"):
            rec = True

for line in found:
    name, ref = [x.strip() for x in line.split("=")]
    ref = ref.split(":")[0]
    print(f"""

.. sphinx_argparse_cli::
   :module: {ref}
   :func: parser
   :prog: {name}
""")
