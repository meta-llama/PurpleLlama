# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import subprocess


def test(param):
    # ruleid: insecure-subprocess-using-shell
    subprocess.run("echo %s" % param, shell=True)
    # ruleid: insecure-subprocess-using-shell
    subprocess.run(f"echo {param} ", shell=True)
    # ruleid: insecure-subprocess-using-shell
    subprocess.run(f"echo {param} ", other=False, shell=True)
    # ruleid: insecure-subprocess-using-shell
    subprocess.run(f"echo {param} ", shell=True, other=False)
    # ruleid: insecure-subprocess-using-shell
    subprocess.run("echo " + param, shell=True)

    # ok: insecure-subprocess-using-shell
    subprocess.run("echo ", shell=True)
    # ok: insecure-subprocess-using-shell
    subprocess.run(f"echo {param} ")
    # ok: insecure-subprocess-using-shell
    subprocess.run(f"echo {param} ", shell=False)
