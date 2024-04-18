# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


def test(var):
    # ok: insecure-eval-use
    eval()
    # ok: insecure-eval-use
    eval("hardcoded_string")

    # ruleid: insecure-eval-use
    eval(var)

    def eval(var):
        return var

    # ok: insecure-eval-use
    eval(var)
