# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


def test(var):
    # ok: insecure-exec-use
    exec()
    # ok: insecure-exec-use
    exec("hardcoded_string")

    # ruleid: insecure-exec-use
    exec(var)

    def exec(var):
        return var

    # ok: insecure-exec-use
    exec(var)
