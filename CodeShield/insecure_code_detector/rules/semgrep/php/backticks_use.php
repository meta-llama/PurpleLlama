/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

<?php

function dangerousBackticks(){
    $user_input = $_GET['command'];
    // ruleid: backticks-use
    output = `adasd {$user_input}`;
}

function notDangerousUsage(){
    // ok: backticks-use
    output = `ls -l`;
}
