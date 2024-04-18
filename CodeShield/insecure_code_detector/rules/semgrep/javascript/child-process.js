/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

const cp = require('child_process');

function testUserInput(input) {
    // ruleid: child-process
    cp.exec(input, (error, stdout, stderr) => {
        console.log(error, stdout, stderr);
    });

    // ruleid: child-process
    cp.execSync(input).toString();

    // ruleid: child-process
    cp.spawn(input, ['/etc'], (error, stdout, stderr) => {
        console.log(error, stdout, stderr);
    });

    // ruleid: child-process
    cp.spawnSync(input).toString();
    // ruleid: child-process
    cp.spawnSync(input, ['/etc']).toString();
}

function testConstantString(input) {
    // ok: child-process
    cp.exec('ls', (error, stdout, stderr) => {
        console.log(error, stdout, stderr);
    });

    // ok: child-process
    cp.execSync('ls').toString();

    // ok: child-process
    cp.spawn('ls', ['/etc'], (error, stdout, stderr) => {
        console.log(error, stdout, stderr);
    });

    // ok: child-process
    cp.spawnSync('ls').toString();
    // ok: child-process
    cp.spawnSync('ls', ['/etc']).toString();
}
