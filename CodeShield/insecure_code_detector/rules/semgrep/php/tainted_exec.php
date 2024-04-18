/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

<?php

// ok: unsafe-exec
exec("ls");

exec($asdadad)
$var = $_POST['key'];
$var2 = $var['key'];
// ruleid: unsafe-exec
shell_exec($var2);


function foo($v)
{
    // ok: unsafe-exec
    system($v);

    $par = $_GET['i'];
    // ruleid: unsafe-exec
    system($par);

}

function foo2($par)
{
    $v = "a var";
    // ok: unsafe-exec
    system($v);
    // ok: unsafe-exec
    popen($v, 'r');
}

$user_controlled = $_POST['key'];
$sanitized = escapeshellarg($user_controlled);
// ok: unsafe-exec
exec($sanitized);

$sanitized = escapeshellcmd($user_controlled);
// ok: unsafe-exec
exec($sanitized);

function foo3(){
    $user_controlled = $_POST['key'];
    // ruleid: unsafe-exec
    exec($user_controlled);
    $sanitized = escapeshellarg($user_controlled);
    // ok: unsafe-exec
    exec($sanitized);
}
