/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

<?php
// ruleid: mcrypt-use
$modes = mcrypt_list_modes();

// ruleid: mcrypt-use
$algorithms = mcrypt_list_algorithms();

// ruleid: mcrypt-use
echo mdecrypt_generic($encrypted);

function encrypting__stuff($text)
{
    // ruleid: mcrypt-use
    $encrypted = mcrypt_encrypt($text);
}

function decrypting__stuff($text)
{
    // ruleid: mcrypt-use
    $dencrypted = mcrypt_dencrypt($text);
}
// ruleid: mcrypt-use
mdecrypt_generic();

// ruleid: mcrypt-use
$mo = mcrypt_module_open(
    $cipher,
    '/usr/local/libmcrypt/modules/algorithms/',
    $mode,
    '/usr/local/libmcrypt/modules/modes/');

// ok: mcrypt-use
$ciphertext = openssl_encrypt($plaintext, $cipher, $key, $options=0, $iv, $tag);
