/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

<?php
// ruleid: unsafe-hash-use
$hash = md5($password);

// ruleid: unsafe-hash-use
$hash = md5_file($file);

// ruleid: unsafe-hash-use
$hash = sha1($password);

// ruleid: unsafe-hash-use
$hash = sha1_file($file);

// ok: unsafe-hash-use
$hash = password_hash("very_secret_password", PASSWORD_BCRYPT);

// ok: unsafe-hash-use
echo hash('sha256', 'The quick brown fox jumped over the lazy dog.');
