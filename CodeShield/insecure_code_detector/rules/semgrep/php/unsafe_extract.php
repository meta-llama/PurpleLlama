/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

<?php

$test = $_POST;
$test2 = $_GET["p"];

// ruleid: unsafe-extract
extract($test);

// ruleid: unsafe-extract
extract($test2);

// ruleid: unsafe-extract
extract($_POST["test"]);

// ruleid: unsafe-extract
extract($_GET);

// ruleid: unsafe-extract
extract($_POST);

// ruleid: unsafe-extract
extract($_FILES);


// ok: unsafe-extract
extract($_POST, EXTR_SKIP);

// ok: unsafe-extract
extract($_GET, EXTR_SKIP, "test");

function extracting($index){
    $ex = $_GET[$index];
// ruleid: unsafe-extract
    extract($ex);
}

class TestClass
{
    public function assign($tmp)
    {
        // ok: unsafe-extract
        extract($tmp);
    }
}
function assign($tmp)
{
    // ok: unsafe-extract
    extract($tmp);

}
