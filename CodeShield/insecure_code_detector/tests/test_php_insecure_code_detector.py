# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


from CodeShield.insecure_code_detector.languages import Language

from .insecure_code_detector_test import InsecureCodeDetectorTest

PHP_TEST_CASES = [
    ("hello world", 0),
    ("eval($_GET['key'])", 1),
    ("ghp_123456789012345678901234567890123456", 1),
    ("AIza1234567890abcdefghijklmnopqrstuvwxyz-_", 1),
    (
        # @lint-ignore PRIVATEKEY for tests only
        """-----BEGIN PRIVATE KEY-----a-----END PRIVATE KEY-----""",
        1,
    ),
    (
        "AR12345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890",
        1,
    ),
    ("ya29.ABCDEFGHIJKLMNOPQRSTUVWXYZ", 1),
    ("sk_live_123456789012345678901234", 1),
    ("SK12345678901234567890123456789012", 1),
    ("HEROKU_API_KEY=1234567890123456789012345678901234567890", 1),
    ("shell_exec($_GET['command'])", 2),
    ("mysql_query($_POST['query'])", 1),
    ("echo($_GET['message'])", 1),
    ("ini_set('display_errors', 'on')", 1),
    ("md5($input)", 1),
    ("exec($command)", 1),
    ("set_cookie($_POST['query'])", 1),
    ("require($_SESSION['file'])", 1),
    ("file_get_contents($_GET['url'])", 2),
    ("password_hash($password, PASSWORD_DEFAULT)", 1),
    ("die($_GET['message'])", 1),
    ("include($_SESSION['file'])", 1),
    ("chmod($_GET['var'])", 1),
    ("curl_exec($_GET['command'])", 2),
    ("header('Location: '$_GET['url'])", 1),
    ("move_uploaded_file($tmp_name, 'uploads/file.exe')", 1),
    ("update($_SESSION['data'])", 1),
    ("mt_rand($a)", 1),
    ("simplexml_load_string($xml, 'SimpleXMLElement', LIBXML_NOENT)", 1),
]


class TestPHPInsecureCodeDetector(InsecureCodeDetectorTest):
    async def test_php_regex_patterns(self) -> None:
        await self.run_regex_tests(Language.PHP, PHP_TEST_CASES)
