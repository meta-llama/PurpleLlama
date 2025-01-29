# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


from CodeShield.insecure_code_detector.languages import Language

from .insecure_code_detector_test import InsecureCodeDetectorTest

# the following test cases contain an input string, and the corresponding number of expected insecure pattern matches
JAVA_TEST_CASES = [
    ('final Cipher cipher = Cipher.getInstance("AES/ECB/PKCS5Padding");', 1),
    ('final Cipher cipher = Cipher.getInstance("DES/ECB/PKCS5Padding");', 2),
    ('final Cipher cipher = Cipher.getInstance("DESede/ECB/PKCS5Padding");', 2),
    ('final Cipher cipher = Cipher.getInstance("AES");', 1),
    (".setBlockModes(KeyProperties.BLOCK_MODE_ECB)", 1),
    ('final Cipher cipher = Cipher.getInstance("AES/CBC/PKCS5Padding")', 1),
    ('final Cipher cipher = Cipher.getInstance("AES/CBC/PKCS7Padding")', 1),
    ('SecretKeyFactory.getInstance("3DES");', 1),
    ('SecretKeyFactory.getInstance("DESede");', 1),
    ('SecretKeyFactory.getInstance("DES");', 1),
    ('SecretKeyFactory.getInstance("Blowfish");', 1),
    ('SecretKeyFactory.getInstance("DESedeWrap");', 1),
    ('SecretKeyFactory.getInstance("RC4");', 1),
    ('SecretKeyFactory.getInstance("RC2");', 1),
    ("Test nullCipher = new javax.crypto.NullCipher();", 1),
    ("Test customCipher = new NullCipher();", 1),
    ('Test test = MessageDigest.getInstance("SHA-1");', 1),
    ('Test test = MessageDigest.getInstance("SHA1", "SUN");', 1),
    ("byte[] test = DigestUtils.getSha1Digest().digest(foo.getBytes());", 1),
    ("byte[] test = DigestUtils.sha1(foo.getBytes());", 1),
    ("byte[] test = DigestUtils.sha1Hex(foo.getBytes());", 1),
    (
        'customXmlFactory.setProperty("javax.xml.stream.isSupportingExternalEntities", true);',
        1,
    ),
    (
        'DocumentBuilderFactory.setFeature("http://apache.org/xml/features/disallow-doctype-decl", false);',
        1,
    ),
    (
        'DocumentBuilderFactory.setFeature("http://xml.org/sax/features/external-parameter-entities", true);',
        1,
    ),
    (
        'DocumentBuilderFactory.setFeature("http://xml.org/sax/features/external-general-entities", true);',
        1,
    ),
    ('Cipher.getInstance("RSA/NONE/NoPadding", "BC");', 1),
    ('Cipher.getInstance("RSA/NONE/NoPadding");', 1),
    ('Test test = MessageDigest.getInstance("MD2");', 1),
    ('Test test = MessageDigest.getInstance("MD2", "SUN");', 1),
    ("byte[] test = DigestUtils.getMd2Digest().digest(foo.getBytes());", 1),
    ("byte[] test = DigestUtils.md2(foo.getBytes());", 1),
    ("byte[] test = DigestUtils.md2Hex(foo.getBytes());", 1),
    ('Test test = MessageDigest.getInstance("MD4");', 1),
    ('Test test = MessageDigest.getInstance("MD4", "SUN");', 1),
    ("byte[] test = DigestUtils.getMd4Digest().digest(foo.getBytes());", 1),
    ("byte[] test = DigestUtils.md4(foo.getBytes());", 1),
    ("byte[] test = DigestUtils.md4Hex(foo.getBytes());", 1),
    ('Test test = MessageDigest.getInstance("MD5");', 1),
    ('Test test = MessageDigest.getInstance("md5");', 1),
    ('Test test = MessageDigest.getInstance("MD5", "SUN");', 1),
    ("byte[] test = DigestUtils.getMd5Digest().digest(foo.getBytes());", 1),
    ("byte[] test = DigestUtils.md5(foo.getBytes());", 1),
    ("byte[] test = DigestUtils.md5Hex(foo.getBytes());", 1),
    ('final Cipher cipher = Cipher.getInstance("RC2");', 1),
    ('final Cipher cipher = Cipher.getInstance("RC4");', 1),
    ('final Cipher cipher = Cipher.getInstance("ARCFOUR");', 1),
    ("WebView.setWebContentsDebuggingEnabled(true);", 1),
    ("mContext.openFileOutput( foo, Context.MODE_WORLD_READABLE );", 1),
    ("mContext.openFileOutput( foo, Context.MODE_WORLD_WRITABLE );", 1),
    ("settings.setAllowFileAccess(true);", 1),
    ("settings.setAllowFileAccessFromURLs(true);", 1),
    ("settings.setAllowUniversalAccessFromURLs(true);", 1),
    ("import org.apache.http.conn.ssl.NoopHostnameVerifier;", 1),
    ("ALLOW_ALL_HOSTNAME_VERIFIER", 1),
    ("HostnameVerifier allowAll = new AllowAllHostnameVerifier();", 1),
    ("HostnameVerifier allowAll = new NullHostnameVerifier();", 1),
    ("HostnameVerifier allowAll = new NoopHostnameVerifier();", 1),
    ("ssf.setHostnameVerifier(SSLSocketFactory.ALLOW_ALL_HOSTNAME_VERIFIER);", 1),
    (
        'String dir = System.getProperty("dir");Runtime rt = Runtime.getRuntime(); Process proc = rt.exec(new String[] {"sh", "-c", "ls " + dir});',
        1,
    ),
    ("Runtime.getRuntime().exec(userInput)", 1),
]


class TestJavaInsecureCodeDetector(InsecureCodeDetectorTest):
    async def test_java_regex_patterns(self) -> None:
        await self.run_regex_tests(Language.JAVA, JAVA_TEST_CASES)
