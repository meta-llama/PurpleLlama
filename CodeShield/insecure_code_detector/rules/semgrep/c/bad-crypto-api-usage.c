/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

void warnOnMd5Usage() {
  // ruleid: md5-usage
  MD5_Init(1);
  EVP_md5(1);
  EVP_md5();
  EVP_md5(1, 2, 3);
}

void warnOnSha1Usage() {
  // ruleid: sha1-usage
  SHA1_Init(1);
  EVP_sha1(1);
  EVP_md5_sha1(1);
}

void warnOnMd2Md4Hash() {
  MD2_Init(1);
  EVP_md2(1);
  MD4_Init(1);
  EVP_md4(1);
}

void warnOnBlowfishEncryption() {
  EVP_bf_cbc(1);
  BF_decrypt(1);
  BF_ecb_encrypt(1);
}

void warnOnDesAlgorithm() {
  EVP_des_function(1);
  EVP_des_ofb(1);
  DES_ecb_encrypt(1);
}

void warnOnEcbModeEncryption() {
  EVP_aes_256_ecb(1);
}

void warnOnUnauthenticatedEncryptionUsage() {
  EVP_aes_256_ecb();
  EVP_aes_128_cbc();
  EVP_aes_192_ctr();
  EVP_aes_256_xts();
  EVP_aes_128_cfb();
  EVP_aes_128_ofb();
}

void warnOnHardCodedSecret() {
  char scrt[] = "secret";
  char password[10];
  char secret[10];
  char privateKey[10];
  char private_key[10];
  char secret_key[10];
  char secretKey[10];
  password = scrt;
  secret = scrt;
  privateKey = srct;
  private_key = srct;
  secret_key = srct;
  secretKey = srct;
  secret_key_abc = srct;
  abc_secret_key = srct;
  abc_secret_key_abc = srct;
  char secretKey2[10] = "secret";
}

void warnOnFixedNonceUsage() {
  char nonce[];
  char iv[];
  char initializationVector[];
  char initialization_vector[];

  nonce = "nonce";
  iv = "nonce";
  initializationVector = "nonce";
  initialization_vector = "nonce";
  initialization_vector_abc = "nonce";
  abc_initialization_vector = "nonce";
  // should not warn
  defiv = "nonce";
}

void warnOnInsufficientKeySize() {
  EVP_RSA_gen(512);
  EVP_RSA_gen(768);
  EVP_RSA_gen(1024);
  EVP_RSA_gen(1536);
  // should not warn
  EVP_RSA_gen(2048);
  RSA_generate_key(0, 512, 0, 0);
  RSA_generate_key(0, 768, 0, 0);
  RSA_generate_key_ex(0, 1024, 0, 0);
  RSA_generate_key_ex(0, 1536, 0, 0);
  // should not warn
  RSA_generate_key_ex(0, 2048, 0, 0);
}

void warnOnInsufficientKeySizeTwo() {
  int keySize1 = 512;
  int keySize2 = 768;
  int keySize3 = 1024;
  int keySize4 = 1536;
  int keySize5 = 2048;

  EVP_RSA_gen(keySize1);
  EVP_RSA_gen(keySize2);
  EVP_RSA_gen(keySize3);
  EVP_RSA_gen(keySize4);

  RSA_generate_key(0, keySize1, 0, 0);
  RSA_generate_key(0, keySize2, 0, 0);
  RSA_generate_key_ex(0, keySize3, 0, 0);
  RSA_generate_key_ex(0, keySize4, 0, 0);
  // should not warn on keySize5

  EVP_RSA_gen(keySize5);
  RSA_generate_key(0, keySize5, 0, 0);
}
