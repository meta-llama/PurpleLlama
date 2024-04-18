/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

var a = new XMLHttpRequest();
// ruleid: unencrypted-fetch
a.open('GET', "http://example.com");
// ruleid: unencrypted-fetch
a.open('POST', 'http://example.com');
// ruleid: unencrypted-fetch
a.open('POST', 'http://localhostexample.com');
// ok: unencrypted-fetch
a.open('POST', 'http://localhost');
// ok: unencrypted-fetch
a.open('POST', 'http://localhost/');
// ok: unencrypted-fetch
a.open('POST', 'http://localhost:1234');
// ok: unencrypted-fetch
a.open('POST', 'http://localhost:1234/abc.txt');
// ok: unencrypted-fetch
a.open('POST', 'http://127.0.0.1');
// ok: unencrypted-fetch
a.open('POST', 'http://127.0.0.1/');
// ok: unencrypted-fetch
a.open('POST', 'http://127.0.0.1:1234');
// ok: unencrypted-fetch
a.open('POST', 'http://127.0.0.1:1234/abc.txt');

var b = new XMLHttpRequest();
// ok: unencrypted-fetch
b.open('GET', "https://example.com");
// ok: unencrypted-fetch
b.open('POST', 'https://example.com');

var c = new XMLHttpRequest();
// ok: unencrypted-fetch
c.open('GET', "test.txt");
// ok: unencrypted-fetch
c.open('POST', 'test.txt');

// ruleid: unencrypted-fetch
fetch('http://example.com');
// ruleid: unencrypted-fetch
fetch('http://example.com', {
    credentials: "include",
});
// ruleid: unencrypted-fetch
fetch('http://localhostexample.com');
// ok: unencrypted-fetch
fetch('http://localhost');
// ok: unencrypted-fetch
fetch('http://localhost/');
// ok: unencrypted-fetch
fetch('http://localhost:1234');
// ok: unencrypted-fetch
fetch('http://localhost:1234/abc.txt');
// ok: unencrypted-fetch
fetch('http://127.0.0.1');
// ok: unencrypted-fetch
fetch('http://127.0.0.1/');
// ok: unencrypted-fetch
fetch('http://127.0.0.1:1234');
// ok: unencrypted-fetch
fetch('http://127.0.0.1:1234/abc.txt');

// ok: unencrypted-fetch
fetch('https://example.com');
// ok: unencrypted-fetch
fetch('https://example.com', {
    credentials: "include",
});

// ok: unencrypted-fetch
fetch('test.txt');
// ok: unencrypted-fetch
fetch('test.txt', {
    credentials: "include",
});
