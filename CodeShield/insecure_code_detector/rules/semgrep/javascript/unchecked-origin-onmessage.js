/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// ruleid: unchecked-origin-onmessage
window.addEventListener('message', function(event) {
    alert(event.data);
});

// ruleid: unchecked-origin-onmessage
window.addEventListener('message', function(event) {
    alert(event.data);
}, true);

// ruleid: unchecked-origin-onmessage
window.addEventListener('message', (event) => {
    alert(event.data);
});

// ruleid: unchecked-origin-onmessage
window.addEventListener('message', (event) => {
    alert(event.data);
}, true);

// ok: unchecked-origin-onmessage
window.addEventListener('message', function(event) {
    if (event.origin === 'https://example.com') {
        alert(event.data);
    }
});

// ok: unchecked-origin-onmessage
window.addEventListener('message', function(event) {
    if (event.origin === 'https://example.com') {
        alert(event.data);
    }
}, true);

// ok: unchecked-origin-onmessage
window.addEventListener('message', (event) => {
    if (event.origin === 'https://example.com') {
        alert(event.data);
    }
});

// ok: unchecked-origin-onmessage
window.addEventListener('message', (event) => {
    if (event.origin === 'https://example.com') {
        alert(event.data);
    }
}, true);

// ruleid: unchecked-origin-onmessage
window.onmessage = function(event) {
    alert(event.data);
};

// ruleid: unchecked-origin-onmessage
window.onmessage = (event) => {
    alert(event.data);
};

// ok: unchecked-origin-onmessage
window.onmessage = function(event) {
    if (event.origin === 'https://example.com') {
        alert(event.data);
    }
};

// ok: unchecked-origin-onmessage
window.onmessage = (event) => {
    if (event.origin === 'https://example.com') {
        alert(event.data);
    }
};
