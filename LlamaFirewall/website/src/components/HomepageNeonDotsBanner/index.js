/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

import { useEffect } from "react";
import styles from './styles.module.css';

export default function HomepageNeonDotsBanner() {
  useEffect(() => {
    function randomRange(min, max) {
      return Math.random() * (max - min) + min;
    }

    function createDot() {
      const dot = document.createElement('div');
      dot.classList.add(styles.dot);

      const size = randomRange(4, 12);
      const fromLeft = Math.random() < 0.5;
      const smallScreen = window.innerWidth <= 996;
      let startX = 0;
      if (smallScreen) {
        startX = fromLeft ? - window.innerWidth *0.02 : window.innerWidth *1.1;
      } else {
        startX = fromLeft ? -window.innerWidth * 0.15  : window.innerWidth * 1.02;
      }
      const startY = Math.random() * window.innerHeight;
      const targetOffset = window.innerWidth * (smallScreen? 0.4 : 0.55);
      const endX = fromLeft ? targetOffset + randomRange(-50, 50)
        : window.innerWidth - targetOffset + randomRange(-50, 50);

      const duration = randomRange(smallScreen? 1000 : 3000, smallScreen ? 4000 : 8000);
      const glowStrength = randomRange(10, 40);

      dot.style.width = `${size}px`;
      dot.style.height = `${size}px`;
      dot.style.left = `${startX}px`;
      dot.style.top = `${startY}px`;
      dot.style.boxShadow = `0 0 ${glowStrength/2}px #FF5F00, 0 0 ${glowStrength}px #FF5F00`;

      const parent = document.getElementById('neon-hero');
      parent.appendChild(dot);

      const animation = dot.animate([
        {
          transform: `translate(0, 0) scale(1)`,
          opacity: 1
        },
        {
          transform: `translateX(${endX - startX}px) scale(0.3)`,
          opacity: 0
        }
      ], {
        duration: duration,
        easing: 'ease-out',
        fill: 'forwards'
      });

      const blockerBar = document.getElementById('hero-blocker');

      function checkCollision() {
        if (!dot.isConnected || smallScreen) return;

        const dotRect = dot.getBoundingClientRect();
        const blockerRect = blockerBar?.getBoundingClientRect();

        if (!blockerRect) return;

        const collision = !(
          dotRect.right < blockerRect.left ||
          dotRect.left > blockerRect.right ||
          dotRect.bottom < blockerRect.top ||
          dotRect.top > blockerRect.bottom
        );

        if (collision) {
          animation.cancel();
          dot.remove();
        } else {
          requestAnimationFrame(checkCollision);
        }
      }

      if (!fromLeft) {
        requestAnimationFrame(checkCollision);
      }

      setTimeout(() => {
        dot.remove();
      }, duration);
    }

    const interval = setInterval(createDot, 10);
    for (let i = 0; i < 5; i++) createDot();

    return () => clearInterval(interval);
  }, []);

  return (
    <div className={styles.neonHero} id="neon-hero"/>
  );
};
