/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

import clsx from 'clsx';
import Heading from '@theme/Heading';
import styles from '../styles.module.css';
import { useEffect, useRef } from 'react';

export default function HomepageDemo() {
    const titleRef = useRef(null);
  useEffect(() => {
    const observer = new IntersectionObserver((entries) => {
      if (entries[0].isIntersecting) {
        titleRef.current.classList.add(styles.fadeIn);
      }
    }, { rootMargin: '0px 0px -150px 0px' });
    observer.observe(titleRef.current);
    return () => {try {return observer.unobserve(titleRef.current);}
    catch (e) {return null;}}
  }, []);

  return (
    <section >
  <div className={styles.container}>
      <div className={styles.titleContainer}>
        <h2 ref={titleRef} className={styles.title}>LlamaFirewall in Action</h2>
      </div>
      <div className={styles.videoWrapper}>
        <video
          id="autoplay-video"
          autoPlay
          muted
          loop
          playsInline
          className={styles.video}
        >
          <source src="img/llamafirewall_demo1.webm" type="video/webm"/>
          Your browser does not support the video tag.
        </video>
        <div className={styles.footer}>
            <p>Usage of an email-based agent application without any protection in place, and with LlamaFirewall enabled.</p>
        </div>
      </div>
    </div>
    </section>
  );
}
