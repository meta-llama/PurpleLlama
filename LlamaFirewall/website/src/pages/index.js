/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import HomepageFeatures from '@site/src/components/HomepageFeatures';
import HomepageDemo from '@site/src/components/HomepageDemo';
import HomepageTutorial from '@site/src/components/HomepageTutorial';
import HomepageNeonDotsBanner from '@site/src/components/HomepageNeonDotsBanner';

import Heading from '@theme/Heading';
import styles from './index.module.css';

function HomepageHeader() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <header className={clsx('hero hero--primary', styles.heroBanner)}>
      <div className="heroContainer">
        <HomepageNeonDotsBanner />
        <div className="left">
          <Heading as="h1" className="hero__title">
            {siteConfig.title}
          </Heading>
          <p className="hero__subtitle">{siteConfig.tagline}</p>
        <Link
              className="button button--secondary button--lg"
              to="/docs/documentation/about-llamafirewall">
              Learn More
            </Link>
        </div>
        <div className="right">
          <div className={styles.logoWrapper}>
            <div id="hero-blocker" className={styles.blocker}/>
            <img src="img/llamafirewall_logo.png" alt="llamafirewall" />
          </div>
        </div>
      </div>
    </header>
  );
}

export default function Home() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout
      title={`${siteConfig.title}`}
      description="Description will go into a meta tag in <head />">
      <HomepageHeader />
      <main>
        <HomepageFeatures />
        <div className={styles.separator}></div>
        <HomepageDemo />
        <div className={styles.separator}></div>
        <HomepageTutorial />
      </main>
    </Layout>
  );
}
