/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

import clsx from 'clsx';
import Heading from '@theme/Heading';
import styles from '../styles.module.css';

const FeatureList = [
  {
    title: 'Easy Integration',
    img: 'img/integration_icon.png',
    description: (
      <>
        LlamaFirewall provides an effortless and seamless deployment into existing AI Agents and LLM applications.
      </>
    ),
  },
  {
    title: 'Latency Efficiency',
    img: 'img/latency_icon.png',
    description: (
      <>
        Optimized for minimal computational overhead, ensuring negligible impact on performance and user experience.
      </>
    ),
  },
  {
    title: 'Extensibility',
    img: 'img/customize_icon.png',
    description: (
      <>
        The framework offers developer-friendly customization and extension, allowing for tailored security measures.
      </>
    ),
  },
];

function Feature({img, title, description}) {
  return (
    <div className={clsx('col col--4')}>
      <div className="text--center">
      <img src={img} className="featureIcon" />
      </div>
      <div className="text--center padding-horiz--md">
        <h3>{title}</h3>
        <p>{description}</p>
      </div>
    </div>
  );
}

export default function HomepageFeatures() {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className="row">
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}
