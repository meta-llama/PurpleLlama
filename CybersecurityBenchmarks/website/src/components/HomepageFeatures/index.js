/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

import clsx from 'clsx';
import Heading from '@theme/Heading';
import styles from './styles.module.css';

const FeatureList = [
  {
    title: 'CyberSecEval 4',
    description: (
      <>
        This version introduces AutoPatchBench, a benchmark that measures an LLM agent's capability to automatically patch security vulnerabilities in native code.
      </>
    ),
  },
  {
    title: 'Prompt Guard',
    description: (
      <>
        Prompt Guard is a new model for guardrailing LLM inputs against prompt attacks - in particular jailbreaking techniques and indirect injections embedded into third party data. For more information, see our Model card.
      </>
    ),
  },
];

function Feature({title, description}) {
  return (
    <div className={clsx('col col--6')}>
      <div className="text--center padding-horiz--md">
        <Heading as="h3">{title}</Heading>
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
