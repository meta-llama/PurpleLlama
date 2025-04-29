import styles from './styles.module.css';

# LlamaFirewall Demos

Be witness to the power of LlamaFirewall in a couple real-world scenarios where protecting against harmful inputs and outputs is crucial.

## Demo 1: Email-Based Agent Application

In this demo, we showcase how LlamaFirewall protects an email-based agent application. The agent is designed to process incoming emails and generate appropriate responses. LlamaFirewall ensures that any malicious content within the emails is detected and neutralized before it can affect the agent's operations. This demo illustrates first what happens when LlamaFirewall is not used, and then how it protects the agent from potential harm.

<div
  className={styles.videoContainer}
>
<video className={styles.video} controls>
  <source src="../img/llamafirewall_demo1.webm" type="video/webm"/>
  Your browser does not support the video tag.
</video>
</div>

## Demo 2: Code Assistant Agent

This demo shows the application of LlamaFirewall in a SWE agent. While the agent assists the user by generating code snippets or using provided external resources, LlamaFirewall monitors the input and output to prevent dangerous additions like insecure code or malicious instructions. This ensures that the code assistant remains a reliable and secure tool.

<div
  className={styles.videoContainer}
>
<video className={styles.video} controls>
  <source src="../img/llamafirewall_demo2.webm" type="video/webm"/>
  Your browser does not support the video tag.
</video>
</div>
