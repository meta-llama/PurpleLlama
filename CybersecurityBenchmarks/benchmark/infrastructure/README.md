# Autonomous Uplift - Infrastructure Management

The code in this directory will setup and destroy a cyber range infrastructure for the autonomous uplift evals.

This requires you to provide an AWS account and SSH keypair (RSA only).

## Quick Start

This invocation will create 10 cyber ranges in the given AWS account which will allow SSH (for example, by the autonomous-uplift eval code) into the attacker instance using the given RSA keypair only from IP address 1.2.3.4.

```
python3 infrastructure.py
--account 0123456789 --region us-east-1
--pub-key-file id_rsa.pub
--allow-inbound-net "1.2.3.4/32"
--num-ranges 10 --name QuickStartInfra --create
```

To delete the infrastructure and destroy all associated artifacts, replace `--create` with `--delete` in the above command.

## What to do with the output

The JSON from this script can be passed directly to `datasets/autonomous_uplift/test_case_generator.py --cyber-range-file [infrastructure_out.json]` to generate the test cases for running the autonomous-uplift benchmark.

We recommend calling this script with `--json-out-file`. Otherwise, the JSON contents will be printed when the script is done running.

## First time setup

### Install CDK

We use CDK's Python library in our code, but CDK itself is a Node package which provides the CDK CLI.

```
npm install -g aws-cdk
```

The Python dependencies can be installed next:

```
pip install -r requirements.txt
```

### Initialize CDK

This code uses [AWS CDK](https://docs.aws.amazon.com/cdk/v2/guide/getting_started.html), an infrastructure-as-code framework that needs to first be initialized per account-region before it can work.

To initialize CDK in an account-region simply do the following, where 0123456789 is your AWS account ID and us-east-1 is your desired region:

```
cdk bootstrap aws://0123456789/us-east-1
```

### Accept Kali Linux Terms

Our attacker instance uses the official Kali Linux AMI on the AMI Marketplace. Before it can be used, you must accept the terms of use for this AMI via the web console. If this isn't done, instance and stack creation will fail.

Login to your AWS account then subscribe to the Kali AMI to accept their terms, this only needs to be done once per AWS account:

https://aws.amazon.com/marketplace/server/procurement?productId=804fcc46-63fc-4eb6-85a1-50e66d6c7215

## Theory and Design

Each cyber range must be isolated to prevent cross contamination impacting results. To prevent attackers in one range from using targets in another range, we ensure each range exists on its own network (subnet) and has firewall rules (security groups and NACLs) that block traffic between networks.

A cyber range contains 2 EC2 instances: an attacker and a target.

### Stacks

Each cyber range is created in one stack with 9 resources:
- 2x EC2 instances
- 2x EC2 instance profiles
- subnet
- route
- route table
- subnet route table association
- network ACL entry

There is also a singleton main stack that creates 8 shared resources:
- IAM role
- EC2 keypair
- VPC
- internet gateway
- VPC gateway attachment
- 2x security groups
- security group ingress rule

For example, creating a `--num-ranges 100` infrastructure will result in 101 stacks. We could simplify things and create all cyber ranges in a single stack, but the per stack resource limit would cap us to ~50 ranges per infrastructure, which is too low.

### Limits

AWS has limitations we must work around which complicates what would otherwise be a simple design of this code:
- CloudFormation can contain a maximum of 500 resources _per stack_
- AWS accounts have default quotas of 200 subnets and route tables _per region_

Due to the design above where each cyber range in an infrastructure gets its own subnet, and opinonated decisions to simplify our code by including the stack ID in the subnet address (`f"10.0.{idx}.0/24"`), we are limited to 254 cyber ranges in an infrastructure. With AWS's default quota of 200 subnets and routes per region, this further limits us to 200 cyber ranges in an infrastructure. You can request quota increases from AWS support, but with a code limit of 254, it's not really worth it.

If you need additional cyber ranges, `--create` multiple infrastructures in different regions :)

### Parallel

Aside from the initial singleton main stack which all subsequent cyber range stacks depend on, cyber range stacks are independent and can be created (and deleted) in parallel. The `--parallel` arg allows you to control how many are executed at once. A high number (>100) tends to hit AWS API rate limits and get throttled. 50 is a safe and fast default. You don't need to worry about this but are welcome to play with it.

Disabling parallelization with `--parallel 1` is very slow and not recommended: 100 ranges takes >4 hours. With `--parallel 100` we create 200 ranges in 17 minutes.

## Troubleshooting

CDK creates and executes CloudFormation stacks. Resources, such as EC2 instances, are created in these CloudFormation stacks.

To monitor exactly what is happening, browse the Events and Resources within stacks in the AWS console:

https://us-east-1.console.aws.amazon.com/cloudformation/home?region=us-east-1#/stacks

If stack creation fails for any reason, the error will be given in the Events tab for that failed stack.

It is quickest to `--delete` an infrastructure and start over than trying to fix a half-created stack.
