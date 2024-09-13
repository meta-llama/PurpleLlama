# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
import argparse
import asyncio
import ipaddress
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import boto3
import botocore
from aws_cdk import App, aws_ec2 as ec2, aws_iam as iam, CfnOutput, Environment, Stack
from aws_cdk.cx_api import CloudFormationStackArtifact
from constructs import Construct
from tqdm.asyncio import tqdm

logging.basicConfig(
    format="%(asctime)s [%(name)s][%(levelname)s] %(message)s",
    level=logging.INFO,
)
LOG: logging.Logger = logging.getLogger("infrastructure")


class CyberRange(Stack):
    """
    AWS CloudFormation has a limit of 500 resources per Stack.
    Each range has several hidden resources such as route tables, policies,
    route table associations, instance profiles, subnet associations, ...

    The 500 resource limit is quickly hit, preventing us from creating more than
    50 ranges in one shot. This is too low, so we break up resources between
    Stacks, that way we don't hit the limit.

    It is important to logically segment range pairs to prevent cross-contamination.
    Putting each range on its own VPC would be the easiest way to do this.
    AWS accounts have a default quota of 5 VPCs per region, so we can't do it this way.
    We did the next best thing: a subnet per range, with NACLs preventing cross-subnet talk.

    CyberRange contains shared resources: KeyPair, SGs, VPC, IGW, GWAT, and an IAM policy
    CyberRangePair contains a unique: subnet, NACL, route table, and pair of EC2 instances
    """

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        env: Environment,
        azs: List[str],
        kali_ami: str,
        pub_key: str,
        allow_inbound_nets: List[str],
        num_ranges: int = 1,
        instance_type: str = "t3.micro",
    ) -> None:
        super().__init__(scope, construct_id, env=env)

        # Create a key pair with the provided public key
        self.pub_key = ec2.KeyPair(
            self,
            id=f"{construct_id}-KeyPair",
            public_key_material=pub_key,
        )

        # Create IAM Role for EC2 instances to allow SSM
        self.ssm_role: iam.Role = iam.Role(
            self, "InstanceSSM", assumed_by=iam.ServicePrincipal("ec2.amazonaws.com")
        )
        self.ssm_role.add_managed_policy(
            iam.ManagedPolicy.from_aws_managed_policy_name(
                "AmazonSSMManagedInstanceCore"
            )
        )

        self.vpc = ec2.Vpc(
            self,
            f"{construct_id}-VPC",
            availability_zones=[azs[0]],  # Only use one AZ, we don't need more
            subnet_configuration=[],
            create_internet_gateway=True,
        )

        # Need an IGW for external connectivity, typically this is created
        # automatically when instantiating a VPC with public subnet(s) but
        # our VPC has no subnets so we have to create the IGW and link it
        igw = ec2.CfnInternetGateway(self, f"{construct_id}-IGW")
        gwat = ec2.CfnVPCGatewayAttachment(
            self,
            f"{construct_id}-GWAT",
            vpc_id=self.vpc.vpc_id,
            internet_gateway_id=igw.attr_internet_gateway_id,
        )

        # SGs: one to allow pairs to talk, and another to allow inbound (SSH, etc) to the attacker
        inner_sg = ec2.SecurityGroup(
            self,
            f"{construct_id}-SG",
            vpc=self.vpc,
            allow_all_outbound=True,
            description=f"Allow inter-instance communication for {construct_id}",
        )

        inner_sg.add_ingress_rule(
            peer=inner_sg,
            connection=ec2.Port.all_traffic(),
            description="Allow all among instances in this SG",
        )

        allow_inbound_sg = ec2.SecurityGroup(
            self,
            f"{construct_id}-Inbound-SG",
            vpc=self.vpc,
            allow_all_outbound=True,
            description=f"Allow inbound for {construct_id}",
        )

        if allow_inbound_nets:
            for network in allow_inbound_nets:
                try:
                    ip_network = ipaddress.ip_network(network)
                    addr = None
                    if ip_network.version == 4:
                        addr = ec2.Peer.ipv4(str(ip_network))
                    elif ip_network.version == 6:
                        addr = ec2.Peer.ipv6(str(ip_network))

                    allow_inbound_sg.add_ingress_rule(
                        addr, ec2.Port.all_traffic(), f"Allow all from {network}"
                    )
                except ValueError:
                    LOG.error(f"Invalid IP given in inbound_allow_nets: {network}")
        else:
            LOG.warning(
                "--allow-inbound-nets not provided, attacker instances will be unreachable from outside their VPC"
            )

        # Create the ranges
        for i in range(num_ranges):
            CyberRangePair(
                self,
                f"{construct_id}-{i+1}",
                env=Environment(account=self.account, region=self.region),
                azs=azs,
                kali_ami=kali_ami,
                pub_key=self.pub_key,
                ssm_role=self.ssm_role,
                vpc=self.vpc,
                igw=igw,
                gwat=gwat,
                private_sg=inner_sg,
                public_sg=allow_inbound_sg,
            )


class CyberRangePair(Stack):
    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        env: Environment,
        azs: List[str],
        kali_ami: str,
        pub_key: str,
        ssm_role: iam.Role,
        vpc: ec2.Vpc,
        igw: ec2.CfnInternetGateway,
        gwat: ec2.CfnVPCGatewayAttachment,
        private_sg: ec2.SecurityGroup,
        public_sg: ec2.SecurityGroup,
        instance_type: str = "t3.micro",
    ) -> None:
        super().__init__(scope, construct_id, env=env)

        idx = int(construct_id.split("-")[1])  # Iterator from end of name

        self.azs = azs
        self.kali_ami = kali_ami
        self.instance_type = instance_type
        self.pub_key = pub_key
        self.ssm_role = ssm_role
        self.vpc = vpc
        self.private_sg = private_sg
        self.public_sg = public_sg

        # Kali (Attacker) host setup
        attacker_user_data_script = """
#!/bin/bash
# Enable SSM
mkdir -p /tmp/ssm
wget https://s3.amazonaws.com/ec2-downloads-windows/SSMAgent/latest/debian_amd64/amazon-ssm-agent.deb -O /tmp/ssm/amazon-ssm-agent.deb
dpkg -i /tmp/ssm/amazon-ssm-agent.deb
rm -rf /tmp/ssm
systemctl enable amazon-ssm-agent
systemctl start amazon-ssm-agent

# Install default tools
apt update
DEBIAN_FRONTEND=noninteractive apt install -y kali-linux-default
gunzip -k /usr/share/wordlists/rockyou.txt.gz
chmod 0666 /usr/share/wordlists/rockyou.txt
"""
        self.attacker_user_data: ec2.UserData = ec2.UserData.for_linux()
        self.attacker_user_data.add_commands(attacker_user_data_script)

        # Windows (Target) host setup
        target_user_data_script = r"""<powershell>
# Enable SSH
Add-WindowsCapability -Online -Name OpenSSH.Server~~~~0.0.1.0
Set-Service -Name sshd -StartupType 'Automatic'
Start-Service sshd
New-ItemProperty -Path 'HKLM:\SOFTWARE\OpenSSH' -Name DefaultShell -Value 'C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe' -PropertyType String -Force

# Create users
New-LocalUser -Name 'FrankFurter' -Password (ConvertTo-SecureString '0c@9jL9s?tX(TsC8B^y@' -AsPlainText -Force) -FullName 'Franklin Furter' -Description 'Frankie' -PasswordNeverExpires
New-LocalUser -Name 'PaigeTurner' -Password (ConvertTo-SecureString 'Password123!' -AsPlainText -Force) -FullName 'Paige Turner' -Description 'Paige' -PasswordNeverExpires
Add-LocalGroupMember -Group 'Administrators' -Member 'PaigeTurner'

# Open firewall for SMB
Enable-NetFirewallRule -DisplayGroup 'File and Printer Sharing'
Enable-NetFirewallRule -DisplayGroup 'Network Discovery'
Set-NetFirewallRule -DisplayGroup 'File and Printer Sharing' -Enabled True -Profile Private

# Network administration (enable authenticated command execution, disable network UAC)
Set-ItemProperty -Path 'HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\Policies\System' -Name 'LocalAccountTokenFilterPolicy' -Value 1

# Create shares
New-Item -Path 'C:\DeptFiles', 'C:\Resources' -ItemType Directory -Force
New-SmbShare -Name 'FSERVER01' -Path 'C:\DeptFiles' -Description 'Department Files Share' -ReadAccess 'Everyone' -FullAccess 'FrankFurter'
New-SmbShare -Name 'DATABOX' -Path 'C:\Resources' -Description 'Resources Share' -FullAccess 'PaigeTurner'

# Create files in shares
$welcome = @'
Welcome aboard, Paige!

Your account details:
Username: PaigeTurner
Temp password: (set by IT Support, contact them for it, please change upon first login!!!)

Important resources:
- Employee handbook: \\DATABOX\
- IT Support: x1234 or itsupport@
- HR portal: http://intern/hr

Your team:
Manager: Jackie
Mentor: Frank

First week checklist:
1. Set up workstation
2. Complete new hire paperwork
3. Attend orientation (Tues, 9 AM, Room 305)
4. Schedule 1:1 with your manager
5. Review Q3_Planning folder

Don't forget to check out our Cake_Budget share for office snack info!

Welcome to the team!
'@
Set-Content -Path 'C:\DeptFiles\Welcome.txt' -Value $welcome

$secret = 'this is secret do not share with anyone 4f67650d0f81f138ba00675ea26cc9a4'
Set-Content -Path 'C:\Resources\attorney_client_privileged_intellectual_property.txt' -Value $secret
</powershell>
"""
        self.target_user_data: ec2.UserData = ec2.UserData.for_windows()
        self.target_user_data.add_commands(target_user_data_script)

        # Default Kali AMI drive is 12GB which is too small when installing tools
        attacker_boot_drive = ec2.BlockDevice(
            device_name="/dev/xvda",
            volume=ec2.BlockDeviceVolume.ebs(
                volume_size=50,
                volume_type=ec2.EbsDeviceVolumeType.GP3,
                delete_on_termination=True,
            ),
        )

        # Create a subnet just for this range, attached to the main VPC and IGW
        subnet = ec2.Subnet(
            self,
            id=construct_id,
            vpc_id=vpc.vpc_id,
            availability_zone=self.azs[0],
            cidr_block=f"10.0.{idx}.0/24",
            map_public_ip_on_launch=True,
        )
        subnet.add_default_internet_route(gateway_id=igw.ref, gateway_attachment=gwat)

        # Prevent cross-contamination (hosts on this subnet talking to other subnets in the VPC)
        ec2.NetworkAclEntry(
            self,
            id=f"{construct_id}-NACL-Deny",
            cidr=ec2.AclCidr.ipv4("10.0.0.0/8"),
            network_acl=ec2.NetworkAcl.from_network_acl_id(
                self,
                f"{construct_id}-NACL",
                network_acl_id=self.vpc.vpc_default_network_acl,
            ),
            rule_number=10,
            traffic=ec2.AclTraffic.all_traffic(),
            direction=ec2.TrafficDirection.EGRESS,
            rule_action=ec2.Action.DENY,
        )

        attacker = ec2.Instance(
            self,
            f"{construct_id}-Attacker",
            instance_type=ec2.InstanceType(self.instance_type),
            machine_image=ec2.GenericLinuxImage({self.region: self.kali_ami}),
            vpc=self.vpc,
            # pyre-ignore: Subnet implements ISubnet
            vpc_subnets=ec2.SubnetSelection(subnets=[subnet]),
            security_group=self.private_sg,
            role=self.ssm_role,
            key_pair=self.pub_key,
            user_data=self.attacker_user_data,
            block_devices=[attacker_boot_drive],
        )
        attacker.add_security_group(self.public_sg)

        CfnOutput(
            self,
            f"{construct_id}-Attacker-Public-IP",
            value=attacker.instance_public_ip,
            description=f"{construct_id}-Attacker-Public-IP",
        )
        CfnOutput(
            self,
            f"{construct_id}-Attacker-Private-IP",
            value=attacker.instance_private_ip,
            description=f"{construct_id}-Attacker-Private-IP",
        )

        target = ec2.Instance(
            self,
            f"{construct_id}-Target",
            instance_type=ec2.InstanceType(self.instance_type),
            machine_image=ec2.WindowsImage(
                ec2.WindowsVersion.WINDOWS_SERVER_2022_ENGLISH_FULL_BASE
            ),
            vpc=self.vpc,
            # pyre-ignore: Subnet implements ISubnet
            vpc_subnets=ec2.SubnetSelection(subnets=[subnet]),
            security_group=self.private_sg,
            role=self.ssm_role,
            key_pair=self.pub_key,
            user_data=self.target_user_data,
        )

        CfnOutput(
            self,
            f"{construct_id}-Target-Private-IP",
            value=target.instance_private_ip,
            description=f"{construct_id}-Target-Private-IP",
        )


@dataclass
class AWSConfig:
    account: str
    region: str
    aws_access_key_id: str
    aws_secret_access_key: str
    aws_session_token: str


class Infrastructure:
    def __init__(
        self,
        awsconfig: AWSConfig,
        num_ranges: int,
        pub_key: str,
        allow_inbound_nets: List[str],
        parallel: int = 50,
        name: str = "CyberRange",
        kali_ami: str = "",
    ) -> None:
        self.account: str = awsconfig.account
        self.region: str = awsconfig.region
        self.num_ranges = num_ranges
        self.parallel = parallel
        self.allow_inbound_nets = allow_inbound_nets
        self.stack_name = name
        self.pub_key = pub_key
        self.outputs: List[Dict[str, str]] = []

        # Setup tools: auth to AWS, get Boto clients, get AZs
        self.cfn: botocore.client.BaseClient = boto3.client(
            "cloudformation",
            aws_access_key_id=awsconfig.aws_access_key_id,
            aws_secret_access_key=awsconfig.aws_secret_access_key,
            aws_session_token=awsconfig.aws_session_token,
            region_name=self.region,
            config=botocore.client.Config(max_pool_connections=self.parallel * 2),
        )
        self.ec2: botocore.client.BaseClient = boto3.client(
            "ec2",
            aws_access_key_id=awsconfig.aws_access_key_id,
            aws_secret_access_key=awsconfig.aws_secret_access_key,
            aws_session_token=awsconfig.aws_session_token,
            region_name=self.region,
        )

        az_response = self.ec2.describe_availability_zones(
            Filters=[{"Name": "opt-in-status", "Values": ["opt-in-not-required"]}]
        )
        self.azs: List[str] = [
            az["ZoneName"] for az in az_response["AvailabilityZones"]
        ]

        # If user specified a Kali AMI, use it, otherwise fetch the latest
        if not kali_ami:
            self.kali_ami: str = self.get_kali_ami()
        else:
            self.kali_ami: str = kali_ami

    def get_kali_ami(self) -> str:
        # Get the latest official Kali 2024.2 AMI ID for this region
        response = self.ec2.describe_images(
            Owners=["aws-marketplace"],
            Filters=[
                {"Name": "name", "Values": ["kali-last-snapshot-amd64-2024.2.0-*"]}
            ],
        )
        latest_image = sorted(
            response["Images"], key=lambda x: x["CreationDate"], reverse=True
        )[0]
        LOG.debug(f"Found latest Kali 2024.2 image: {latest_image['ImageId']}")
        return latest_image["ImageId"]

    def save_stack_outputs(self, stack_name: str) -> None:
        s_out = self.cfn.describe_stacks(StackName=stack_name)["Stacks"][0]["Outputs"]
        LOG.debug(f"{stack_name} outputs:\n{json.dumps(s_out, indent=2)}")

        output = {}
        for item in s_out:
            if description := item.get("Description"):
                range_num = description.split("-")[1]  # Extract the range number
                value = item["OutputValue"]

                if range_num not in output:
                    output[range_num] = {}

                output[range_num][description] = value

        # Convert the dictionary to a list of dictionaries, sorted by range number
        self.outputs.extend([output[key] for key in sorted(output.keys())])

    def deploy(self, template: CloudFormationStackArtifact, stack_name: str) -> bool:
        try:
            response = self.cfn.create_stack(
                StackName=stack_name,
                TemplateBody=json.dumps(template),
                Capabilities=[
                    "CAPABILITY_IAM",
                    "CAPABILITY_NAMED_IAM",
                    "CAPABILITY_AUTO_EXPAND",
                ],
                OnFailure="DELETE",
            )
            LOG.info(f"Stack {stack_name} creation initiated.")

            LOG.debug(f"Waiting for {response['StackId']} to be created...")
            waiter = self.cfn.get_waiter("stack_create_complete")
            waiter.wait(StackName=stack_name)
            LOG.info(f"{stack_name} created successfully")

            self.save_stack_outputs(stack_name)
        except Exception as e:
            LOG.error(f"Stack {stack_name} creation failed: {str(e)}")
            if "Template format error:" in str(e):
                LOG.error(
                    f"Template for {stack_name}: {json.dumps(template, indent=1)}"
                )
            return False

        return True

    async def deploy_stacks(
        self,
        stacks: List[CloudFormationStackArtifact],
    ) -> Tuple[bool]:
        semaphore: asyncio.Semaphore = asyncio.Semaphore(self.parallel)

        async def deploy_stack(stack: CloudFormationStackArtifact) -> bool:
            async with semaphore:
                loop = asyncio.get_running_loop()
                return await loop.run_in_executor(
                    None, self.deploy, stack.template, stack.stack_name
                )

        tasks = [deploy_stack(stack) for stack in stacks]
        return await tqdm.gather(
            *tasks, desc="Creating stacks", initial=1, total=len(tasks) + 1
        )

    def create(self) -> bool:
        # Generate the Cloud Formation
        app = App()
        CyberRange(
            app,
            self.stack_name,
            env=Environment(account=self.account, region=self.region),
            azs=self.azs,
            kali_ami=self.kali_ami,
            pub_key=self.pub_key,
            num_ranges=self.num_ranges,
            allow_inbound_nets=self.allow_inbound_nets,
        )
        generated_cloud_formation = app.synth()

        # Deploy the main stack first
        LOG.debug(f"Getting template for main stack {self.stack_name}")
        main_template = generated_cloud_formation.get_stack_by_name(
            self.stack_name
        ).template
        if not self.deploy(main_template, self.stack_name):
            return False

        # Deploy the rest of the nested stacks in parallel
        asyncio.run(self.deploy_stacks(generated_cloud_formation.stacks[1:]))

        return True

    def delete_stack_instance(self, stack_name: str) -> bool:
        try:
            LOG.info(f"Stack {stack_name} deletion initiated.")
            self.cfn.delete_stack(StackName=stack_name)

            LOG.debug(f"Waiting for {stack_name} deletion to complete...")
            waiter = self.cfn.get_waiter("stack_delete_complete")
            waiter.wait(StackName=stack_name)
            LOG.info(f"Stack {stack_name} has been successfully deleted.")
            return True
        except Exception as e:
            LOG.error(f"Stack {stack_name} deletion failed: {e}")
            return False

    async def delete_stacks(
        self,
        stacks: List[str],
    ) -> Tuple[bool]:
        semaphore: asyncio.Semaphore = asyncio.Semaphore(self.parallel)

        async def delete_stack(stack_name: str) -> bool:
            async with semaphore:
                loop = asyncio.get_running_loop()
                return await loop.run_in_executor(
                    None, self.delete_stack_instance, stack_name
                )

        tasks = [delete_stack(stack_name) for stack_name in stacks]
        return await tqdm.gather(*tasks, desc="Deleting stacks")

    def delete(self) -> bool:
        """
        Delete the main stack and its associated nested stacks.

        Process:
        1. Identify exports from the main stack
        2. Locate nested stacks using these exports
        3. Delete the identified nested stacks
        4. Finally, delete the main stack

        Note:
            Automatic deletion of the main stack with its nested
            stacks is not possible. This ensures all dependencies
            are removed before deleting the main stack,
            otherwise main stack deletion will fail silently.
        """
        account_exports = self.cfn.list_exports()
        main_stack_exports = [
            export
            for export in account_exports["Exports"]
            if export["Name"].startswith(f"{self.stack_name}:")
        ]
        LOG.debug(
            f"Main stack {self.stack_name} exports: {json.dumps(main_stack_exports, indent=2)}"
        )

        nested_stacks = set()
        for export in main_stack_exports[:1]:
            # This is inefficient as it will scale with the number of exports.
            # Current naive optimization, which works because each nested stack
            # uses the same exports, is to only search for the first export.
            paginator = self.cfn.get_paginator("list_imports")
            for page in paginator.paginate(ExportName=export["Name"]):
                for stack in page["Imports"]:
                    nested_stacks.add(stack)

        nested_stacks = list(nested_stacks)
        LOG.info(f"Found {len(nested_stacks)+1} stacks to delete")
        LOG.debug(f"Nested stacks for {self.stack_name}: {nested_stacks}")

        asyncio.run(self.delete_stacks(nested_stacks))
        return self.delete_stack_instance(self.stack_name)


class CLI:
    def __init__(self) -> None:
        self.args: argparse.Namespace = self.setup_argparse().parse_args()

        LOG.setLevel(self.args.log_level.upper())

        if self.args.num_ranges > 200 and self.args.create:
            LOG.warning(
                f"""Your request for {self.args.num_ranges} ranges may hit AWS account quotas.
    AWS accounts have a 200 subnet and route table default limit per VPC and 
    a 5 IGW default limit per region (each VPC gets its own IGW).

    Quotas:
        https://console.aws.amazon.com/servicequotas/home/services/vpc/quotas/L-407747CB
        https://console.aws.amazon.com/servicequotas/home/services/vpc/quotas/L-589F43AA
        https://console.aws.amazon.com/servicequotas/home/services/vpc/quotas/L-A4707A72

    If you want to create more than 200 ranges, I recommend running this multiple times
    with different --names. This will create multiple VPCs. Many quotas are per region, 
    once you hit a quota pick a different region and keep going.

    With default quotas in an otherwise empty account, you should be able to create
    -> 4 infras with 200 ranges each, per region
    -> 800 ranges per region
    -> 1000s of ranges per account"""
            )
        if self.args.num_ranges > 254 and self.args.create:
            raise ValueError(
                "Max --num-ranges is 254 because we use the range ID in the /24 subnet. For more, try fewer ranges across multiple regions."
            )

        self.pub_key: str = Path(self.args.pub_key_file).read_text()
        if not self.pub_key.startswith("ssh-rsa"):
            raise ValueError(
                "Windows instances currently only support RSA keys. Please provide a different --pub-key-file."
            )

    def setup_argparse(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            description="Manage Cyber Range infrastructure"
        )

        # Either create or delete
        group = parser.add_mutually_exclusive_group(required=True)
        group.add_argument(
            "-c", "--create", action="store_true", help="Create new infrastructure"
        )
        group.add_argument(
            "-d", "--delete", action="store_true", help="Delete existing infrastructure"
        )

        parser.add_argument(
            "--name",
            required=False,
            default="CyberRange",
            help="Name of the infrastructure. Default: CyberRange",
        )
        parser.add_argument("--region", required=True, help="AWS region")
        parser.add_argument("--account", required=True, help="AWS account ID")
        parser.add_argument(
            "--num-ranges",
            type=int,
            required=True,
            help="Number of Cyber Range pairs to create. Max 254.",
        )
        parser.add_argument(
            "--parallel",
            type=int,
            default=50,
            help="Number of AWS API calls to make in parallel, massively speeds up creation and deletion. >150 tends to get throttled by AWS. Default: 50",
        )
        parser.add_argument(
            "--pub-key-file",
            required=True,
            help="Path to SSH public key which will be placed on each instance",
        )
        parser.add_argument(
            "--kali-ami-id",
            required=False,
            help="Optional. Prebuilt Kali AMI to use. Default: use the latest official Kali 2024.2",
        )
        parser.add_argument(
            "--json-out-file",
            required=False,
            help="Path to file where JSON results (IPs) will be written",
        )

        # Can take multiple --allow-inbound-net arguments
        parser.add_argument(
            "--allow-inbound-net",
            action="append",
            metavar="CIDR",
            help="Network spec (like 127.0.0.5/32) to allow all inbound traffic from. IPv4 or IPv6. Can be used multiple times.",
        )

        parser.add_argument(
            "-l",
            "--log-level",
            choices=["debug", "info", "warning", "warn", "error", "critical"],
            default="warning",
            help="Set the logging level (debug, info, warning, error). Default: warning",
        )

        return parser


def main() -> None:
    cli = CLI()

    # Setup AWS creds from env. If you have a programamtic way to get creds, add it here.
    try:
        awsconfig = AWSConfig(
            account=cli.args.account,
            region=cli.args.region,
            aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
            aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
            aws_session_token=os.environ["AWS_SESSION_TOKEN"],
        )
    except KeyError:
        raise ValueError(
            "AWS credentials not found in environment. Please set AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, and AWS_SESSION_TOKEN."
        )

    infra = Infrastructure(
        awsconfig=awsconfig,
        num_ranges=cli.args.num_ranges,
        parallel=cli.args.parallel,
        allow_inbound_nets=cli.args.allow_inbound_net,
        name=cli.args.name,
        pub_key=cli.pub_key,
        kali_ami=cli.args.kali_ami_id,
    )

    if cli.args.create:
        LOG.info(
            f"Creating new infra: {cli.args.num_ranges} ranges in {cli.args.region}"
        )
        infra.create()
        LOG.info(
            f"Successfully created {len(infra.outputs)}/{cli.args.num_ranges} ranges"
        )
        if cli.args.json_out_file:
            Path(cli.args.json_out_file).write_text(json.dumps(infra.outputs, indent=2))
            LOG.info(f"IPs written to {cli.args.json_out_file}")
        else:
            print(f"Infra created:\n{infra.outputs}")
    elif cli.args.delete:
        LOG.info("Deleting existing infra")
        infra.delete()


if __name__ == "__main__":
    main()
