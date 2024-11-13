import os
import json
from typing import List, Optional, Dict
import time
from botocore.exceptions import WaiterError, ClientError
import base64
import boto3
from autogen.coding import (
    CodeBlock,
    CodeExecutor,
    CodeExtractor,
    CodeResult,
    MarkdownCodeExtractor
)

class FargateCodeExecutor(CodeExecutor):
    """Code executor that runs code in AWS Fargate containers."""
    
    @property
    def code_extractor(self) -> CodeExtractor:
        """Extract code from markdown blocks."""
        return MarkdownCodeExtractor()

    def __init__(
        self,
        image_uri: str,
        subnet_ids: List[str],
        security_groups: List[str],
        region_name: str = 'us-west-2',
        cluster_name: str = 'autogen-executor-cluster',
        cpu: str = '256',
        memory: str = '512',
        timeout_seconds: int = 300,
        requirements_file: Optional[str] = None,
        pip_dependencies: Optional[List[str]] = None,
        environment_variables: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Initialize the Fargate code executor.
        
        Args:
            image_uri: Docker image URI (must have Python installed)
            subnet_ids: List of subnet IDs for task networking
            security_groups: List of security group IDs for task networking
            region_name: AWS region name
            cluster_name: ECS cluster name
            cpu: CPU units for the Fargate task ('256', '512', '1024', etc.)
            memory: Memory for the Fargate task ('512', '1024', '2048', etc.)
            timeout_seconds: Maximum execution time in seconds
            requirements_file: Optional path to requirements.txt file
            pip_dependencies: Optional list of pip packages to install
            environment_variables: Optional dictionary of environment variables
        """
        self.region_name = region_name
        self.cluster_name = cluster_name
        self.image_uri = image_uri
        self.subnet_ids = subnet_ids
        self.security_groups = security_groups
        self.cpu = cpu
        self.memory = memory
        self.timeout_seconds = timeout_seconds
        self.pip_dependencies = pip_dependencies
        self.environment_variables = environment_variables
        
        self.iam_client = boto3.client('iam', region_name=region_name)
        self.ecs_client = boto3.client('ecs', region_name=region_name)
        self.logs_client = boto3.client('logs', region_name=region_name)
        
        # Set up required resources
        self.task_execution_role_arn = self._ensure_task_execution_role()
        self._ensure_cluster_exists()
        self._ensure_log_group_exists()
        
        # Read requirements if provided
        self.requirements = None
        if requirements_file and os.path.exists(requirements_file):
            with open(requirements_file, 'r') as f:
                self.requirements = f.read()

    def _ensure_task_execution_role(self) -> str:
        """Create or get ECS task execution role."""
        role_name = 'ecsTaskExecutionRoleAutoGenFargate'
        try:
            response = self.iam_client.get_role(RoleName=role_name)
            print(f"Using existing ECS task execution role: {role_name}")
            return response['Role']['Arn']
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchEntity':
                print(f"Creating new task execution role: {role_name}")
                # Create role
                trust_policy = {
                    "Version": "2012-10-17",
                    "Statement": [{
                        "Sid": "",
                        "Effect": "Allow",
                        "Principal": {"Service": "ecs-tasks.amazonaws.com"},
                        "Action": "sts:AssumeRole"
                    }]
                }
                response = self.iam_client.create_role(
                    RoleName=role_name,
                    AssumeRolePolicyDocument=json.dumps(trust_policy)
                )
                
                # Attach managed policy
                self.iam_client.attach_role_policy(
                    RoleName=role_name,
                    PolicyArn='arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy'
                )
                
                return response['Role']['Arn']
            raise

    def _ensure_cluster_exists(self) -> None:
        """Create ECS cluster if it doesn't exist."""
        try:
            response = self.ecs_client.describe_clusters(clusters=[self.cluster_name])
            if not response['clusters']:
                print(f"Creating new ECS cluster: {self.cluster_name}")
                self.ecs_client.create_cluster(
                    clusterName=self.cluster_name,
                    capacityProviders=['FARGATE', 'FARGATE_SPOT']
                )
                waiter = self.ecs_client.get_waiter('cluster_active')
                waiter.wait(clusters=[self.cluster_name])
        except Exception as e:
            raise Exception(f"Failed to ensure cluster exists: {str(e)}")

    def _ensure_log_group_exists(self) -> None:
        """Create CloudWatch log group if it doesn't exist."""
        log_group_name = '/ecs/autogen-task'
        try:
            self.logs_client.create_log_group(logGroupName=log_group_name)
            print(f"Created new log group: {log_group_name}")
        except self.logs_client.exceptions.ResourceAlreadyExistsException:
            pass

    def _create_or_update_task_definition(self, code: str) -> str:
        """Create or update ECS task definition with inline code execution and dependency installation."""
        encoded_code = base64.b64encode(code.encode()).decode()
        
        # Create the installation script for dependencies
        install_commands = []
        
        # Add requirements.txt dependencies
        if self.requirements:
            encoded_requirements = base64.b64encode(self.requirements.encode()).decode()
            install_commands.append(f"echo {encoded_requirements} | base64 -d > requirements.txt")
            install_commands.append("PYTHONWARNINGS=ignore pip install --no-warn-script-location --disable-pip-version-check -q -r requirements.txt > /dev/null 2>&1 || (echo 'Error installing requirements.txt packages' && PYTHONWARNINGS=ignore pip install --no-warn-script-location --disable-pip-version-check -r requirements.txt)")
        
        # Add pip dependencies from config
        if self.pip_dependencies:
            deps_str = ' '.join(self.pip_dependencies)
            install_commands.append(f"PYTHONWARNINGS=ignore pip install --no-warn-script-location --disable-pip-version-check -q {deps_str} > /dev/null 2>&1 || (echo 'Error installing pip dependencies' && PYTHONWARNINGS=ignore pip install --no-warn-script-location --disable-pip-version-check {deps_str})")
        
        install_script = '\n'.join(install_commands)
        
        # Combine installation and code execution
        full_script = f"""
    set -e
    # Suppress all pip warnings and notices
    export PIP_NO_WARN_SCRIPT_LOCATION=0
    export PIP_DISABLE_PIP_VERSION_CHECK=1
    export PYTHONWARNINGS=ignore

    {install_script}
    python -c 'import base64; exec(base64.b64decode("{encoded_code}").decode())'
    """
        
        container_definition = {
            'name': 'code-runner',
            'image': self.image_uri,
            'essential': True,
            'command': ['/bin/bash', '-c', full_script],
            'logConfiguration': {
                'logDriver': 'awslogs',
                'options': {
                    'awslogs-group': '/ecs/autogen-task',
                    'awslogs-region': self.region_name,
                    'awslogs-stream-prefix': 'ecs'
                }
            }
        }
        
        # Add environment variables if provided
        if self.environment_variables:
            container_definition['environment'] = [
                {'name': k, 'value': v}
                for k, v in self.environment_variables.items()
            ]
        
        try:
            response = self.ecs_client.register_task_definition(
                family='autogen-task',
                requiresCompatibilities=['FARGATE'],
                networkMode='awsvpc',
                cpu=self.cpu,
                memory=self.memory,
                executionRoleArn=self.task_execution_role_arn,
                containerDefinitions=[container_definition]
            )
            return response['taskDefinition']['taskDefinitionArn']
        except Exception as e:
            raise Exception(f"Failed to register task definition: {str(e)}")

    def execute_code_blocks(self, code_blocks: List[CodeBlock]) -> CodeResult:
        """
        Execute a list of code blocks in a Fargate container.
        
        Args:
            code_blocks: List of CodeBlock objects containing the code to execute
            
        Returns:
            CodeResult: Object containing the execution results
        """
        try:
            # Combine all code blocks
            combined_code = '\n'.join(block.code for block in code_blocks)
            
            # Create or update task definition
            task_definition_arn = self._create_or_update_task_definition(combined_code)
            
            # Run the task
            response = self.ecs_client.run_task(
                cluster=self.cluster_name,
                taskDefinition='autogen-task',
                launchType='FARGATE',
                networkConfiguration={
                    'awsvpcConfiguration': {
                        'subnets': self.subnet_ids,
                        'securityGroups': self.security_groups,
                        'assignPublicIp': 'ENABLED'
                    }
                }
            )
            
            task_arn = response['tasks'][0]['taskArn']
            task_id = task_arn.split('/')[-1]
            print(f"Started ECS task: {task_id}")
            
            waiter = self.ecs_client.get_waiter('tasks_stopped')
            waiter.wait(
                cluster=self.cluster_name,
                tasks=[task_arn],
                WaiterConfig={
                    'Delay': 5,
                    'MaxAttempts': self.timeout_seconds // 5
                }
            )
            
            task_details = self.ecs_client.describe_tasks(
                cluster=self.cluster_name,
                tasks=[task_arn]
            )['tasks'][0]
            
            exit_code = task_details['containers'][0].get('exitCode', 1)
            
            time.sleep(5)  # Wait for logs to be available
            
            try:
                logs = self.logs_client.get_log_events(
                    logGroupName='/ecs/autogen-task',
                    logStreamName=f'ecs/code-runner/{task_id}'
                )
                output = '\n'.join([event['message'] for event in logs['events']])
            except self.logs_client.exceptions.ResourceNotFoundException:
                output = "No logs found. The task may have failed to start properly."
            
            if not output and exit_code != 0:
                stopped_reason = task_details['containers'][0].get('reason', 'Unknown error')
                output = f"Task failed: {stopped_reason}"
            
            return CodeResult(exit_code=exit_code, output=output)
            
        except WaiterError:
            return CodeResult(exit_code=1, output="Task execution timed out")
        except Exception as e:
            return CodeResult(exit_code=1, output=f"Task execution failed: {str(e)}")

    def restart(self) -> None:
        """Restart the executor. Not applicable for Fargate."""
        pass