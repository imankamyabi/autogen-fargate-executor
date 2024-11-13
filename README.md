# AutoGen Fargate Executor

A code execution backend for AutoGen that runs code in AWS Fargate containers. This executor allows you to run untrusted code in isolated containers, making it suitable for production environments where code execution needs to be sandboxed.

## Author
**Iman Kamyabi**  
Email: engkamyabi@gmail.com

## Features

- Executes code in isolated AWS Fargate containers
- Automatic creation of required AWS resources (ECS cluster, IAM roles, CloudWatch log groups)
- Support for installing pip dependencies
- Support for custom environment variables
- Support for requirements.txt file
- Integration with AutoGen's code execution interface

## Installation

```bash
pip install autogen-fargate-executor
```

Or using Poetry:

```bash
poetry add autogen-fargate-executor
```

## Prerequisites

1. AWS credentials configured either through environment variables, AWS CLI, or IAM roles
2. VPC with subnets and security groups configured for Fargate tasks
3. Required AWS permissions to create:
   - IAM roles and policies
   - ECS clusters and tasks
   - CloudWatch log groups

## Usage

### Basic Usage with AutoGen

```python
from autogen_fargate_executor import FargateCodeExecutor
from autogen import ConversableAgent

# Initialize executor
executor = FargateCodeExecutor(
    image_uri='python:3.11',
    subnet_ids=['subnet-xxx'],
    security_groups=['sg-xxx'],
)

# Create AutoGen agent with Fargate executor
agent = ConversableAgent(
    name="CodeExecutor",
    llm_config=False,
    code_execution_config={"executor": executor},
    is_termination_msg=lambda msg: "TERMINATE" in msg.get("content", "").strip().upper(),
)
```

### Advanced Configuration

```python
executor = FargateCodeExecutor(
    region_name='us-west-2',             # AWS region
    image_uri='python:3.11',             # Docker image to use
    subnet_ids=['subnet-xxx'],           # List of subnet IDs
    security_groups=['sg-xxx'],          # List of security group IDs
    cluster_name='my-cluster',           # Optional custom cluster name
    cpu='512',                          # CPU units for Fargate task
    memory='1024',                      # Memory (MB) for Fargate task
    timeout_seconds=600,                # Maximum execution time
    requirements_file='requirements.txt', # Optional requirements file
    pip_dependencies=['pandas', 'requests'], # Optional pip packages
    environment_variables={              # Optional environment variables
        'API_KEY': 'xxx',
        'DEBUG': 'true'
    }
)
```

## AWS Resources

The executor will automatically create and manage these AWS resources:

1. ECS Cluster (default name: autogen-executor-cluster)
2. IAM Role (name: ecsTaskExecutionRoleAutoGenFargate)
3. CloudWatch Log Group (name: /ecs/autogen-task)

## Security Considerations

1. Each code execution runs in its own isolated container
2. Containers are terminated after code execution
3. Resources are isolated within your VPC
4. Task execution role follows principle of least privilege
5. All container logs are captured in CloudWatch

## Cost Considerations

This executor uses AWS Fargate, which bills based on:
1. vCPU and memory used per second
2. CloudWatch logs storage and ingestion
3. Data transfer costs

Consider configuring appropriate CPU and memory settings based on your workload.

## Development

### Running Tests

```bash
poetry install
poetry run pytest tests/
```

## License

MIT

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.