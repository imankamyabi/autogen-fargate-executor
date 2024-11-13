import pytest
from unittest.mock import Mock, patch, ANY
from botocore.exceptions import ClientError, WaiterError
from autogen_fargate_executor import FargateCodeExecutor
from autogen.coding import CodeBlock

@pytest.fixture
def code_block():
    """Fixture for creating a valid CodeBlock instance"""
    return CodeBlock(
        code="print('test')",
        language="python",
        display_before_execution=True,
        save_to_file=False,
        save_file_path=None,
        show_code=True
    )

@pytest.fixture
def mock_boto3_clients():
    with patch('boto3.client') as mock_client:
        # Mock IAM client
        mock_iam = Mock()
        # Mock ECS client
        mock_ecs = Mock()
        # Mock CloudWatch Logs client
        mock_logs = Mock()
        
        def mock_client_factory(service_name, region_name=None):
            if service_name == 'iam':
                return mock_iam
            elif service_name == 'ecs':
                return mock_ecs
            elif service_name == 'logs':
                return mock_logs
        
        mock_client.side_effect = mock_client_factory
        yield mock_iam, mock_ecs, mock_logs

@pytest.fixture
def executor_config():
    return {
        'image_uri': 'python:3.11',
        'subnet_ids': ['subnet-1', 'subnet-2'],
        'security_groups': ['sg-1'],
        'region_name': 'us-west-2',
    }

def test_initialize_executor(mock_boto3_clients, executor_config):
    mock_iam, mock_ecs, mock_logs = mock_boto3_clients
    
    # Mock existing role
    mock_iam.get_role.return_value = {
        'Role': {'Arn': 'arn:aws:iam::123456789012:role/ecsTaskExecutionRoleAutoGenFargate'}
    }
    
    # Mock existing cluster
    mock_ecs.describe_clusters.return_value = {'clusters': [{'clusterArn': 'test'}]}
    
    executor = FargateCodeExecutor(**executor_config)
    assert executor.image_uri == 'python:3.11'
    assert executor.subnet_ids == ['subnet-1', 'subnet-2']
    assert executor.security_groups == ['sg-1']

def test_create_role_if_not_exists(mock_boto3_clients, executor_config):
    mock_iam, mock_ecs, mock_logs = mock_boto3_clients
    
    # Mock role doesn't exist
    mock_iam.get_role.side_effect = ClientError(
        {'Error': {'Code': 'NoSuchEntity', 'Message': 'Role not found'}},
        'GetRole'
    )
    
    # Mock role creation
    mock_iam.create_role.return_value = {
        'Role': {'Arn': 'arn:aws:iam::123456789012:role/ecsTaskExecutionRoleAutoGenFargate'}
    }
    
    # Mock existing cluster
    mock_ecs.describe_clusters.return_value = {'clusters': [{'clusterArn': 'test'}]}
    
    executor = FargateCodeExecutor(**executor_config)
    
    # Verify role was created
    mock_iam.create_role.assert_called_once()
    mock_iam.attach_role_policy.assert_called_once()

def test_create_cluster_if_not_exists(mock_boto3_clients, executor_config):
    mock_iam, mock_ecs, mock_logs = mock_boto3_clients
    
    # Mock existing role
    mock_iam.get_role.return_value = {
        'Role': {'Arn': 'arn:aws:iam::123456789012:role/ecsTaskExecutionRoleAutoGenFargate'}
    }
    
    # Mock cluster doesn't exist
    mock_ecs.describe_clusters.return_value = {'clusters': []}
    
    executor = FargateCodeExecutor(**executor_config)
    
    # Verify cluster was created
    mock_ecs.create_cluster.assert_called_once_with(
        clusterName='autogen-executor-cluster',
        capacityProviders=['FARGATE', 'FARGATE_SPOT']
    )

def test_pip_dependencies(mock_boto3_clients, code_block):
    mock_iam, mock_ecs, mock_logs = mock_boto3_clients
    
    # Mock existing role and cluster
    mock_iam.get_role.return_value = {
        'Role': {'Arn': 'arn:aws:iam::123456789012:role/ecsTaskExecutionRoleAutoGenFargate'}
    }
    mock_ecs.describe_clusters.return_value = {'clusters': [{'clusterArn': 'test'}]}
    
    # Mock task definition registration
    mock_ecs.register_task_definition.return_value = {
        'taskDefinition': {'taskDefinitionArn': 'arn:aws:ecs:us-west-2:123456789012:task-definition/test:1'}
    }
    
    # Mock task execution
    mock_ecs.run_task.return_value = {
        'tasks': [{'taskArn': 'arn:aws:ecs:us-west-2:123456789012:task/test'}]
    }
    
    # Mock task description
    mock_ecs.describe_tasks.return_value = {
        'tasks': [{'containers': [{'exitCode': 0}]}]
    }
    
    # Mock logs
    mock_logs.get_log_events.return_value = {
        'events': [{'message': 'test output'}]
    }
    
    config = {
        'image_uri': 'python:3.11',
        'subnet_ids': ['subnet-1'],
        'security_groups': ['sg-1'],
        'pip_dependencies': ['pandas', 'requests'],
        'environment_variables': {'API_KEY': 'test'}
    }
    
    executor = FargateCodeExecutor(**config)
    
    # Execute some code to trigger task definition registration
    result = executor.execute_code_blocks([code_block])
    
    # Verify task definition was registered
    mock_ecs.register_task_definition.assert_called_once()
    call_args = mock_ecs.register_task_definition.call_args[1]
    
    # Verify task definition contents
    container_def = call_args['containerDefinitions'][0]
    command = container_def['command'][2]  # The bash -c command string
    
    # Check for pip dependencies
    assert 'pip install' in command
    assert 'pandas requests' in command
    
    # Check for environment variables
    assert container_def['environment'] == [{'name': 'API_KEY', 'value': 'test'}]

def test_execute_code_blocks(mock_boto3_clients, executor_config):
    mock_iam, mock_ecs, mock_logs = mock_boto3_clients
    
    # Mock existing role and cluster
    mock_iam.get_role.return_value = {
        'Role': {'Arn': 'arn:aws:iam::123456789012:role/ecsTaskExecutionRoleAutoGenFargate'}
    }
    mock_ecs.describe_clusters.return_value = {'clusters': [{'clusterArn': 'test'}]}
    
    # Mock task execution
    mock_ecs.register_task_definition.return_value = {
        'taskDefinition': {'taskDefinitionArn': 'arn:aws:ecs:us-west-2:123456789012:task-definition/test:1'}
    }
    mock_ecs.run_task.return_value = {
        'tasks': [{'taskArn': 'arn:aws:ecs:us-west-2:123456789012:task/test'}]
    }
    mock_ecs.describe_tasks.return_value = {
        'tasks': [{
            'containers': [{'exitCode': 0}]
        }]
    }
    
    # Mock logs
    mock_logs.get_log_events.return_value = {
        'events': [{'message': 'Hello, World!'}]
    }
    
    executor = FargateCodeExecutor(**executor_config)
    code_block = CodeBlock(
        code="print('Hello, World!')",
        language="python",
        display_before_execution=True,
        save_to_file=False,
        save_file_path=None,
        show_code=True
    )
    
    result = executor.execute_code_blocks([code_block])
    
    assert result.exit_code == 0
    assert "Hello, World!" in result.output

def test_execution_error(mock_boto3_clients, executor_config):
    mock_iam, mock_ecs, mock_logs = mock_boto3_clients
    
    # Mock existing role and cluster
    mock_iam.get_role.return_value = {
        'Role': {'Arn': 'arn:aws:iam::123456789012:role/ecsTaskExecutionRoleAutoGenFargate'}
    }
    mock_ecs.describe_clusters.return_value = {'clusters': [{'clusterArn': 'test'}]}
    
    # Mock task execution
    mock_ecs.register_task_definition.return_value = {
        'taskDefinition': {'taskDefinitionArn': 'arn:aws:ecs:us-west-2:123456789012:task-definition/test:1'}
    }
    mock_ecs.run_task.return_value = {
        'tasks': [{'taskArn': 'arn:aws:ecs:us-west-2:123456789012:task/test'}]
    }
    
    # Mock task failure
    mock_ecs.describe_tasks.return_value = {
        'tasks': [{
            'containers': [{'exitCode': 1, 'reason': 'Error occurred'}]
        }]
    }
    
    # Mock logs
    mock_logs.get_log_events.return_value = {
        'events': [{'message': 'Error: something went wrong'}]
    }
    
    executor = FargateCodeExecutor(**executor_config)
    code_block = CodeBlock(
        code="invalid python code",
        language="python",
        display_before_execution=True,
        save_to_file=False,
        save_file_path=None,
        show_code=True
    )
    
    result = executor.execute_code_blocks([code_block])
    
    assert result.exit_code == 1
    assert "Error: something went wrong" in result.output