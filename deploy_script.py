#!/usr/bin/env python3
"""
Automated deployment script for DPA Agent to AWS AgentCore Runtime.

This script automates the deployment process including:
- Environment validation
- Project structure setup
- AgentCore configuration
- Deployment to AWS
"""

import os
import sys
import subprocess
import json
import boto3
from pathlib import Path
import logging

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not available, use system environment variables

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class DPAAgentDeployer:
    """Automated deployer for DPA Agent to AgentCore Runtime."""
    
    def __init__(self):
        self.project_dir = Path.cwd()
        self.agent_file = "dpa_agent.py"
        self.requirements_file = "requirements.txt"
        self.mcp_server_file = "dpa_mcp_server.py"
        
    def validate_environment(self):
        """Validate the deployment environment."""
        logger.info("🔍 Validating environment...")
        
        # Check Python version
        if sys.version_info < (3, 10):
            raise RuntimeError("Python 3.10+ is required")
        
        # Check AWS credentials
        try:
            session = boto3.Session()
            credentials = session.get_credentials()
            if not credentials:
                raise RuntimeError("AWS credentials not configured")
            logger.info("✅ AWS credentials found")
        except Exception as e:
            raise RuntimeError(f"AWS credentials error: {e}")
        
        # Check required files
        required_files = [self.agent_file, self.requirements_file, self.mcp_server_file]
        for file in required_files:
            if not (self.project_dir / file).exists():
                raise RuntimeError(f"Required file not found: {file}")
        
        # Check S3 bucket configuration
        s3_bucket = os.getenv("DPA_S3_BUCKET")
        if not s3_bucket:
            logger.warning("⚠️  DPA_S3_BUCKET not set - images will be stored locally (not recommended for production)")
            logger.info("Set DPA_S3_BUCKET environment variable for cloud image storage")
        else:
            logger.info(f"✅ S3 bucket configured: {s3_bucket}")
        
        logger.info("✅ Environment validation passed")
    
    def check_model_access(self):
        """Check if required models are accessible in Bedrock."""
        logger.info("🔍 Checking Bedrock model access...")
        
        try:
            bedrock = boto3.client('bedrock')
            
            # List available models
            response = bedrock.list_foundation_models()
            available_models = {model['modelId'] for model in response['modelSummaries']}
            
            # Required models
            required_models = [
                "us.anthropic.claude-sonnet-4-20250514-v1:0",
                "amazon.nova-canvas-v1:0",
                "amazon.nova-reel-v1:0", 
                "amazon.nova-pro-v1:0"
            ]
            
            missing_models = []
            for model in required_models:
                if model not in available_models:
                    missing_models.append(model)
            
            if missing_models:
                logger.warning(f"⚠️  Models may need access enabled: {missing_models}")
                logger.info("Enable model access in the Bedrock console: https://console.aws.amazon.com/bedrock/home#/modelaccess")
            else:
                logger.info("✅ All required models are available")
                
        except Exception as e:
            logger.warning(f"⚠️  Could not verify model access: {e}")
    
    def setup_s3_bucket(self):
        """Set up S3 bucket for image storage if configured."""
        s3_bucket = os.getenv("DPA_S3_BUCKET")
        if not s3_bucket:
            logger.info("⚪ Skipping S3 setup - DPA_S3_BUCKET not configured")
            return
        
        logger.info(f"🪣 Setting up S3 bucket: {s3_bucket}")
        
        try:
            s3_client = boto3.client('s3')
            
            # Check if bucket exists
            try:
                s3_client.head_bucket(Bucket=s3_bucket)
                logger.info(f"✅ S3 bucket {s3_bucket} already exists")
            except s3_client.exceptions.NoSuchBucket:
                # Create bucket
                region = boto3.Session().region_name or 'us-east-1'
                if region == 'us-east-1':
                    s3_client.create_bucket(Bucket=s3_bucket)
                else:
                    s3_client.create_bucket(
                        Bucket=s3_bucket,
                        CreateBucketConfiguration={'LocationConstraint': region}
                    )
                logger.info(f"✅ Created S3 bucket: {s3_bucket}")
            
            # Set up bucket policy for public read access
            bucket_policy = {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Sid": "PublicReadGetObject",
                        "Effect": "Allow",
                        "Principal": "*",
                        "Action": "s3:GetObject",
                        "Resource": f"arn:aws:s3:::{s3_bucket}/*"
                    }
                ]
            }
            
            s3_client.put_bucket_policy(
                Bucket=s3_bucket,
                Policy=json.dumps(bucket_policy)
            )
            
            # Enable CORS
            cors_config = {
                'CORSRules': [
                    {
                        'AllowedHeaders': ['*'],
                        'AllowedMethods': ['GET', 'HEAD'],
                        'AllowedOrigins': ['*'],
                        'MaxAgeSeconds': 3000
                    }
                ]
            }
            
            s3_client.put_bucket_cors(Bucket=s3_bucket, CORSConfiguration=cors_config)
            
            # Setup CloudFront distribution
            self.setup_cloudfront_distribution(s3_bucket)
            
        except Exception as e:
            logger.warning(f"⚠️  S3 bucket setup failed: {e}")
            logger.info("You may need to create the bucket manually or check permissions")
    
    def setup_cloudfront_distribution(self, bucket_name):
        """Set up CloudFront distribution for S3 bucket."""
        logger.info(f"☁️ Setting up CloudFront distribution for {bucket_name}")
        
        try:
            cloudfront = boto3.client('cloudfront')
            region = boto3.Session().region_name or 'us-east-1'
            origin_domain = f"{bucket_name}.s3.{region}.amazonaws.com"
            
            # Check if distribution already exists
            existing_domain = os.getenv("CLOUDFRONT_DOMAIN")
            if existing_domain:
                logger.info(f"✅ CloudFront domain already configured: {existing_domain}")
                return existing_domain
            
            distribution_config = {
                'CallerReference': f"dpa-agent-{int(__import__('time').time())}",
                'Comment': f'DPA Agent Assets Distribution - {bucket_name}',
                'DefaultCacheBehavior': {
                    'TargetOriginId': bucket_name,
                    'ViewerProtocolPolicy': 'redirect-to-https',
                    'MinTTL': 0,
                    'DefaultTTL': 86400,
                    'MaxTTL': 31536000,
                    'ForwardedValues': {
                        'QueryString': False,
                        'Cookies': {'Forward': 'none'}
                    },
                    'TrustedSigners': {
                        'Enabled': False,
                        'Quantity': 0
                    }
                },
                'Origins': {
                    'Quantity': 1,
                    'Items': [
                        {
                            'Id': bucket_name,
                            'DomainName': origin_domain,
                            'S3OriginConfig': {
                                'OriginAccessIdentity': ''
                            }
                        }
                    ]
                },
                'Enabled': True,
                'PriceClass': 'PriceClass_100'
            }
            
            response = cloudfront.create_distribution(DistributionConfig=distribution_config)
            distribution_id = response['Distribution']['Id']
            domain_name = response['Distribution']['DomainName']
            
            logger.info(f"✅ CloudFront distribution created: {distribution_id}")
            logger.info(f"📡 Domain: https://{domain_name}")
            logger.info("⏳ Distribution is deploying (this may take 10-15 minutes)")
            
            # Update environment variables
            os.environ["CLOUDFRONT_DOMAIN"] = domain_name
            os.environ["CLOUDFRONT_DISTRIBUTION_ID"] = distribution_id
            
            return domain_name
            
        except Exception as e:
            logger.warning(f"⚠️  CloudFront setup failed: {e}")
            logger.info("You may need to create the distribution manually")
    
    def setup_project_structure(self):
        """Set up the proper project structure."""
        logger.info("📁 Setting up project structure...")
        
        # Create __init__.py if it doesn't exist
        init_file = self.project_dir / "__init__.py"
        if not init_file.exists():
            init_file.touch()
            logger.info("✅ Created __init__.py")
        
        # Verify all files are in place
        logger.info("✅ Project structure ready")
    
    def install_dependencies(self):
        """Install required dependencies."""
        logger.info("📦 Installing dependencies...")
        
        try:
            # Install AgentCore toolkit and PyYAML for config management
            subprocess.run([
                sys.executable, "-m", "pip", "install", 
                "bedrock-agentcore-starter-toolkit", "PyYAML"
            ], check=True, capture_output=True)
            
            # Install project requirements
            subprocess.run([
                sys.executable, "-m", "pip", "install", 
                "-r", self.requirements_file
            ], check=True, capture_output=True)
            
            logger.info("✅ Dependencies installed")
            
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to install dependencies: {e}")
    
    def configure_agent(self):
        """Configure the agent for deployment."""
        logger.info("⚙️  Configuring agent...")
        
        try:
            # Run agentcore configure with default options
            cmd = ["agentcore", "configure", "-e", self.agent_file]
            
            # Use subprocess with input to provide default answers
            process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Provide default answers (press enter for defaults)
            inputs = "\n" * 10  # Multiple enters for default options
            stdout, stderr = process.communicate(input=inputs)
            
            if process.returncode != 0:
                logger.error(f"Configuration failed: {stderr}")
                raise RuntimeError("Agent configuration failed")
            
            # Now update the configuration file with environment variables
            self._update_config_with_env_vars()
            
            logger.info("✅ Agent configured with environment variables")
            
        except Exception as e:
            raise RuntimeError(f"Failed to configure agent: {e}")
    
    def _update_config_with_env_vars(self):
        """Update the AgentCore configuration file with environment variables."""
        config_file = Path(".bedrock_agentcore.yaml")
        
        if not config_file.exists():
            logger.warning("⚠️  Configuration file not found - environment variables not set")
            return
        
        # Read current configuration
        try:
            import yaml
        except ImportError:
            logger.warning("⚠️  PyYAML not installed - cannot update configuration with environment variables")
            logger.info("Install with: pip install PyYAML")
            return
        
        try:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f) or {}
            
            # Collect environment variables from .env and system
            env_vars = {}
            
            # Add S3 bucket if configured
            s3_bucket = os.getenv("DPA_S3_BUCKET")
            if s3_bucket:
                env_vars["DPA_S3_BUCKET"] = s3_bucket
                logger.info(f"📝 Added DPA_S3_BUCKET: {s3_bucket}")
            
            # Add CloudFront endpoint if configured
            cloudfront_domain = os.getenv("CLOUDFRONT_DOMAIN")
            if cloudfront_domain:
                env_vars["CLOUDFRONT_DOMAIN"] = cloudfront_domain
                logger.info(f"📝 Added CLOUDFRONT_DOMAIN: {cloudfront_domain}")
            
            # Add CloudFront distribution ID if configured
            cloudfront_distribution_id = os.getenv("CLOUDFRONT_DISTRIBUTION_ID")
            if cloudfront_distribution_id:
                env_vars["CLOUDFRONT_DISTRIBUTION_ID"] = cloudfront_distribution_id
                logger.info(f"📝 Added CLOUDFRONT_DISTRIBUTION_ID: {cloudfront_distribution_id}")
            
            # Add AWS region
            aws_region = os.getenv("AWS_DEFAULT_REGION") or boto3.Session().region_name or "us-east-1"
            env_vars["AWS_DEFAULT_REGION"] = aws_region
            logger.info(f"📝 Added AWS_DEFAULT_REGION: {aws_region}")
            
            # Add log level
            log_level = os.getenv("LOG_LEVEL", "INFO")
            env_vars["LOG_LEVEL"] = log_level
            
            # Add MCP server path
            mcp_server_path = os.getenv("MCP_SERVER_PATH", f"./{self.mcp_server_file}")
            env_vars["MCP_SERVER_PATH"] = mcp_server_path
            
            # Add any other environment variables that start with DPA_
            for key, value in os.environ.items():
                if key.startswith("DPA_") and key not in env_vars:
                    env_vars[key] = value
                    logger.info(f"📝 Added {key}: {value}")
            
            # Update configuration with environment variables
            if env_vars:
                # Add environment variables to each agent's AWS configuration
                if 'agents' in config:
                    for agent_name, agent_config in config['agents'].items():
                        if 'aws' in agent_config:
                            agent_config['aws']['environment_variables'] = env_vars
                            logger.info(f"📝 Added environment variables to agent: {agent_name}")
                
                # Also add to top-level bedrock_agentcore for compatibility
                if 'bedrock_agentcore' not in config:
                    config['bedrock_agentcore'] = {}
                
                config['bedrock_agentcore']['environment_variables'] = env_vars
                
                # Write updated configuration
                with open(config_file, 'w') as f:
                    yaml.dump(config, f, default_flow_style=False)
                
                logger.info(f"✅ Updated configuration with {len(env_vars)} environment variables")
            else:
                logger.info("📝 No environment variables to add")
                
        except Exception as e:
            logger.warning(f"⚠️  Could not update configuration with environment variables: {e}")
            logger.info("Environment variables will need to be set manually")
    
    def deploy_agent(self):
        """Deploy the agent to AgentCore Runtime."""
        logger.info("🚀 Deploying agent to AWS...")
        
        try:
            # Run agentcore launch
            result = subprocess.run(
                ["agentcore", "launch"],
                capture_output=True,
                text=True,
                check=True
            )
            
            # Extract agent ARN from output
            output = result.stdout
            agent_arn = None
            
            for line in output.split('\n'):
                if 'arn:aws:bedrock-agentcore' in line:
                    agent_arn = line.strip()
                    break
            
            if agent_arn:
                logger.info(f"✅ Agent deployed successfully!")
                logger.info(f"🎯 Agent ARN: {agent_arn}")
                
                # Save ARN to file for later use
                with open("agent_arn.txt", "w") as f:
                    f.write(agent_arn)
                
                return agent_arn
            else:
                logger.warning("⚠️  Deployment completed but ARN not found in output")
                return None
                
        except subprocess.CalledProcessError as e:
            logger.error(f"Deployment failed: {e.stderr}")
            raise RuntimeError("Agent deployment failed")
    
    def test_deployment(self, agent_arn):
        """Test the deployed agent."""
        if not agent_arn:
            logger.warning("⚠️  Cannot test deployment - no agent ARN available")
            return
        
        logger.info("🧪 Testing deployed agent...")
        
        try:
            # Test with agentcore invoke
            test_payload = '{"prompt": "Hello! Can you help me create product advertising content?"}'
            
            result = subprocess.run(
                ["agentcore", "invoke", test_payload],
                capture_output=True,
                text=True,
                timeout=60  # Increased timeout for first invocation
            )
            
            if result.returncode == 0:
                logger.info("✅ Agent test successful!")
                logger.info(f"Response preview: {result.stdout[:200]}...")
            else:
                logger.warning(f"⚠️  Agent test failed: {result.stderr}")
                logger.info("This might be normal for first deployment - the agent may need time to initialize")
                
        except subprocess.TimeoutExpired:
            logger.warning("⚠️  Agent test timed out - this is normal for first deployment")
        except FileNotFoundError:
            logger.warning("⚠️  agentcore CLI not found - install with: pip install bedrock-agentcore-starter-toolkit")
        except Exception as e:
            logger.warning(f"⚠️  Could not test agent: {e}")
    
    def create_test_script(self, agent_arn):
        """Create test scripts for the deployed agent."""
        if not agent_arn:
            return
        
        logger.info("📝 Creating test scripts...")
        
        # Basic test script
        test_script = f'''#!/usr/bin/env python3
"""
Test script for deployed DPA Agent on AgentCore Runtime.
"""

import json
import uuid
import boto3

# Your deployed agent ARN
AGENT_ARN = "{agent_arn}"

def test_agent():
    """Test the deployed DPA agent."""
    client = boto3.client('bedrock-agentcore')
    
    test_cases = [
        {{"prompt": "Hello! What can you help me create for product advertising?"}},
        {{"prompt": "Generate a professional product image of a luxury watch on marble"}},
        {{"prompt": "Create marketing copy for a sustainable water bottle"}},
    ]
    
    for i, payload in enumerate(test_cases, 1):
        print(f"\\n🧪 Test Case {{i}}: {{payload['prompt'][:50]}}...")
        
        try:
            response = client.invoke_agent_runtime(
                agentRuntimeArn=AGENT_ARN,
                sessionId=str(uuid.uuid4()),
                inputText=json.dumps(payload)
            )
            
            # Process streaming response
            result = ""
            for event in response['completion']:
                if 'chunk' in event:
                    chunk = event['chunk']
                    if 'bytes' in chunk:
                        result += chunk['bytes'].decode('utf-8')
            
            print(f"✅ Response: {{result[:200]}}...")
            
        except Exception as e:
            print(f"❌ Error: {{e}}")

if __name__ == "__main__":
    test_agent()
'''
        
        with open("test_deployed_agent.py", "w") as f:
            f.write(test_script)
        
        # Image upload test script with multiple image support
        image_test_script = f'''#!/usr/bin/env python3
"""
Test script for uploading single and multiple images to the deployed DPA Agent.
"""

import json
import uuid
import boto3
import base64
from pathlib import Path

# Your deployed agent ARN
AGENT_ARN = "{agent_arn}"

def encode_image(image_path: str) -> str:
    """Encode image file to base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def test_single_image(image_path: str, prompt: str):
    """Test uploading a single image."""
    client = boto3.client('bedrock-agentcore')
    
    print(f"📤 Encoding image: {{image_path}}")
    image_base64 = encode_image(image_path)
    image_filename = Path(image_path).name
    
    payload = {{
        "prompt": prompt,
        "image": image_base64,
        "image_filename": image_filename
    }}
    
    print(f"🚀 Single image request: {{image_filename}}")
    return _send_request(client, payload)

def test_multiple_images(image_paths: list, prompt: str):
    """Test uploading multiple images."""
    client = boto3.client('bedrock-agentcore')
    
    images_data = []
    for image_path in image_paths:
        print(f"📤 Encoding image: {{image_path}}")
        image_base64 = encode_image(image_path)
        image_filename = Path(image_path).name
        images_data.append({{"data": image_base64, "filename": image_filename}})
    
    payload = {{
        "prompt": prompt,
        "images": images_data
    }}
    
    print(f"🚀 Multiple images request: {{[img['filename'] for img in images_data]}}")
    return _send_request(client, payload)

def _send_request(client, payload):
    """Send request and process response."""
    try:
        response = client.invoke_agent_runtime(
            agentRuntimeArn=AGENT_ARN,
            sessionId=str(uuid.uuid4()),
            inputText=json.dumps(payload)
        )
        
        result = ""
        for event in response['completion']:
            if 'chunk' in event:
                chunk = event['chunk']
                if 'bytes' in chunk:
                    chunk_text = chunk['bytes'].decode('utf-8')
                    result += chunk_text
                    print(chunk_text, end='', flush=True)
        
        print(f"\\n\\n✅ Request completed!")
        return result
        
    except Exception as e:
        print(f"❌ Error: {{e}}")
        return None

def main():
    """Main test function."""
    
    # Single image tests
    single_tests = [
        {{"image": "test_product.jpg", "prompt": "Analyze and enhance this product image"}},
    ]
    
    # Multiple image tests  
    multiple_tests = [
        {{"images": ["product.jpg", "background.jpg"], "prompt": "Combine product with background"}},
        {{"images": ["person.jpg", "garment.jpg"], "prompt": "Virtual try-on with these images"}},
    ]
    
    print("🧪 Testing Image Upload with DPA Agent")
    print("=" * 50)
    
    # Test single images
    for i, test in enumerate(single_tests, 1):
        if Path(test["image"]).exists():
            print(f"\\n📷 Single Image Test {{i}}:")
            test_single_image(test["image"], test["prompt"])
        else:
            print(f"\\n❌ Single Image Test {{i}}: {{test['image']}} not found")
        print("-" * 50)
    
    # Test multiple images
    for i, test in enumerate(multiple_tests, 1):
        missing = [img for img in test["images"] if not Path(img).exists()]
        if not missing:
            print(f"\\n📷📷 Multiple Image Test {{i}}:")
            test_multiple_images(test["images"], test["prompt"])
        else:
            print(f"\\n❌ Multiple Image Test {{i}}: {{missing}} not found")
        print("-" * 50)

if __name__ == "__main__":
    main()
'''
        
        with open("test_image_upload.py", "w") as f:
            f.write(image_test_script)
        
        logger.info("✅ Test scripts created:")
        logger.info("   • test_deployed_agent.py - Basic agent testing")
        logger.info("   • test_image_upload.py - Image upload testing")
    
    def deploy(self):
        """Run the complete deployment process."""
        try:
            logger.info("🚀 Starting DPA Agent deployment to AgentCore Runtime")
            logger.info("=" * 60)
            
            self.validate_environment()
            self.check_model_access()
            self.setup_s3_bucket()
            self.setup_project_structure()
            self.install_dependencies()
            self.configure_agent()
            agent_arn = self.deploy_agent()
            self.test_deployment(agent_arn)
            self.create_test_script(agent_arn)
            
            logger.info("=" * 60)
            logger.info("🎉 Deployment completed successfully!")
            
            if agent_arn:
                logger.info(f"🎯 Your DPA Agent ARN: {agent_arn}")
                logger.info("📝 Test your agent with:")
                logger.info("   • python test_deployed_agent.py (basic testing)")
                logger.info("   • python test_image_upload.py (image upload testing)")
            
            logger.info("📚 See deploy_to_agentcore.md for detailed usage instructions")
            
        except Exception as e:
            logger.error(f"❌ Deployment failed: {e}")
            sys.exit(1)


def main():
    """Main entry point."""
    deployer = DPAAgentDeployer()
    deployer.deploy()


if __name__ == "__main__":
    main()