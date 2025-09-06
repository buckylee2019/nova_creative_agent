# DPA Agent on AWS AgentCore Runtime

This project demonstrates how to deploy a Dynamic Product Advertising (DPA) Agent built with Strands Agents to AWS AgentCore Runtime. The agent specializes in creating product advertising content using Amazon Nova models.

## 🎯 What This Agent Does

The DPA Agent is a specialized AI assistant for product advertising that can:

### 🎨 Image Generation (Nova Canvas)
- Create professional product advertising images
- Extend product images with new backgrounds (outpainting)
- Edit/modify parts of images (inpainting)
- Optimize prompts for better visual results

### 🎬 Video Creation (Nova Reel)
- Generate product showcase videos (up to 6 seconds)
- Support multiple aspect ratios (16:9, 9:16, 1:1)
- Async video generation with status monitoring

### 📝 Content & Analysis (Nova Pro)
- Generate marketing copy and product descriptions
- Analyze product images for advertising effectiveness
- Provide creative optimization suggestions

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   User Request  │───▶│  AgentCore       │───▶│  DPA Agent      │
│                 │    │  Runtime         │    │  (Strands)      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
                                                         ▼
                                               ┌─────────────────┐
                                               │  MCP Server     │
                                               │  (Nova Tools)   │
                                               └─────────────────┘
                                                         │
                                                         ▼
                                               ┌─────────────────┐
                                               │  Amazon Nova    │
                                               │  Models         │
                                               └─────────────────┘
```

## 📁 Project Structure

```
dpa-agent/
├── agentcore_dpa_agent.py      # AgentCore-compatible agent
├── dpa_agent.py                # Original Strands agent
├── dpa_mcp_server.py           # MCP server with Nova tools
├── agentcore_requirements.txt  # Dependencies for deployment
├── requirements.txt            # Original requirements
├── deploy_script.py            # Automated deployment script
├── test_agentcore_agent.py     # Local testing script
├── deploy_to_agentcore.md      # Detailed deployment guide
├── __init__.py                 # Python package marker
└── README_AGENTCORE.md         # This file
```

## 🚀 Quick Start

### 1. Prerequisites

- AWS Account with credentials configured
- Python 3.10+ installed
- Amazon Bedrock model access enabled for:
  - Anthropic Claude Sonnet 4.0
  - Amazon Nova Canvas, Reel, and Pro

### 2. Configure S3 Storage

Set up an S3 bucket for storing generated images. You can either:

**Option A: Use .env file (Recommended)**
```bash
# Create or edit .env file
echo "DPA_S3_BUCKET=your-dpa-images-bucket" > .env
```

**Option B: Set environment variable**
```bash
export DPA_S3_BUCKET=your-dpa-images-bucket
```

The deployment script will create the bucket if it doesn't exist.

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install bedrock-agentcore-starter-toolkit
pip install -r agentcore_requirements.txt
```

### 4. Test Locally (Optional)

```bash
python test_agentcore_agent.py
```

### 5. Deploy to AWS

**Option A: Automated Deployment**
```bash
python deploy_script.py
```

**Option B: Manual Deployment**
```bash
# Configure
agentcore configure -e agentcore_dpa_agent.py

# Deploy
agentcore launch

# Test
agentcore invoke '{"prompt": "Create a luxury watch advertisement"}'
```

### 6. Use Your Deployed Agent

```python
import boto3
import json
import uuid

client = boto3.client('bedrock-agentcore')

response = client.invoke_agent_runtime(
    agentRuntimeArn="your-agent-arn",
    sessionId=str(uuid.uuid4()),
    inputText=json.dumps({
        "prompt": "Generate a professional product image of running shoes in an urban setting"
    })
)

# Process streaming response
for event in response['completion']:
    if 'chunk' in event:
        chunk = event['chunk']
        if 'bytes' in chunk:
            print(chunk['bytes'].decode('utf-8'), end='')
```

## 🔧 Configuration

### Environment Variables

- `MCP_SERVER_PATH`: Path to MCP server (default: `./dpa_mcp_server.py`)
- `AWS_DEFAULT_REGION`: AWS region for Bedrock models
- `LOG_LEVEL`: Logging level (DEBUG, INFO, WARNING, ERROR)
- `DPA_S3_BUCKET`: S3 bucket name for storing generated images (required for production)

### Model Configuration

The agent uses these models by default:
- **Primary Model**: `us.anthropic.claude-sonnet-4-20250514-v1:0`
- **Image Generation**: Amazon Nova Canvas
- **Video Generation**: Amazon Nova Reel  
- **Text Analysis**: Amazon Nova Pro

## 📊 Monitoring

### CloudWatch Integration
- Automatic logging to CloudWatch Logs
- Structured log format for easy searching
- Error tracking and performance metrics

### Health Monitoring
```bash
# Check agent health
curl https://your-agent-endpoint/health

# Expected response
{"status": "healthy", "agent_ready": true}
```

### AgentCore Observability
Enable in AWS Console for:
- Request tracing
- Performance analytics
- Error analysis
- Usage patterns

## 🎨 Example Use Cases

### Product Image Generation
```json
{
  "prompt": "Create a professional advertising image for a coffee mug in a cozy kitchen setting with warm lighting"
}
```
*Response includes S3 presigned URL for 24-hour access to the generated image.*

### Single Image Upload
```json
{
  "prompt": "Replace the background with a modern office environment while keeping the headphones unchanged",
  "image": "base64_encoded_image_data",
  "image_filename": "headphones.jpg"
}
```

### Multiple Images Upload
```json
{
  "prompt": "Use the first image as the product and the second as background reference. Create a professional advertising image.",
  "images": [
    {
      "data": "base64_encoded_product_image",
      "filename": "product.jpg"
    },
    {
      "data": "base64_encoded_background_image", 
      "filename": "background.jpg"
    }
  ]
}
```

### Virtual Try-On (Multiple Images)
```json
{
  "prompt": "Perform virtual try-on using the person and garment images",
  "images": [
    {
      "data": "base64_encoded_person_image",
      "filename": "person.jpg"
    },
    {
      "data": "base64_encoded_garment_image",
      "filename": "garment.jpg"
    }
  ]
}
```
*Upload multiple images for complex advertising tasks like virtual try-on, background replacement, or image comparison.*

### Marketing Copy
```json
{
  "prompt": "Write compelling marketing copy for a sustainable water bottle targeting environmentally conscious consumers"
}
```

### Video Creation
```json
{
  "prompt": "Create a 6-second product showcase video for wireless earbuds with a modern, tech-focused aesthetic"
}
```
*Videos are saved to your configured S3 bucket and accessible via presigned URLs.*

## 🔒 Security & Permissions

### Required IAM Permissions

The agent needs permissions for:
- Amazon Bedrock model invocation
- S3 access (for image and video storage)
  - `s3:PutObject` - Upload generated images
  - `s3:GetObject` - Generate presigned URLs
  - `s3:CreateBucket` - Create bucket if needed
- CloudWatch logging
- AgentCore runtime operations

### Authentication Options

1. **No Auth** (development/testing)
2. **OAuth Integration** (production)
3. **API Keys** (programmatic access)
4. **AWS IAM** (service-to-service)

## 🚨 Troubleshooting

### Common Issues

1. **Model Access Denied**
   ```
   Solution: Enable model access in Bedrock console
   ```

2. **MCP Server Connection Failed**
   ```
   Solution: Ensure dpa_mcp_server.py is in the project directory
   ```

3. **Deployment Timeout**
   ```
   Solution: Check CloudWatch logs for detailed error messages
   ```

4. **Agent Not Responding**
   ```
   Solution: Check health endpoint and restart if needed
   ```

### Debug Commands

```bash
# Check deployment status
agentcore status

# View recent logs
agentcore logs

# Test connectivity
agentcore invoke '{"prompt": "Hello"}'

# Update deployment
agentcore launch --update
```

## 📈 Performance Optimization

### Response Times
- **Text Generation**: ~2-5 seconds
- **Image Generation**: ~10-30 seconds
- **Video Generation**: ~60-180 seconds (async)

### Scaling Considerations
- AgentCore Runtime auto-scales based on demand
- Consider request batching for high-volume scenarios
- Use async patterns for video generation

### Cost Optimization
- Monitor token usage in CloudWatch
- Use appropriate model sizes for different tasks
- Implement request caching where appropriate

## 🔄 CI/CD Integration

### GitHub Actions Example

```yaml
name: Deploy DPA Agent

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      
      - name: Install dependencies
        run: |
          pip install bedrock-agentcore-starter-toolkit
          pip install -r agentcore_requirements.txt
      
      - name: Configure AWS credentials
        uses: aws-actions/configure-aws-credentials@v2
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: us-west-2
      
      - name: Deploy agent
        run: python deploy_script.py
```

## 📚 Additional Resources

- [AWS AgentCore Documentation](https://docs.aws.amazon.com/bedrock-agentcore/)
- [Strands Agents SDK](https://strandsagents.com/)
- [Amazon Nova Models](https://docs.aws.amazon.com/bedrock/latest/userguide/nova-models.html)
- [Model Context Protocol](https://modelcontextprotocol.io/)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test locally with `python test_agentcore_agent.py`
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**Ready to deploy your DPA Agent to production? Start with `python deploy_script.py`** 🚀