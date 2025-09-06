#!/usr/bin/env python3
"""
DPA MCP Server - Model Context Protocol server for Dynamic Product Advertising
Provides tools for Amazon Nova Canvas, Nova Reel, and Nova Pro models
"""

import boto3
import json
import base64
import time
import random
import os
from PIL import Image
from fastmcp import FastMCP
from typing import Annotated, List, Dict, Any, Optional
import dpa_config
from pydantic import Field
import logging
from botocore.config import Config

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not available, use system environment variables

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("DPAServer")

mcp = FastMCP("DPA Server")

class DPAOperator:
    def __init__(self):
        # Initialize with extended timeout for Nova models
        config = Config(
            connect_timeout=3600,  # 60 minutes
            read_timeout=3600,     # 60 minutes
            retries={'max_attempts': 1}
        )
        self.session = boto3.Session()
        self.bedrock_runtime = self.session.client(
            'bedrock-runtime', 
            region_name='us-east-1',
            config=config
        )
        
        # Debug environment variables
        s3_bucket = dpa_config.DPA_S3_BUCKET
        cloudfront_endpoint = dpa_config.CLOUDFRONT_ENDPOINT
        aws_region = os.getenv("AWS_DEFAULT_REGION")
        log_level = os.getenv("LOG_LEVEL")
        
        logger.info("DPA Operator initialized with Bedrock Runtime client")
        logger.info(f"Environment Variables Debug:")
        logger.info(f"  DPA_S3_BUCKET: {s3_bucket}")
        logger.info(f"  CLOUDFRONT_ENDPOINT: {cloudfront_endpoint}")
        logger.info(f"  AWS_DEFAULT_REGION: {aws_region}")
        logger.info(f"  LOG_LEVEL: {log_level}")
        
        # Show all environment variables starting with DPA_ or CLOUDFRONT_
        logger.info("All relevant environment variables:")
        for key, value in os.environ.items():
            if key.startswith(('DPA_', 'CLOUDFRONT_', 'AWS_')):
                logger.info(f"  {key}: {value}")
dpa_operator = DPAOperator()

@mcp.tool()
def generate_image_nova_canvas(
    prompt: Annotated[str, Field(description="Text description of the image to generate")],
    negative_prompt: Annotated[str, Field(description="What to avoid in the image")] = "",
    width: Annotated[int, Field(description="Image width (320-2048, must be multiple of 64)")] = 1024,
    height: Annotated[int, Field(description="Image height (320-2048, must be multiple of 64)")] = 1024,
    quality: Annotated[str, Field(description="Image quality: standard or premium")] = "standard",
    cfg_scale: Annotated[float, Field(description="CFG scale (1.0-10.0)")] = 7.0,
    seed: Annotated[int, Field(description="Random seed (0-858993459), -1 for random")] = -1
) -> Dict[str, Any]:
    """Generate an image using Amazon Nova Canvas"""
    
    try:
        # Generate random seed if not provided
        if seed == -1:
            seed = random.randint(0, 858993459)
        
        # Validate dimensions (must be multiple of 64)
        width = ((width + 31) // 64) * 64
        height = ((height + 31) // 64) * 64
        width = max(320, min(2048, width))
        height = max(320, min(2048, height))
        
        # Build request payload
        request_payload = {
            "taskType": "TEXT_IMAGE",
            "textToImageParams": {
                "text": prompt
            },
            "imageGenerationConfig": {
                "seed": seed,
                "quality": quality,
                "width": width,
                "height": height,
                "numberOfImages": 1,
                "cfgScale": cfg_scale
            }
        }
        
        # Add negative prompt if provided
        if negative_prompt:
            request_payload["textToImageParams"]["negativeText"] = negative_prompt
        
        logger.info(f"Generating image with Nova Canvas: {prompt[:50]}...")
        
        # Invoke Nova Canvas
        response = dpa_operator.bedrock_runtime.invoke_model(
            modelId="amazon.nova-canvas-v1:0",
            body=json.dumps(request_payload),
            contentType="application/json"
        )
        
        # Parse response
        response_body = json.loads(response['body'].read())
        
        if 'images' in response_body and response_body['images']:
            base64_image = response_body['images'][0]
            
            # Save image to file and return path
            import time
            filename = f"nova_canvas_{int(time.time())}"
            save_result = _save_base64_image(base64_image, filename)
            
            if save_result["success"]:
                return {
                    "success": True,
                    "output_path": save_result["output_path"],
                    "file_size": save_result["file_size"],
                    "prompt": prompt,
                    "negative_prompt": negative_prompt,
                    "dimensions": f"{width}x{height}",
                    "quality": quality,
                    "cfg_scale": cfg_scale,
                    "seed": seed,
                    "model": "amazon.nova-canvas-v1:0"
                }
            else:
                return save_result
        else:
            return {"success": False, "error": "No image generated"}
            
    except Exception as e:
        logger.error(f"Error generating image: {str(e)}")
        return {"success": False, "error": str(e)}

# @mcp.tool()
# def generate_video_nova_reel(
#     prompt: Annotated[str, Field(description="Text description of the video to generate")],
#     s3_output_bucket: Annotated[str, Field(description="S3 bucket URI for output (s3://bucket-name)")],
#     duration_seconds: Annotated[int, Field(description="Video duration in seconds (6 max)")] = 6,
#     fps: Annotated[int, Field(description="Frames per second (24 recommended)")] = 24,
#     dimension: Annotated[str, Field(description="Video dimensions: 1280x720, 720x1280, 1024x1024")] = "1280x720",
#     seed: Annotated[int, Field(description="Random seed (0-2147483646), -1 for random")] = -1
# ) -> Dict[str, Any]:
#     """Generate a video using Amazon Nova Reel (asynchronous)"""
    
#     try:
#         # Generate random seed if not provided
#         if seed == -1:
#             seed = random.randint(0, 2147483646)
        
#         # Build request payload
#         model_input = {
#             "taskType": "TEXT_VIDEO",
#             "textToVideoParams": {"text": prompt},
#             "videoGenerationConfig": {
#                 "fps": fps,
#                 "durationSeconds": min(duration_seconds, 6),  # Max 6 seconds
#                 "dimension": dimension,
#                 "seed": seed,
#             },
#         }
        
#         # Configure S3 output
#         output_config = {"s3OutputDataConfig": {"s3Uri": s3_output_bucket}}
        
#         logger.info(f"Starting video generation with Nova Reel: {prompt[:50]}...")
        
#         # Start async video generation
#         response = dpa_operator.bedrock_runtime.start_async_invoke(
#             modelId="amazon.nova-reel-v1:0",
#             modelInput=model_input,
#             outputDataConfig=output_config
#         )
        
#         invocation_arn = response["invocationArn"]
        
#         return {
#             "success": True,
#             "invocation_arn": invocation_arn,
#             "prompt": prompt,
#             "duration_seconds": duration_seconds,
#             "fps": fps,
#             "dimension": dimension,
#             "seed": seed,
#             "s3_output_bucket": s3_output_bucket,
#             "model": "amazon.nova-reel-v1:0",
#             "status": "SUBMITTED"
#         }
        
#     except Exception as e:
#         logger.error(f"Error starting video generation: {str(e)}")
#         return {"success": False, "error": str(e)}

# @mcp.tool()
# def check_video_generation_status(
#     invocation_arn: Annotated[str, Field(description="ARN of the async video generation job")]
# ) -> Dict[str, Any]:
#     """Check the status of a Nova Reel video generation job"""
    
#     try:
#         response = dpa_operator.bedrock_runtime.get_async_invoke(
#             invocationArn=invocation_arn
#         )
        
#         status = response["status"]
        
#         result = {
#             "success": True,
#             "invocation_arn": invocation_arn,
#             "status": status,
#             "model": response.get("modelId", "amazon.nova-reel-v1:0")
#         }
        
#         if status == "Completed":
#             output_s3_uri = response["outputDataConfig"]["s3OutputDataConfig"]["s3Uri"]
#             result["video_s3_uri"] = f"{output_s3_uri}/output.mp4"
#             result["message"] = "Video generation completed successfully"
#         elif status == "Failed":
#             result["error"] = response.get("failureMessage", "Video generation failed")
#         elif status == "InProgress":
#             result["message"] = "Video generation in progress"
        
#         return result
        
#     except Exception as e:
#         logger.error(f"Error checking video status: {str(e)}")
#         return {"success": False, "error": str(e)}

@mcp.tool()
def generate_text_nova_pro(
    prompt: Annotated[str, Field(description="Text prompt for Nova Pro")],
    system_prompt: Annotated[str, Field(description="System instructions")] = "",
    max_tokens: Annotated[int, Field(description="Maximum tokens to generate")] = 500,
    temperature: Annotated[float, Field(description="Temperature (0.0-1.0)")] = 0.7,
    top_p: Annotated[float, Field(description="Top-p sampling (0.0-1.0)")] = 0.9,
    top_k: Annotated[int, Field(description="Top-k sampling")] = 20
) -> Dict[str, Any]:
    """Generate text using Amazon Nova Pro"""
    
    try:
        # Build message list
        message_list = [{"role": "user", "content": [{"text": prompt}]}]
        
        # Build system list if provided
        system_list = []
        if system_prompt:
            system_list = [{"text": system_prompt}]
        
        # Configure inference parameters
        inf_params = {
            "maxTokens": max_tokens,
            "topP": top_p,
            "topK": top_k,
            "temperature": temperature
        }
        
        # Build request body
        request_body = {
            "schemaVersion": "messages-v1",
            "messages": message_list,
            "inferenceConfig": inf_params,
        }
        
        if system_list:
            request_body["system"] = system_list
        
        logger.info(f"Generating text with Nova Pro: {prompt[:50]}...")
        
        # Invoke Nova Pro
        response = dpa_operator.bedrock_runtime.invoke_model(
            modelId="us.amazon.nova-pro-v1:0",
            body=json.dumps(request_body)
        )
        
        # Parse response
        response_body = json.loads(response['body'].read())
        
        if 'output' in response_body and 'message' in response_body['output']:
            generated_text = response_body['output']['message']['content'][0]['text']
            
            return {
                "success": True,
                "generated_text": generated_text,
                "prompt": prompt,
                "system_prompt": system_prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "model": "us.amazon.nova-pro-v1:0",
                "usage": response_body.get('usage', {})
            }
        else:
            return {"success": False, "error": "No text generated"}
            
    except Exception as e:
        logger.error(f"Error generating text: {str(e)}")
        return {"success": False, "error": str(e)}

@mcp.tool()
def optimize_dpa_prompt(
    product_name: Annotated[str, Field(description="Product name")],
    product_category: Annotated[str, Field(description="Product category")],
    scene_description: Annotated[str, Field(description="Desired scene description")],
    style: Annotated[str, Field(description="Visual style preference")] = "professional photography"
) -> Dict[str, Any]:
    """Optimize a prompt for DPA image generation using Nova Pro"""
    
    system_prompt = """You are an expert at creating optimized prompts for Amazon Nova Canvas image generation for Dynamic Product Advertising (DPA). 

Your task is to create a detailed, effective prompt that will generate high-quality product advertising images.

Guidelines:
- Include specific visual details about lighting, composition, and style
- Mention the product naturally within the scene
- Use professional photography terminology
- Keep prompts concise but descriptive
- Focus on commercial/advertising quality output"""
    
    user_prompt = f"""Create an optimized image generation prompt for:

Product: {product_name}
Category: {product_category}
Scene: {scene_description}
Style: {style}

Return only the optimized prompt in English, no explanations."""
    
    result = generate_text_nova_pro(
        prompt=user_prompt,
        system_prompt=system_prompt,
        max_tokens=200,
        temperature=0.3
    )
    
    if result["success"]:
        optimized_prompt = result["generated_text"].strip()
        return {
            "success": True,
            "original_inputs": {
                "product_name": product_name,
                "product_category": product_category,
                "scene_description": scene_description,
                "style": style
            },
            "optimized_prompt": optimized_prompt
        }
    else:
        return result

@mcp.tool()
def save_uploaded_image(
    image_base64: Annotated[str, Field(description="Base64 encoded image data")],
    filename: Annotated[str, Field(description="Filename for the uploaded image")] = "uploaded_image"
) -> Dict[str, Any]:
    """Save an uploaded base64 image to temporary storage for processing"""
    
    try:
        import uuid
        from datetime import datetime
        
        # Generate unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        
        # Remove extension from filename if present, we'll add .png
        if '.' in filename:
            filename = filename.rsplit('.', 1)[0]
        
        full_filename = f"{timestamp}_{unique_id}_{filename}"
        
        # Save using existing helper function
        save_result = _save_base64_image(image_base64, full_filename)
        
        if save_result["success"]:
            # Also save to local temp for immediate processing
            import base64
            decoded_image = base64.b64decode(image_base64)
            
            temp_dir = "/tmp/uploaded_images"
            os.makedirs(temp_dir, exist_ok=True)
            
            local_path = os.path.join(temp_dir, f"{full_filename}.png")
            with open(local_path, "wb") as f:
                f.write(decoded_image)
            
            logger.info(f"Image saved to both S3 and local temp: {local_path}")
            
            return {
                "success": True,
                "local_path": local_path,
                "s3_url": save_result.get("output_path"),
                "file_size": save_result.get("file_size"),
                "storage_type": save_result.get("storage_type", "unknown")
            }
        else:
            return save_result
            
    except Exception as e:
        logger.error(f"Error saving uploaded image: {str(e)}")
        return {"success": False, "error": str(e)}

@mcp.tool()
def understand_image_nova_pro(
    image_path: Annotated[str, Field(description="Path to the image file")],
    prompt: Annotated[str, Field(description="Question or instruction about the image")],
    system_prompt: Annotated[str, Field(description="System instructions")] = "You are an expert at analyzing product images for advertising purposes.",
    max_tokens: Annotated[int, Field(description="Maximum tokens to generate")] = 1000
) -> Dict[str, Any]:
    """Understand and analyze images using Amazon Nova Pro vision capabilities"""
    
    try:
        # Check if file exists first
        if not os.path.exists(image_path):
            logger.error(f"Image file not found: {image_path}")
            return {"success": False, "error": f"Image file not found: {image_path}"}
        
        logger.info(f"Reading image file: {image_path}")
        
        # Read and encode image
        with open(image_path, "rb") as image_file:
            image_data = image_file.read()
            image_base64 = base64.b64encode(image_data).decode('utf-8')
            
        logger.info(f"Successfully encoded image, size: {len(image_data)} bytes")
        
        # Build message with image and text
        message_list = [{
            "role": "user", 
            "content": [
                {
                    "image": {
                        "format": "png",
                        "source": {"bytes": image_base64}
                    }
                },
                {"text": prompt}
            ]
        }]
        
        # Build system list
        system_list = [{"text": system_prompt}]
        
        # Configure inference parameters
        inf_params = {
            "maxTokens": max_tokens,
            "topP": 0.9,
            "temperature": 0.3
        }
        
        # Build request body
        request_body = {
            "schemaVersion": "messages-v1",
            "messages": message_list,
            "system": system_list,
            "inferenceConfig": inf_params,
        }
        
        logger.info(f"Analyzing image {image_path} with Nova Pro: {prompt[:50]}...")
        
        # Invoke Nova Pro with vision
        response = dpa_operator.bedrock_runtime.invoke_model(
            modelId="us.amazon.nova-pro-v1:0",
            body=json.dumps(request_body)
        )
        
        # Parse response
        response_body = json.loads(response['body'].read())
        
        if 'output' in response_body and 'message' in response_body['output']:
            analysis = response_body['output']['message']['content'][0]['text']
            
            return {
                "success": True,
                "analysis": analysis,
                "image_path": image_path,
                "prompt": prompt,
                "system_prompt": system_prompt,
                "model": "us.amazon.nova-pro-v1:0",
                "usage": response_body.get('usage', {})
            }
        else:
            return {"success": False, "error": "No analysis generated"}
            
    except FileNotFoundError:
        return {"success": False, "error": f"Image file not found: {image_path}"}
    except Exception as e:
        logger.error(f"Error analyzing image: {str(e)}")
        return {"success": False, "error": str(e)}

# @mcp.tool()
# def outpaint_image_nova_canvas(
#     image_path: Annotated[str, Field(description="Path to the source image file")],
#     mask_image_path: Annotated[str, Field(description="Path to the mask image file (white areas will be outpainted)")],
#     prompt: Annotated[str, Field(description="Description of how to extend/modify the image")],
#     negative_prompt: Annotated[str, Field(description="What to avoid in the outpainting")] = "",
#     cfg_scale: Annotated[float, Field(description="CFG scale (1.0-10.0)")] = 7.0,
#     seed: Annotated[int, Field(description="Random seed (0-858993459), -1 for random")] = -1
# ) -> Dict[str, Any]:
#     """Outpaint/extend an image using Amazon Nova Canvas with mask image"""
    
#     try:
#         # Read and encode source image
#         with open(image_path, "rb") as image_file:
#             image_data = image_file.read()
#             image_base64 = base64.b64encode(image_data).decode('utf-8')
        
#         # Read and encode mask image
#         with open(mask_image_path, "rb") as mask_file:
#             mask_data = mask_file.read()
#             mask_base64 = base64.b64encode(mask_data).decode('utf-8')
        
#         # Generate random seed if not provided
#         if seed == -1:
#             seed = random.randint(0, 858993459)
        
#         # Build request payload for inpainting
#         request_payload = {
#             "taskType": "INPAINTING",
#             "inPaintingParams": {
#                 "text": prompt,
#                 "image": image_base64,
#                 "maskImage": mask_base64
#             },
#             "imageGenerationConfig": {
#                 "seed": seed,
#                 "quality": "standard",
#                 "numberOfImages": 1,
#                 "cfgScale": cfg_scale
#             }
#         }
        
#         # Add negative prompt if provided
#         if negative_prompt:
#             request_payload["inPaintingParams"]["negativeText"] = negative_prompt
        
#         logger.info(f"Outpainting image {image_path} with mask {mask_image_path}: {prompt[:50]}...")
        
#         # Invoke Nova Canvas for outpainting
#         response = dpa_operator.bedrock_runtime.invoke_model(
#             modelId="amazon.nova-canvas-v1:0",
#             body=json.dumps(request_payload),
#             contentType="application/json"
#         )
        
#         # Parse response
#         response_body = json.loads(response['body'].read())
        
#         if 'images' in response_body and response_body['images']:
#             outpainted_image = response_body['images'][0]
            
#             return {
#                 "success": True,
#                 "outpainted_image_base64": outpainted_image,
#                 "source_image_path": image_path,
#                 "mask_image_path": mask_image_path,
#                 "prompt": prompt,
#                 "negative_prompt": negative_prompt,
#                 "cfg_scale": cfg_scale,
#                 "seed": seed,
#                 "model": "amazon.nova-canvas-v1:0"
#             }
#         else:
#             return {"success": False, "error": "No outpainted image generated"}
            
#     except FileNotFoundError as e:
#         return {"success": False, "error": f"Image file not found: {str(e)}"}
#     except Exception as e:
#         logger.error(f"Error outpainting image: {str(e)}")
#         return {"success": False, "error": str(e)}

@mcp.tool()
def generate_image_nova_canvas(
    prompt: Annotated[str, Field(description="Text description of the image to generate")],
    negative_prompt: Annotated[str, Field(description="What to avoid in the image")] = "",
    width: Annotated[int, Field(description="Image width (320-2048, must be multiple of 64)")] = 1024,
    height: Annotated[int, Field(description="Image height (320-2048, must be multiple of 64)")] = 1024,
    quality: Annotated[str, Field(description="Image quality: standard or premium")] = "standard",
    cfg_scale: Annotated[float, Field(description="CFG scale (1.0-10.0)")] = 7.0,
    seed: Annotated[int, Field(description="Random seed (0-858993459), -1 for random")] = -1,
    output_filename: Annotated[str, Field(description="Output filename (without extension)")] = "generated_image"
) -> Dict[str, Any]:
    """Generate an image using Amazon Nova Canvas"""
    
    try:
        # Generate random seed if not provided
        if seed == -1:
            seed = random.randint(0, 858993459)
        
        # Validate dimensions (must be multiple of 64)
        width = ((width + 31) // 64) * 64
        height = ((height + 31) // 64) * 64
        width = max(320, min(2048, width))
        height = max(320, min(2048, height))
        
        # Build request payload
        request_payload = {
            "taskType": "TEXT_IMAGE",
            "textToImageParams": {
                "text": prompt
            },
            "imageGenerationConfig": {
                "seed": seed,
                "quality": quality,
                "width": width,
                "height": height,
                "numberOfImages": 1,
                "cfgScale": cfg_scale
            }
        }
        
        # Add negative prompt if provided
        if negative_prompt:
            request_payload["textToImageParams"]["negativeText"] = negative_prompt
        
        logger.info(f"Generating image with Nova Canvas: {prompt[:50]}...")
        
        # Invoke Nova Canvas
        response = dpa_operator.bedrock_runtime.invoke_model(
            modelId="amazon.nova-canvas-v1:0",
            body=json.dumps(request_payload),
            contentType="application/json"
        )
        
        # Parse response
        response_body = json.loads(response['body'].read())
        
        if 'images' in response_body and response_body['images']:
            base64_image = response_body['images'][0]
            
            # Save image to file
            save_result = _save_base64_image(base64_image, output_filename)
            if not save_result["success"]:
                return save_result
            
            return {
                "success": True,
                "output_path": save_result["output_path"],
                "local_path": save_result["local_path"],
                "file_size": save_result["file_size"],
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "dimensions": f"{width}x{height}",
                "quality": quality,
                "cfg_scale": cfg_scale,
                "seed": seed,
                "model": "amazon.nova-canvas-v1:0"
            }
        else:
            return {"success": False, "error": "No image generated"}
            
    except Exception as e:
        logger.error(f"Error generating image: {str(e)}")
        return {"success": False, "error": str(e)}

@mcp.tool()
def outpaint_image_nova_canvas(
    image_path: Annotated[str, Field(description="Path to the source image file")],
    mask_image_path: Annotated[str, Field(description="Path to the mask image file (white areas will be outpainted)")],
    prompt: Annotated[str, Field(description="Description of how to extend/modify the image")],
    negative_prompt: Annotated[str, Field(description="What to avoid in the outpainting")] = "",
    cfg_scale: Annotated[float, Field(description="CFG scale (1.0-10.0)")] = 7.0,
    seed: Annotated[int, Field(description="Random seed (0-858993459), -1 for random")] = -1,
    output_filename: Annotated[str, Field(description="Output filename (without extension)")] = "outpainted_image"
) -> Dict[str, Any]:
    """Outpaint/extend an image using Amazon Nova Canvas with mask image"""
    
    try:
        # Read and encode source image
        with open(image_path, "rb") as image_file:
            image_data = image_file.read()
            image_base64 = base64.b64encode(image_data).decode('utf-8')
        
        # Read and encode mask image
        with open(mask_image_path, "rb") as mask_file:
            mask_data = mask_file.read()
            mask_base64 = base64.b64encode(mask_data).decode('utf-8')
        
        # Generate random seed if not provided
        if seed == -1:
            seed = random.randint(0, 858993459)
        
        # Build request payload for inpainting
        request_payload = {
            "taskType": "INPAINTING",
            "inPaintingParams": {
                "text": prompt,
                "image": image_base64,
                "maskImage": mask_base64
            },
            "imageGenerationConfig": {
                "seed": seed,
                "quality": "standard",
                "numberOfImages": 1,
                "cfgScale": cfg_scale
            }
        }
        
        # Add negative prompt if provided
        if negative_prompt:
            request_payload["inPaintingParams"]["negativeText"] = negative_prompt
        
        logger.info(f"Outpainting image {image_path} with mask {mask_image_path}: {prompt[:50]}...")
        
        # Invoke Nova Canvas for outpainting
        response = dpa_operator.bedrock_runtime.invoke_model(
            modelId="amazon.nova-canvas-v1:0",
            body=json.dumps(request_payload),
            contentType="application/json"
        )
        
        # Parse response
        response_body = json.loads(response['body'].read())
        
        if 'images' in response_body and response_body['images']:
            outpainted_image = response_body['images'][0]
            
            # Save image to file
            save_result = _save_base64_image(outpainted_image, output_filename)
            if not save_result["success"]:
                return save_result
            
            return {
                "success": True,
                "output_path": save_result["output_path"],
                "local_path": save_result["local_path"],
                "file_size": save_result["file_size"],
                "source_image_path": image_path,
                "mask_image_path": mask_image_path,
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "cfg_scale": cfg_scale,
                "seed": seed,
                "model": "amazon.nova-canvas-v1:0"
            }
        else:
            return {"success": False, "error": "No outpainted image generated"}
            
    except FileNotFoundError as e:
        return {"success": False, "error": f"Image file not found: {str(e)}"}
    except Exception as e:
        logger.error(f"Error outpainting image: {str(e)}")
        return {"success": False, "error": str(e)}

@mcp.tool()
def inpaint_image_nova_canvas(
    image_path: Annotated[str, Field(description="Path to the source image file")],
    prompt: Annotated[str, Field(description="Text prompt describing the whole image after editing")] = "",
    mask_prompt: Annotated[str, Field(description="Describe area to change (e.g., 'the dog', 'flowers')")] = "",
    mask_image_path: Annotated[str, Field(description="Path to mask image (white=modify, black=preserve)")] = "",
    negative_prompt: Annotated[str, Field(description="What to avoid in the result")] = "",
    quality: Annotated[str, Field(description="standard or premium")] = "standard",
    cfg_scale: Annotated[float, Field(description="CFG scale (1.0-10.0)", ge=1.0, le=10.0)] = 5.0,
    seed: Annotated[int, Field(description="Random seed (-1 for random)")] = -1,
    output_filename: Annotated[str, Field(description="Output filename prefix")] = "inpainted_image"
) -> Dict[str, Any]:
    """Inpaint image to add, remove, or replace elements using Amazon Nova Canvas"""
    
    try:
        # Validate inputs
        if not mask_prompt and not mask_image_path:
            return {"success": False, "error": "Either mask_prompt or mask_image_path is required"}
        
        # Read and encode source image
        with open(image_path, "rb") as image_file:
            image_data = image_file.read()
            image_base64 = base64.b64encode(image_data).decode('utf-8')
        
        # Generate random seed if not provided
        if seed == -1:
            seed = random.randint(0, 2147483647)
        
        # Build request payload
        request_payload = {
            "taskType": "INPAINTING",
            "inPaintingParams": {
                "image": image_base64
            },
            "imageGenerationConfig": {
                "quality": quality,
                "cfgScale": cfg_scale,
                "seed": seed
            }
        }
        
        # Add text prompt if provided (omit for removal)
        if prompt:
            request_payload["inPaintingParams"]["text"] = prompt
        
        # Add negative prompt if provided
        if negative_prompt:
            request_payload["inPaintingParams"]["negativeText"] = negative_prompt
        
        # Add mask (either prompt or image)
        if mask_image_path:
            with open(mask_image_path, "rb") as mask_file:
                mask_data = mask_file.read()
                mask_base64 = base64.b64encode(mask_data).decode('utf-8')
            request_payload["inPaintingParams"]["maskImage"] = mask_base64
        elif mask_prompt:
            request_payload["inPaintingParams"]["maskPrompt"] = mask_prompt
        
        logger.info(f"Inpainting image with {'mask image' if mask_image_path else 'mask prompt'}...")
        
        # Invoke Nova Canvas
        response = dpa_operator.bedrock_runtime.invoke_model(
            modelId="amazon.nova-canvas-v1:0",
            body=json.dumps(request_payload)
        )
        
        # Parse response
        response_body = json.loads(response['body'].read())
        
        if 'images' in response_body and response_body['images']:
            inpainted_image = response_body['images'][0]
            
            # Save image to file
            save_result = _save_base64_image(inpainted_image, output_filename)
            if not save_result["success"]:
                return save_result
            
            return {
                "success": True,
                "output_path": save_result["output_path"],
                "local_path": save_result["local_path"],
                "file_size": save_result["file_size"],
                "source_image_path": image_path,
                "mask_image_path": mask_image_path if mask_image_path else None,
                "mask_prompt": mask_prompt if mask_prompt else None,
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "cfg_scale": cfg_scale,
                "seed": seed,
                "model": "amazon.nova-canvas-v1:0",
                "task_type": "INPAINTING"
            }
        else:
            return {"success": False, "error": "No inpainted image generated"}
            
    except FileNotFoundError as e:
        return {"success": False, "error": f"Image file not found: {str(e)}"}
    except Exception as e:
        logger.error(f"Error inpainting image: {str(e)}")
        return {"success": False, "error": str(e)}

@mcp.tool()
def outpaint_with_object_mask(
    image_path: Annotated[str, Field(description="Path to the source image file")],
    object_category: Annotated[str, Field(description="Object category to preserve (e.g., 'person', 'car', 'product')")],
    outpaint_prompt: Annotated[str, Field(description="Description of what to generate in the background")],
    mask_type: Annotated[str, Field(description="Mask type: 'object' (preserve object, change background) or 'background' (change object, preserve background)")] = "object",
    negative_prompt: Annotated[str, Field(description="What to avoid in the outpainting")] = "",
    cfg_scale: Annotated[float, Field(description="CFG scale (1.0-10.0)")] = 7.0,
    seed: Annotated[int, Field(description="Random seed (0-858993459), -1 for random")] = -1,
    output_filename: Annotated[str, Field(description="Output filename (without extension)")] = "outpainted_image"
) -> Dict[str, Any]:
    """Outpaint image by automatically detecting object and generating background"""
    
    try:
        # First, generate mask for the object
        mask_result = generate_mask_for_object(
            image_path=image_path,
            object_category=object_category,
            mask_type=mask_type,  # Black object, white background for outpainting
            output_filename=f"{output_filename}_mask"
        )
        
        if not mask_result["success"]:
            return {"success": False, "error": f"Failed to generate mask: {mask_result['error']}"}
        
        mask_path = mask_result["mask_path"]
        
        # Now use the generated mask for outpainting
        outpaint_result = outpaint_image_nova_canvas(
            image_path=image_path,
            mask_image_path=mask_path,
            prompt=outpaint_prompt,
            negative_prompt=negative_prompt,
            cfg_scale=cfg_scale,
            seed=seed,
            output_filename=output_filename
        )
        
        if outpaint_result["success"]:
            # Add mask generation info to result
            outpaint_result["mask_generation"] = {
                "object_category": object_category,
                "objects_detected": mask_result["objects_detected"],
                "bounding_boxes": mask_result["bounding_boxes"],
                "mask_path": mask_path
            }
        
        return outpaint_result
        
    except Exception as e:
        logger.error(f"Error in outpaint with object mask: {str(e)}")
        return {"success": False, "error": str(e)}

@mcp.tool()
def virtual_try_on_nova_canvas(
    source_image_path: Annotated[str, Field(description="Path to source image (person/room scene)")],
    reference_image_path: Annotated[str, Field(description="Path to reference image (garment/furniture/object)")],
    mask_type: Annotated[str, Field(description="Mask type: GARMENT, PROMPT, or IMAGE")] = "GARMENT",
    garment_class: Annotated[str, Field(description="For GARMENT mask: UPPER_BODY, LOWER_BODY, FOOTWEAR, FULL_BODY")] = "UPPER_BODY",
    mask_shape: Annotated[str, Field(description="Mask shape: CONTOUR or BOUNDING_BOX")] = "BOUNDING_BOX",
    mask_prompt: Annotated[str, Field(description="For PROMPT mask: describe item to replace")] = "",
    mask_image_path: Annotated[str, Field(description="For IMAGE mask: path to custom mask image")] = "",
    preserve_body_pose: Annotated[bool, Field(description="Preserve body pose of person")] = False,
    preserve_hands: Annotated[bool, Field(description="Preserve hands of person")] = False,
    preserve_face: Annotated[bool, Field(description="Preserve face of person")] = False,
    long_sleeve_style: Annotated[str, Field(description="NATURAL or ROLLED_UP")] = "NATURAL",
    tucking_style: Annotated[str, Field(description="NATURAL or TUCKED_IN")] = "NATURAL", 
    outer_layer_style: Annotated[str, Field(description="NATURAL or OPEN")] = "NATURAL",
    merge_style: Annotated[str, Field(description="BALANCED, SEAMLESS, or DETAILED")] = "BALANCED",
    return_mask: Annotated[bool, Field(description="Return the mask image used")] = False,
    quality: Annotated[str, Field(description="standard or premium")] = "standard",
    cfg_scale: Annotated[float, Field(description="CFG scale (1.0-10.0)", ge=1.0, le=10.0)] = 5.0,
    num_images: Annotated[int, Field(description="Number of images to generate (1-5)")] = 1,
    seed: Annotated[int, Field(description="Random seed (0-2147483647), -1 for random")] = -1,
    output_filename: Annotated[str, Field(description="Output filename prefix (without extension)")] = "vto_result"
) -> Dict[str, Any]:
    """Virtual try-on for garments, furniture, and objects using Amazon Nova Canvas"""
    
    try:
        # Validate parameters
        if mask_type not in ["GARMENT", "PROMPT", "IMAGE"]:
            return {"success": False, "error": "mask_type must be GARMENT, PROMPT, or IMAGE"}
        
        if mask_type == "GARMENT" and garment_class not in ["UPPER_BODY", "LOWER_BODY", "FOOTWEAR", "FULL_BODY"]:
            return {"success": False, "error": "garment_class must be UPPER_BODY, LOWER_BODY, FOOTWEAR, or FULL_BODY"}
        
        if merge_style not in ["BALANCED", "SEAMLESS", "DETAILED"]:
            return {"success": False, "error": "merge_style must be BALANCED, SEAMLESS, or DETAILED"}
        
        # Read and validate source image
        from PIL import Image
        with open(source_image_path, "rb") as image_file:
            source_data = image_file.read()
            source_base64 = base64.b64encode(source_data).decode('utf-8')
        
        # Check source image dimensions
        source_img = Image.open(source_image_path)
        if source_img.width < 320 or source_img.width > 4096 or source_img.height < 320 or source_img.height > 4096:
            return {"success": False, "error": f"Source image dimensions {source_img.width}x{source_img.height} invalid. Width and height must be 320-4096 pixels"}
        
        # Read and encode reference image
        with open(reference_image_path, "rb") as image_file:
            reference_data = image_file.read()
            reference_base64 = base64.b64encode(reference_data).decode('utf-8')
        
        # Generate random seed if not provided
        if seed == -1:
            seed = random.randint(0, 2147483647)
        
        # Build request payload
        request_payload = {
            "taskType": "VIRTUAL_TRY_ON",
            "virtualTryOnParams": {
                "sourceImage": source_base64,
                "referenceImage": reference_base64,
                "maskType": mask_type
            }
        }
        
        # Add mask-specific parameters
        if mask_type == "GARMENT":
            garment_mask = {
                "garmentClass": garment_class,
                "maskShape": mask_shape
            }
            
            # Add garment styling if not default
            styling = {}
            if long_sleeve_style != "NATURAL":
                styling["longSleeveStyle"] = long_sleeve_style
            if tucking_style != "NATURAL":
                styling["tuckingStyle"] = tucking_style
            if outer_layer_style != "NATURAL":
                styling["outerLayerStyle"] = outer_layer_style
            
            if styling:
                garment_mask["garmentStyling"] = styling
            
            request_payload["virtualTryOnParams"]["garmentBasedMask"] = garment_mask
            
        elif mask_type == "PROMPT":
            prompt_mask = {
                "maskPrompt": mask_prompt,
                "maskShape": mask_shape
            }
            request_payload["virtualTryOnParams"]["promptBasedMask"] = prompt_mask
            
        elif mask_type == "IMAGE":
            if not mask_image_path:
                return {"success": False, "error": "mask_image_path is required when mask_type is IMAGE"}
            
            if not os.path.exists(mask_image_path):
                return {"success": False, "error": f"Mask image file not found: {mask_image_path}"}
            
            # Read and encode mask image
            with open(mask_image_path, "rb") as mask_file:
                mask_data = mask_file.read()
                mask_base64 = base64.b64encode(mask_data).decode('utf-8')
            
            request_payload["virtualTryOnParams"]["imageBasedMask"] = {
                "maskImage": mask_base64
            }
        
        # Add mask exclusions (preserve features)
        exclusions = {}
        if preserve_body_pose:
            exclusions["preserveBodyPose"] = "ON"
        if preserve_hands:
            exclusions["preserveHands"] = "ON"
        if preserve_face:
            exclusions["preserveFace"] = "ON"
        
        if exclusions:
            request_payload["virtualTryOnParams"]["maskExclusions"] = exclusions
        
        # Add merge style
        request_payload["virtualTryOnParams"]["mergeStyle"] = merge_style
        
        # Add return mask option
        if return_mask:
            request_payload["virtualTryOnParams"]["returnMask"] = True
        
        # Add image generation config
        request_payload["imageGenerationConfig"] = {
            "numberOfImages": min(max(num_images, 1), 5),  # Clamp to 1-5
            "quality": quality,
            "cfgScale": cfg_scale,
            "seed": seed
        }
        
        logger.info(f"Virtual try-on: {mask_type} mask for {garment_class if mask_type == 'GARMENT' else 'object'}...")
        
        # Invoke Nova Canvas
        response = dpa_operator.bedrock_runtime.invoke_model(
            modelId="amazon.nova-canvas-v1:0",
            body=json.dumps(request_payload),
            contentType="application/json"
        )
        
        # Parse response
        response_body = json.loads(response['body'].read())
        
        if 'images' in response_body and response_body['images']:
            # Save all generated images
            output_paths = []
            for i, image_base64 in enumerate(response_body['images']):
                filename = f"{output_filename}_{i+1}" if num_images > 1 else output_filename
                save_result = _save_base64_image(image_base64, filename)
                
                if save_result["success"]:
                    output_paths.append({
                        "image_index": i + 1,
                        "output_path": save_result["output_path"],
                        "local_path": save_result["local_path"],
                        "file_size": save_result["file_size"]
                    })
            
            result = {
                "success": True,
                "images_generated": len(output_paths),
                "output_images": output_paths,
                "source_image_path": source_image_path,
                "reference_image_path": reference_image_path,
                "mask_type": mask_type,
                "garment_class": garment_class if mask_type == "GARMENT" else None,
                "mask_prompt": mask_prompt if mask_type == "PROMPT" else None,
                "preserve_settings": {
                    "body_pose": preserve_body_pose,
                    "hands": preserve_hands,
                    "face": preserve_face
                },
                "merge_style": merge_style,
                "quality": quality,
                "cfg_scale": cfg_scale,
                "seed": seed,
                "model": "amazon.nova-canvas-v1:0",
                "task_type": "VIRTUAL_TRY_ON"
            }
            
            # Save mask image if returned
            if return_mask and 'maskImage' in response_body:
                mask_save_result = _save_base64_image(response_body['maskImage'], f"{output_filename}_mask")
                if mask_save_result["success"]:
                    result["mask_image_path"] = mask_save_result["output_path"]
            
            return result
        else:
            return {"success": False, "error": "No images generated"}
            
    except FileNotFoundError as e:
        return {"success": False, "error": f"Image file not found: {str(e)}"}
    except Exception as e:
        logger.error(f"Error in virtual try-on: {str(e)}")
        return {"success": False, "error": str(e)}

@mcp.tool()
def resize_image_nova_canvas(
    image_path: Annotated[str, Field(description="Path to the source image file")],
    target_width: Annotated[int, Field(description="Target width (320-2048, multiple of 64)")] = 2048,
    target_height: Annotated[int, Field(description="Target height (320-2048, multiple of 64)")] = 2048,
    fill_prompt: Annotated[str, Field(description="Prompt for AI to fill empty areas")] = "seamless background extension",
    quality: Annotated[str, Field(description="Image quality: standard or premium")] = "standard",
    cfg_scale: Annotated[float, Field(description="CFG scale (1.0-10.0)")] = 3.0,
    seed: Annotated[int, Field(description="Random seed (0-858993459), -1 for random")] = -1,
    output_filename: Annotated[str, Field(description="Output filename (without extension and without directory name)")] = "resized_image"
) -> Dict[str, Any]:
    """Resize an image using Nova Canvas OUTPAINTING to fill empty areas intelligently"""
    
    try:
        from PIL import Image
        
        # Read source image
        original_image = Image.open(image_path)
        
        # Generate random seed if not provided
        if seed == -1:
            seed = random.randint(0, 858993459)
        
        # Validate dimensions
        target_width = ((target_width + 31) // 64) * 64
        target_height = ((target_height + 31) // 64) * 64
        target_width = max(320, min(2048, target_width))
        target_height = max(320, min(2048, target_height))
        
        # Check if resize is needed
        if original_image.size == (target_width, target_height):
            # Save original image with new filename
            save_result = _save_pil_image(original_image, output_filename)
            if not save_result["success"]:
                return save_result
            
            return {
                "success": True,
                "output_path": save_result["output_path"],
                "local_path": save_result["local_path"],
                "file_size": save_result["file_size"],
                "source_image_path": image_path,
                "target_dimensions": f"{target_width}x{target_height}",
                "message": "No resize needed - dimensions already match"
            }
        
        # Create canvas and mask
        canvas, mask = _create_canvas_and_mask(original_image, (target_width, target_height))
        
        # Convert to base64
        canvas_base64 = _pil_to_base64(canvas)
        mask_base64 = _pil_to_base64(mask)
        
        # Build request payload for OUTPAINTING
        request_payload = {
            "taskType": "OUTPAINTING",
            "outPaintingParams": {
                "image": canvas_base64,
                "maskImage": mask_base64,
                "text": fill_prompt,
                "outPaintingMode": "DEFAULT"
            },
            "imageGenerationConfig": {
                "seed": seed,
                "quality": quality,
                "numberOfImages": 1,
                "cfgScale": cfg_scale
            }
        }
        
        logger.info(f"Resizing image {image_path} to {target_width}x{target_height} using OUTPAINTING...")
        
        # Invoke Nova Canvas for outpainting
        response = dpa_operator.bedrock_runtime.invoke_model(
            modelId="amazon.nova-canvas-v1:0",
            body=json.dumps(request_payload),
            contentType="application/json"
        )
        
        # Parse response
        response_body = json.loads(response['body'].read())
        
        if 'images' in response_body and response_body['images']:
            resized_image_base64 = response_body['images'][0]
            
            # Save image to file
            save_result = _save_base64_image(resized_image_base64, output_filename)
            if not save_result["success"]:
                return save_result
            
            return {
                "success": True,
                "output_path": save_result["output_path"],
                "file_size": save_result["file_size"],
                "source_image_path": image_path,
                "original_dimensions": f"{original_image.size[0]}x{original_image.size[1]}",
                "target_dimensions": f"{target_width}x{target_height}",
                "fill_prompt": fill_prompt,
                "quality": quality,
                "cfg_scale": cfg_scale,
                "seed": seed,
                "model": "amazon.nova-canvas-v1:0",
                "method": "OUTPAINTING"
            }
        else:
            return {"success": False, "error": "No resized image generated"}
            
    except FileNotFoundError:
        return {"success": False, "error": f"Image file not found: {image_path}"}
    except Exception as e:
        logger.error(f"Error resizing image: {str(e)}")
        return {"success": False, "error": str(e)}

def _create_canvas_and_mask(original_image, target_size):
    """Create canvas and mask for outpainting (from demo_web.py approach)"""
    from PIL import Image
    
    orig_width, orig_height = original_image.size
    target_width, target_height = target_size
    
    # Ensure RGB mode
    if original_image.mode != 'RGB':
        original_image = original_image.convert('RGB')
    
    # Create canvas with neutral background
    canvas = Image.new('RGB', target_size, color=(128, 128, 128))
    
    # Calculate scale to fit original within target (don't upscale)
    scale = min(target_width / orig_width, target_height / orig_height, 1.0)
    
    # Calculate new dimensions
    new_width = int(orig_width * scale)
    new_height = int(orig_height * scale)
    
    # Resize if needed
    if scale < 1.0:
        original_image = original_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    else:
        new_width, new_height = orig_width, orig_height
    
    # Center the image on canvas
    x_offset = (target_width - new_width) // 2
    y_offset = (target_height - new_height) // 2
    
    # Paste original onto canvas
    canvas.paste(original_image, (x_offset, y_offset))
    
    # Create mask: BLACK for preserve, WHITE for fill
    mask = Image.new('RGB', target_size, color=(255, 255, 255))  # White background (areas to fill)
    black_area = Image.new('RGB', (new_width, new_height), color=(0, 0, 0))  # Black area (preserve original)
    mask.paste(black_area, (x_offset, y_offset))
    
    return canvas, mask

def _pil_to_base64(image):
    """Convert PIL Image to base64 string"""
    from PIL import Image
    from io import BytesIO
    
    # Ensure RGB mode
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Convert to base64
    buffer = BytesIO()
    image.save(buffer, format='PNG')
    image_bytes = buffer.getvalue()
    return base64.b64encode(image_bytes).decode('utf-8')

def _save_pil_image(image, filename):
    """Save PIL Image to file"""
    try:
        import os
        from PIL import Image
        
        # Use /tmp directory for writable access
        output_dir = "/tmp/generated_images"
        os.makedirs(output_dir, exist_ok=True)
        
        # Save image
        output_path = os.path.join(output_dir, f"{filename}.png")
        image.save(output_path, format='PNG')
        
        # Get file size
        file_size = os.path.getsize(output_path)
        
        return {
            "success": True,
            "output_path": output_path,
            "file_size": file_size
        }
        
    except Exception as e:
        logger.error(f"Error saving PIL image: {str(e)}")
        return {"success": False, "error": str(e)}

@mcp.tool()
def generate_mask_for_object(
    image_path: Annotated[str, Field(description="Path to the source image file")],
    object_category: Annotated[str, Field(description="Object category to detect (e.g., 'person', 'car', 'product')")],
    mask_type: Annotated[str, Field(description="Mask type: 'background' (white object, black background) or 'object' (black object, white background)")] = "background",
    individual_masks: Annotated[bool, Field(description="Generate separate mask files for each detected object")] = False,
    output_filename: Annotated[str, Field(description="Output mask filename (without extension)")] = "generated_mask"
) -> Dict[str, Any]:
    """Generate a mask for a specific object using Nova Pro image grounding"""
    
    try:
        from PIL import Image, ImageDraw
        import re
        
        # Load and resize image for detection
        image_pil = Image.open(image_path)
        orig_width, orig_height = image_pil.size
        
        # Resize for detection (smaller for faster processing)
        image_short_size = 360
        ratio = image_short_size / min(orig_width, orig_height)
        detect_width = round(ratio * orig_width)
        detect_height = round(ratio * orig_height)
        
        detect_image = image_pil.resize((detect_width, detect_height), Image.Resampling.LANCZOS)
        
        # Convert to base64 for Nova Pro
        from io import BytesIO
        buffer = BytesIO()
        detect_image.save(buffer, format="PNG")
        image_data = buffer.getvalue()
        image_base64 = base64.b64encode(image_data).decode('utf-8')
        
        # Create detection prompt
        detection_prompt = f"""
Detect bounding box of objects in the image, only detect "{object_category}" category objects with high confidence, output in a list of bounding box format.
Output example:
[
    {{"{object_category}": [x1, y1, x2, y2]}},
    ...
]
"""
        
        # Build message for Nova Pro
        message_list = [{
            "role": "user", 
            "content": [
                {
                    "image": {
                        "format": "png",
                        "source": {"bytes": image_base64}
                    }
                },
                {"text": detection_prompt}
            ]
        }]
        
        # Configure inference parameters
        inf_params = {
            "maxTokens": 1024,
            "temperature": 0.0
        }
        
        # Build request body
        request_body = {
            "schemaVersion": "messages-v1",
            "messages": message_list,
            "inferenceConfig": inf_params,
        }
        
        logger.info(f"Detecting {object_category} in image {image_path}...")
        
        # Invoke Nova Pro for object detection
        response = dpa_operator.bedrock_runtime.invoke_model(
            modelId="us.amazon.nova-pro-v1:0",
            body=json.dumps(request_body)
        )
        
        # Parse response
        response_body = json.loads(response['body'].read())
        
        if 'output' in response_body and 'message' in response_body['output']:
            detection_result = response_body['output']['message']['content'][0]['text']
            
            # Parse bounding boxes from result
            bboxes = _parse_detection_result(detection_result)
            
            if not bboxes:
                return {"success": False, "error": f"No {object_category} objects detected in image"}
            
            result = {
                "success": True,
                "source_image_path": image_path,
                "object_category": object_category,
                "mask_type": mask_type,
                "objects_detected": len(bboxes),
                "bounding_boxes": bboxes,
                "model": "us.amazon.nova-pro-v1:0"
            }
            
            if individual_masks:
                # Generate separate masks for each object
                mask_files = []
                for i, bbox in enumerate(bboxes):
                    individual_mask = _create_individual_mask_from_bbox(
                        bbox, 
                        (orig_width, orig_height), 
                        (detect_width, detect_height),
                        mask_type
                    )
                    
                    mask_filename = f"{output_filename}_{object_category}_{i+1}"
                    save_result = _save_pil_image(individual_mask, mask_filename)
                    
                    if save_result["success"]:
                        mask_files.append({
                            "object_index": i + 1,
                            "mask_path": save_result["output_path"],
                            "file_size": save_result["file_size"],
                            "bounding_box": bbox
                        })
                
                result["individual_masks"] = mask_files
            else:
                # Generate combined mask (original behavior)
                mask = _create_mask_from_bboxes(
                    bboxes, 
                    (orig_width, orig_height), 
                    (detect_width, detect_height),
                    mask_type
                )
                
                save_result = _save_pil_image(mask, output_filename)
                if not save_result["success"]:
                    return save_result
                
                result["mask_path"] = save_result["output_path"]
                result["file_size"] = save_result["file_size"]
            
            return result
        else:
            return {"success": False, "error": "No detection result from Nova Pro"}
            
    except FileNotFoundError:
        return {"success": False, "error": f"Image file not found: {image_path}"}
    except Exception as e:
        logger.error(f"Error generating mask: {str(e)}")
        return {"success": False, "error": str(e)}

def _parse_detection_result(detection_text):
    """Parse detection result to extract bounding boxes"""
    import re
    import json
    
    try:
        # Clean up the text for JSON parsing
        json_string = re.sub(r"\s", "", detection_text)
        json_string = re.sub(r"\(", "[", json_string)
        json_string = re.sub(r"\)", "]", json_string)
        
        # Find all bounding box patterns
        bbox_pattern = r"\[(\d+),(\d+),(\d+),(\d+)\]"
        matches = re.findall(bbox_pattern, json_string)
        
        bboxes = []
        for match in matches:
            x1, y1, x2, y2 = map(int, match)
            if x1 < x2 and y1 < y2:  # Valid bounding box
                bboxes.append([x1, y1, x2, y2])
        
        return bboxes
        
    except Exception as e:
        logger.error(f"Error parsing detection result: {str(e)}")
        return []

def _create_mask_from_bboxes(bboxes, original_size, detect_size, mask_type):
    """Create mask image from bounding boxes"""
    from PIL import Image, ImageDraw
    
    orig_width, orig_height = original_size
    detect_width, detect_height = detect_size
    
    # Create mask at original image size
    if mask_type == "background":
        # White object, black background (for object replacement)
        mask = Image.new('RGB', original_size, color=(0, 0, 0))  # Black background
        fill_color = (255, 255, 255)  # White for object area
    else:
        # Black object, white background (for background generation)
        # BUT Nova Canvas INPAINTING expects INVERTED masks!
        mask = Image.new('RGB', original_size, color=(0, 0, 0))  # Black background  
        fill_color = (255, 255, 255)  # White for background areas (inverted)
    
    draw = ImageDraw.Draw(mask)
    
    # Scale bounding boxes to original image size
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        
        # Convert from Nova Pro coordinates (0-1000) to detection image coordinates
        x1 = (x1 / 1000) * detect_width
        x2 = (x2 / 1000) * detect_width
        y1 = (y1 / 1000) * detect_height
        y2 = (y2 / 1000) * detect_height
        
        # Scale to original image size
        x1 = (x1 / detect_width) * orig_width
        x2 = (x2 / detect_width) * orig_width
        y1 = (y1 / detect_height) * orig_height
        y2 = (y2 / detect_height) * orig_height
        
        if mask_type == "object":
            # White object area (for replacement)
            draw.rectangle([x1, y1, x2, y2], fill=fill_color)
        else:
            # For background generation, we need to fill EVERYTHING EXCEPT the object
            # First fill entire image white, then make object area black
            mask = Image.new('RGB', original_size, color=(255, 255, 255))  # White background
            draw = ImageDraw.Draw(mask)
            draw.rectangle([x1, y1, x2, y2], fill=(0, 0, 0))  # Black object area
    
    return mask
    """Internal helper function to save base64 image to file"""
    try:
        import os
        
        # Decode base64 image
        image_data = base64.b64decode(base64_image)
        
        # Use /tmp directory for writable access
        output_dir = "/tmp/generated_images"
        os.makedirs(output_dir, exist_ok=True)
        
        # Save image
        output_path = os.path.join(output_dir, f"{filename}.png")
        with open(output_path, "wb") as f:
            f.write(image_data)
        
        return {
            "success": True,
            "output_path": output_path,
            "file_size": len(image_data)
        }
        
    except Exception as e:
        logger.error(f"Error saving image: {str(e)}")
        return {"success": False, "error": str(e)}


def _create_individual_mask_from_bbox(bbox, original_size, detect_size, mask_type):
    """Create mask image from single bounding box"""
    from PIL import Image, ImageDraw
    
    orig_width, orig_height = original_size
    detect_width, detect_height = detect_size
    
    # Scale bounding box to original image size
    x1, y1, x2, y2 = bbox
    
    # Convert from Nova Pro coordinates (0-1000) to detection image coordinates
    x1 = (x1 / 1000) * detect_width
    x2 = (x2 / 1000) * detect_width
    y1 = (y1 / 1000) * detect_height
    y2 = (y2 / 1000) * detect_height
    
    # Scale to original image size
    x1 = (x1 / detect_width) * orig_width
    x2 = (x2 / detect_width) * orig_width
    y1 = (y1 / detect_height) * orig_height
    y2 = (y2 / detect_height) * orig_height
    
    # Create mask at original image size
    if mask_type == "object":
        # White object, black background (for object replacement)
        mask = Image.new('RGB', original_size, color=(0, 0, 0))  # Black background
        draw = ImageDraw.Draw(mask)
        draw.rectangle([x1, y1, x2, y2], fill=(255, 255, 255))  # White object area
    else:
        # Black object, white background (for background generation)
        mask = Image.new('RGB', original_size, color=(255, 255, 255))  # White background
        draw = ImageDraw.Draw(mask)
        draw.rectangle([x1, y1, x2, y2], fill=(0, 0, 0))  # Black object area
    
    return mask

def _save_base64_image(base64_image: str, filename: str) -> Dict[str, Any]:
    """Internal helper function to save base64 image locally and optionally to S3/CloudFront"""
    try:
        import os
        import uuid
        from datetime import datetime
        
        # Decode base64 image
        image_data = base64.b64decode(base64_image)
        
        # Always save locally first (needed for post-processing)
        output_dir = "/tmp/generated_images"
        os.makedirs(output_dir, exist_ok=True)
        
        # Create unique filename with timestamp and UUID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        local_filename = f"{timestamp}_{unique_id}_{filename}.png"
        local_path = os.path.join(output_dir, local_filename)
        
        # Save image locally
        with open(local_path, "wb") as f:
            f.write(image_data)
        
        logger.info(f"Image saved locally: {local_path}")
        
        # Prepare base response with local path
        response = {
            "success": True,
            "local_path": local_path,
            "file_size": len(image_data),
            "storage_type": "local"
        }
        
        # Get S3 bucket and CloudFront endpoint from config
        s3_bucket = dpa_config.DPA_S3_BUCKET
        cloudfront_endpoint = dpa_config.CLOUDFRONT_ENDPOINT
        
        logger.info(f"S3 Bucket: {s3_bucket}")
        logger.info(f"CloudFront Endpoint: {cloudfront_endpoint}")
        
        if s3_bucket:
            # Also save to S3 and add CloudFront/S3 URLs
            try:
                s3_client = dpa_operator.session.client('s3')
                
                # Use same filename for S3 key
                s3_key = f"dpa-images/{local_filename}"
                
                # Upload to S3
                s3_client.put_object(
                    Bucket=s3_bucket,
                    Key=s3_key,
                    Body=image_data,
                    ContentType='image/png',
                    Metadata={
                        'generated_by': 'dpa-agent',
                        'timestamp': timestamp,
                        'original_filename': filename,
                        'local_path': local_path
                    }
                )
                
                logger.info(f"Image also saved to S3: s3://{s3_bucket}/{s3_key}")
                
                # Add S3 information to response
                response.update({
                    "s3_bucket": s3_bucket,
                    "s3_key": s3_key,
                    "storage_type": "local_and_s3"
                })
                
                # Generate CloudFront URL if endpoint is configured
                if cloudfront_endpoint:
                    # Remove protocol if present and ensure proper format
                    cloudfront_domain = cloudfront_endpoint.replace("https://", "").replace("http://", "")
                    cloudfront_url = f"https://{cloudfront_domain}/{s3_key}"
                    
                    logger.info(f"CloudFront URL generated: {cloudfront_url}")
                    
                    response.update({
                        "output_path": cloudfront_url,  # Primary URL for user access

                        "storage_type": "local_and_cloudfront"
                    })
                else:
                    # Fallback to S3 presigned URL if CloudFront not configured
                    presigned_url = s3_client.generate_presigned_url(
                        'get_object',
                        Params={'Bucket': s3_bucket, 'Key': s3_key},
                        ExpiresIn=86400  # 24 hours
                    )
                    
                    logger.warning("CloudFront endpoint not configured, using S3 presigned URL")
                    
                    response.update({
                        "output_path": presigned_url,  # Primary URL for user access
                        "s3_presigned_url": presigned_url,
                        "storage_type": "local_and_s3_presigned"
                    })
                
            except Exception as s3_error:
                logger.warning(f"S3 upload failed: {s3_error}, using local storage only")
                response.update({
                    "output_path": local_path,  # Use local path as primary if S3 fails
                    "s3_error": str(s3_error)
                })
        else:
            # No S3 configured, use local path as primary
            response["output_path"] = local_path
        
        return response
        
    except Exception as e:
        logger.error(f"Error saving image: {str(e)}")
        return {"success": False, "error": str(e)}

if __name__ == "__main__":
    mcp.run()
