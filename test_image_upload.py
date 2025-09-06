#!/usr/bin/env python3
"""
Test script for uploading images to the deployed DPA Agent.
"""

import json
import uuid
import boto3
import base64
import os
from pathlib import Path

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

def encode_image(image_path: str) -> str:
    """Encode image file to base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def test_single_image_upload(agent_arn: str, image_path: str, prompt: str):
    """Test uploading a single image to the DPA agent."""
    
    # Initialize AgentCore client
    client = boto3.client('bedrock-agentcore')
    
    # Encode the image
    print(f"📤 Encoding image: {image_path}")
    image_base64 = encode_image(image_path)
    image_filename = Path(image_path).name
    
    # Create payload with single image
    payload = {
        "prompt": prompt,
        "image": image_base64,
        "image_filename": image_filename
    }
    
    print(f"🚀 Sending request to agent...")
    print(f"   Prompt: {prompt}")
    print(f"   Image: {image_filename} ({len(image_base64)} chars)")
    
    return _send_request(client, agent_arn, payload)

def test_multiple_images_upload(agent_arn: str, image_paths: list, prompt: str):
    """Test uploading multiple images to the DPA agent."""
    
    # Initialize AgentCore client
    client = boto3.client('bedrock-agentcore')
    
    # Encode all images
    images_data = []
    total_size = 0
    
    for image_path in image_paths:
        print(f"📤 Encoding image: {image_path}")
        image_base64 = encode_image(image_path)
        image_filename = Path(image_path).name
        
        images_data.append({
            "data": image_base64,
            "filename": image_filename
        })
        total_size += len(image_base64)
    
    # Create payload with multiple images
    payload = {
        "prompt": prompt,
        "images": images_data
    }
    
    print(f"🚀 Sending request to agent...")
    print(f"   Prompt: {prompt}")
    print(f"   Images: {len(images_data)} files ({total_size} total chars)")
    for img_data in images_data:
        print(f"     • {img_data['filename']}")
    
    return _send_request(client, agent_arn, payload)

def _send_request(client, agent_arn: str, payload: dict):
    """Send request to agent and process response."""
    try:
        # Convert payload to bytes as required by AgentCore API
        payload_bytes = json.dumps(payload).encode('utf-8')
        
        # Invoke the agent with correct AgentCore API format
        response = client.invoke_agent_runtime(
            agentRuntimeArn=agent_arn,
            runtimeSessionId=str(uuid.uuid4()),
            payload=payload_bytes
        )
        
        # Process response - handle StreamingBody
        result = ""
        print("\n📝 Agent Response:")
        print("-" * 40)
        
        # Check if response contains a StreamingBody (as shown in your error message)
        if 'response' in response and hasattr(response['response'], 'read'):
            # Read the streaming body
            stream_data = response['response'].read()
            try:
                # Try to decode as text
                result = stream_data.decode('utf-8')
                print(f"📝 Read {len(result)} characters from streaming response")
                
                # Filter out base64 data from display (but keep in result)
                display_text = _filter_base64_from_display(result)
                if display_text.strip():
                    print(display_text)
                    
            except UnicodeDecodeError:
                # If it's binary data, convert to base64 for display
                result = base64.b64encode(stream_data).decode('utf-8')
                print(f"📝 Received binary data ({len(stream_data)} bytes) - converted to base64")
                print("[BINARY_DATA_AS_BASE64]")
                
        # Handle other possible response structures
        elif 'body' in response:
            # If response has body (streaming)
            for event in response['body']:
                if 'chunk' in event:
                    chunk = event['chunk']
                    if 'bytes' in chunk:
                        chunk_text = chunk['bytes'].decode('utf-8')
                        result += chunk_text
                        
                        # Filter out base64 data from display (but keep in result)
                        display_text = _filter_base64_from_display(chunk_text)
                        if display_text.strip():
                            print(display_text, end='', flush=True)
        elif 'completion' in response:
            # Alternative response structure
            for event in response['completion']:
                if 'chunk' in event:
                    chunk = event['chunk']
                    if 'bytes' in chunk:
                        chunk_text = chunk['bytes'].decode('utf-8')
                        result += chunk_text
                        
                        # Filter out base64 data from display (but keep in result)
                        display_text = _filter_base64_from_display(chunk_text)
                        if display_text.strip():
                            print(display_text, end='', flush=True)
        else:
            # Direct response (non-streaming)
            if isinstance(response, dict) and 'result' in response:
                result = str(response['result'])
                display_text = _filter_base64_from_display(result)
                print(display_text)
            else:
                result = str(response)
                display_text = _filter_base64_from_display(result)
                print(display_text)
        
        print(f"\n\n✅ Request completed successfully!")
        print(f"📊 Total response length: {len(result)} characters")
        
        # Check if response contains S3 URLs
        if "s3.amazonaws.com" in result or "presigned" in result.lower():
            print("🔗 Response contains S3 URLs for generated images")
        
        return result
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print(f"🔍 Error type: {type(e)}")
        
        # Show response structure if available
        if 'response' in locals():
            print(f"🔍 Response type: {type(response)}")
            if isinstance(response, dict):
                print(f"🔍 Response keys: {list(response.keys())}")
                # Show non-sensitive response metadata
                for key, value in response.items():
                    if key != 'response':  # Skip the StreamingBody
                        print(f"🔍   {key}: {type(value)} = {str(value)[:100]}...")
                    else:
                        print(f"🔍   {key}: {type(value)} (StreamingBody)")
            else:
                print(f"🔍 Response: {str(response)[:200]}...")
        
        # Show full traceback for debugging
        import traceback
        print(f"🔍 Full traceback:")
        traceback.print_exc()
        
        return None

def _filter_base64_from_display(text: str) -> str:
    """Filter out base64 data from display while keeping readable content."""
    import re
    
    # Pattern to match base64 strings (typically long strings of alphanumeric + / + =)
    base64_pattern = r'[A-Za-z0-9+/]{50,}={0,2}'
    
    # Replace long base64 strings with placeholder
    filtered_text = re.sub(base64_pattern, '[BASE64_DATA_HIDDEN]', text)
    
    # If the entire text was base64, show a summary
    if filtered_text.strip() == '[BASE64_DATA_HIDDEN]':
        return f"[Generated image data - {len(text)} chars] "
    
    return filtered_text

def _display_result_summary(result: str):
    """Display a summary of the agent's response."""
    import re
    
    print(f"\n📊 Result Summary:")
    print(f"   • Total length: {len(result)} characters")
    
    # Check for S3 URLs
    s3_urls = re.findall(r'https://[^/]+\.s3\.amazonaws\.com/[^\s]+', result)
    if s3_urls:
        print(f"   • Generated images: {len(s3_urls)}")
        for i, url in enumerate(s3_urls, 1):
            # Extract filename from URL
            filename = url.split('/')[-1].split('?')[0]
            print(f"     {i}. {filename}")
    
    # Check for key phrases
    key_phrases = [
        "Image available at:",
        "Image saved to:",
        "analysis:",
        "advertising",
        "professional",
        "background"
    ]
    
    found_phrases = []
    for phrase in key_phrases:
        if phrase.lower() in result.lower():
            found_phrases.append(phrase)
    
    if found_phrases:
        print(f"   • Key topics: {', '.join(found_phrases[:3])}")
    
    # Show first 200 chars of readable content (non-base64)
    readable_content = _filter_base64_from_display(result)
    if len(readable_content) > 200:
        preview = readable_content[:200] + "..."
    else:
        preview = readable_content
    
    if preview.strip() and preview.strip() != '[BASE64_DATA_HIDDEN]':
        print(f"   • Preview: {preview.strip()}")

def test_simple_text_only():
    """Test with simple text-only request first to verify API format."""
    agent_arn = "arn:aws:bedrock-agentcore:us-west-2:322216473749:runtime/dpa_agent-D8Xdy45Kiy"
    client = boto3.client('bedrock-agentcore')
    
    # Simple text-only payload
    simple_payload = {
        "prompt": "Hello! Can you help me create product advertising content?"
    }
    
    print("🧪 Testing simple text-only request first...")
    result = _send_request(client, agent_arn, simple_payload)
    
    if result:
        print("✅ Simple test successful!")
        return True
    else:
        print("❌ Simple test failed!")
        return False

def main():
    """Main test function with real test scenarios."""
    
    # Configuration - Your deployed agent ARN
    agent_arn = "arn:aws:bedrock-agentcore:us-west-2:322216473749:runtime/dpa_agent-D8Xdy45Kiy"
    
    # Define test images based on the provided images
    breakdancer_image = "/private/tmp/generated_images/falling_person.png"  # The breakdancer/street performer image
    dogs_plaza_image = "/private/tmp/generated_images/dog_replaced_with_car.png"    # The dogs in European plaza image
    
    # Single image test cases using the provided images
    single_image_tests = [
        {
            "image_path": breakdancer_image,
            "prompt": "Analyze this dynamic action photo and create a professional advertising image for athletic wear or sports equipment. Enhance the energy and movement while maintaining the urban street vibe."
        },
        {
            "image_path": breakdancer_image,
            "prompt": "Transform this street performance image into a high-end advertisement for fitness or dance products. Replace the background with a modern studio setting while preserving the dancer's pose and energy."
        },
        {
            "image_path": dogs_plaza_image,
            "prompt": "Create a professional pet product advertisement using this charming plaza scene. Focus on the dogs and enhance the European elegance of the setting for premium pet accessories marketing."
        },
        {
            "image_path": dogs_plaza_image,
            "prompt": "Analyze this urban pet scene and suggest how to optimize it for pet food or pet care service advertising. What elements make it effective for marketing?"
        }
    ]
    
    # Multiple image test cases
    multiple_image_tests = [
        {
            "image_paths": [breakdancer_image, dogs_plaza_image],
            "prompt": "Compare these two lifestyle images for advertising potential. The first shows dynamic action/sports, the second shows peaceful pet companionship. Analyze which elements from each could be combined for a lifestyle brand campaign."
        },
        {
            "image_paths": [breakdancer_image, dogs_plaza_image],
            "prompt": "Create a concept for a 'Life in Motion' advertising campaign using these contrasting scenes - one showing human athleticism, the other showing peaceful pet life. How would you adapt each image for the campaign?"
        }
    ]
    
    print("🧪 Testing DPA Agent with Real Images")
    print("🎨 Breakdancer & Dogs Plaza Test Scenarios")
    print("=" * 60)
    
    # Check if test images exist
    available_images = []
    for img_path in [breakdancer_image, dogs_plaza_image]:
        if Path(img_path).exists():
            available_images.append(img_path)
            print(f"✅ Found test image: {img_path}")
        else:
            print(f"❌ Missing test image: {img_path}")
    
    if not available_images:
        print("\n⚠️  No test images found!")
        print("Please save the provided images as:")
        print(f"   • {breakdancer_image} (the breakdancer/street performer)")
        print(f"   • {dogs_plaza_image} (the dogs in European plaza)")
        return
    
    # Test single image uploads
    print(f"\n📷 SINGLE IMAGE TESTS ({len(available_images)} images available)")
    print("=" * 40)
    
    for i, test_case in enumerate(single_image_tests, 1):
        if not Path(test_case["image_path"]).exists():
            print(f"\n⚪ Skipping Test {i}: {test_case['image_path']} not found")
            continue
            
        print(f"\n🔍 Single Image Test {i}: {Path(test_case['image_path']).name}")
        print(f"   Scenario: {test_case['prompt'][:80]}...")
        
        # Run single image test
        result = test_single_image_upload(
            agent_arn=agent_arn,
            image_path=test_case["image_path"],
            prompt=test_case["prompt"]
        )
        
        if result:
            _display_result_summary(result)
        
        print("-" * 60)
    
    # Test multiple image uploads (only if both images are available)
    if len(available_images) >= 2:
        print(f"\n📷📷 MULTIPLE IMAGE TESTS")
        print("=" * 40)
        
        for i, test_case in enumerate(multiple_image_tests, 1):
            # Check if all required images exist
            missing_files = [img for img in test_case["image_paths"] if not Path(img).exists()]
            
            if missing_files:
                print(f"\n⚪ Skipping Test {i}: Missing {missing_files}")
                continue
            
            print(f"\n🔍 Multiple Image Test {i}: {[Path(p).name for p in test_case['image_paths']]}")
            print(f"   Scenario: {test_case['prompt'][:80]}...")
            
            # Run multiple image test
            result = test_multiple_images_upload(
                agent_arn=agent_arn,
                image_paths=test_case["image_paths"],
                prompt=test_case["prompt"]
            )
            
            if result:
                _display_result_summary(result)
            
            print("-" * 60)
    else:
        print(f"\n⚪ Skipping multiple image tests - need both test images")

if __name__ == "__main__":
    print("🎨 DPA Agent - Real Image Testing")
    print("Testing with breakdancer and dogs plaza images")
    print("=" * 50)
    
    # Check AWS credentials
    try:
        boto3.Session().get_credentials()
        print("✅ AWS credentials configured")
    except:
        print("❌ AWS credentials not found - please configure AWS CLI or environment variables")
        exit(1)
    
    # Check if test images exist
    breakdancer_exists = Path("/private/tmp/generated_images/falling_person.png").exists()
    dogs_plaza_exists = Path("/private/tmp/generated_images/dog_replaced_with_car.png").exists()

    print(f"📷 Test Images Status:")
    print(f"   • breakdancer.jpg: {'✅ Found' if breakdancer_exists else '❌ Missing'}")
    print(f"   • dogs_plaza.jpg: {'✅ Found' if dogs_plaza_exists else '❌ Missing'}")
    
    if not (breakdancer_exists or dogs_plaza_exists):
        print("\n⚠️  No test images found!")
        print("Please save the provided images as 'breakdancer.jpg' and 'dogs_plaza.jpg'")
        print("in the same directory as this script.")
        exit(1)
    
    print(f"\n🎯 Agent ARN: arn:aws:bedrock-agentcore:us-west-2:322216473749:runtime/dpa_agent-D8Xdy45Kiy")
    print()
    
    response = input("Continue with image upload tests? (y/N): ").strip().lower()
    if response in ['y', 'yes']:
        main()
    else:
        print("Test cancelled.")