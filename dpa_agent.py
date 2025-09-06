#!/usr/bin/env python3
"""
DPA Agent for AWS AgentCore Runtime - Dynamic Product Advertising Agent using Nova Models

This version is adapted for AWS AgentCore Runtime deployment using the BedrockAgentCoreApp.
"""

import os
import logging
from typing import Optional, Dict, Any
from bedrock_agentcore import BedrockAgentCoreApp
from strands import Agent
from strands.models import BedrockModel
from strands.tools.mcp import MCPClient
from mcp import StdioServerParameters, stdio_client

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not available, use system environment variables

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the AgentCore app
app = BedrockAgentCoreApp()

# Global agent instance
agent_instance = None
mcp_client = None

def initialize_agent():
    """Initialize the DPA agent with MCP tools."""
    global agent_instance, mcp_client
    
    if agent_instance is not None:
        return agent_instance
    
    try:
        # MCP server path - can be configured via environment variable
        mcp_server_path = os.getenv("MCP_SERVER_PATH", "./dpa_mcp_server.py")
        
        # Check if MCP server file exists
        if not os.path.exists(mcp_server_path):
            logger.warning(f"MCP server file not found: {mcp_server_path}")
            logger.info("Creating agent without MCP tools...")
            tools = []
        else:
            # Create MCP client
            mcp_client = MCPClient(
                lambda: stdio_client(
                    StdioServerParameters(
                        command="python",
                        args=[mcp_server_path]
                    )
                )
            )
            
            # Enter MCP client context
            mcp_client.__enter__()
            
            # Get tools
            tools = mcp_client.list_tools_sync()
            logger.info(f"🎨 DPA Agent initialized with {len(tools)} Nova tools")
        
        # Use Claude 3.5 Sonnet for better reasoning about creative tasks
        model = BedrockModel(
            model_id="us.anthropic.claude-sonnet-4-20250514-v1:0",
            temperature=0.7,
        )
        
        # Specialized system prompt for DPA tasks
        system_prompt = """
        You are a Dynamic Product Advertising (DPA) specialist with access to Amazon Nova models.
        
        Your capabilities include:
        🎨 IMAGE GENERATION (Nova Canvas):
        - generate_image_nova_canvas: Create product advertising images
        - outpaint_image_nova_canvas: Extend/expand existing images
        - outpaint_with_object_mask: Edit images by keeping the mask part, and change the part that doesn't mask.
        - inpaint_image_nova_canvas: Edit/modify parts of images that being mask by text or mask image
        
        🎬 VIDEO GENERATION (Nova Reel):
        - generate_video_nova_reel: Create product videos (async)
        - check_video_generation_status: Monitor video generation progress
        
        📝 TEXT & ANALYSIS (Nova Pro):
        - generate_text_nova_pro: Generate marketing copy and descriptions
        - understand_image_nova_pro: Analyze product images
        - optimize_dpa_prompt: Optimize prompts for better image generation
        
        🔍 UPLOADED IMAGE HANDLING PROTOCOL:
        When you see "[UPLOADED IMAGE X: /path/to/image]" patterns in the user message, you MUST:
        1. FIRST: Use understand_image_nova_pro to analyze each uploaded image at those paths
        2. IDENTIFY: What type of product or subject is in each image
        3. PRESERVE: Keep the core product/subject intact in any modifications
        4. FOLLOW: User's specific instructions while maintaining product integrity
        5. CONTEXT: Always consider the advertising/marketing context
        
        MULTIPLE IMAGE SCENARIOS:
        - Single image: "[UPLOADED IMAGE 1: /path/to/image]"
        - Multiple images: "[UPLOADED IMAGE 1: /path1]", "[UPLOADED IMAGE 2: /path2]", etc.
        - For multiple images, analyze each one separately first
        - Common use cases: product + background, before + after, source + reference for virtual try-on
        
        UPLOADED IMAGE DETECTION:
        - Look for "[UPLOADED IMAGE X: /path/to/image]" patterns in user messages
        - Extract the file paths and use them with image processing tools
        - Always analyze uploaded images first before making modifications
        - For multiple images, determine the relationship between them (e.g., product + background, source + reference)
        
        🎯 PRODUCT ADVERTISING WORKFLOW (Most Common):
        For background changes/removal (the standard advertising workflow):
        1. Analyze uploaded product image with understand_image_nova_pro
        2. Use outpaint_with_object_mask to automatically:
           - Detect and preserve the product (object_category: 'product')
           - Remove/replace the background with new advertising scene
           - Generate professional advertising backgrounds
        3. Alternative: Use outpaint_image_nova_canvas with manual mask if needed
        
        IMPORTANT RESPONSE FORMATTING:
        - When you generate, edit, or create images using any Nova Canvas tools, ALWAYS include the complete URL in your response
        - Look for "output_path" in the tool results and include it in your response
        - For S3 storage: "Image available at: [presigned_url]" (24-hour access)
        - For local storage: "Image saved to: [local_path]" (development only)
        - Check the "storage_type" field to determine the appropriate message
        - Be explicit about where images are accessible
        
        🛠️ OUTPAINT TOOL SELECTION GUIDE:
        
        PRIMARY CHOICE - outpaint_with_object_mask:
        - Use for automatic product detection and background replacement
        - Best for: product photos, e-commerce images, advertising
        - Parameters: object_category='product' (or 'person', 'car', etc.)
        - Automatically creates mask around the product
        - Generates new background while preserving product perfectly
        
        SECONDARY CHOICE - outpaint_image_nova_canvas:
        - Use when you need precise control over what to preserve
        - Requires manual mask creation or specific mask image
        - Best for: complex scenes, multiple objects, artistic control
        
        PRODUCT PRESERVATION RULES:
        - Never alter the core product unless explicitly requested
        - Maintain product proportions, colors, and key features
        - Focus modifications on background, lighting, or context
        - If unsure about product details, ask for clarification
        - Always explain what you're preserving vs. what you're changing
        - PREFER outpaint_with_object_mask for standard product advertising
        
        BEST PRACTICES:
        - Always optimize prompts for commercial/advertising quality
        - Use professional photography terminology
        - Consider lighting, composition, and brand aesthetics
        - For videos, keep descriptions clear and engaging
        - When analyzing images, focus on advertising effectiveness
        - Respect brand guidelines and product integrity
        
        Be creative, professional, and results-oriented while always preserving the essence of uploaded products.
        """
        
        # Create agent with tools
        agent_instance = Agent(
            tools=tools,
            model=model,
            system_prompt=system_prompt
        )
        
        logger.info("✅ DPA Agent initialized successfully")
        return agent_instance
        
    except Exception as e:
        logger.error(f"Failed to initialize agent: {e}")
        raise


def _normalize_streaming_event(event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Normalize streaming events from different formats to a consistent format.
    Prioritizes processed formats over raw Bedrock events to avoid duplicates.
    
    Args:
        event: Raw streaming event from Strands agent
        
    Returns:
        Normalized event with consistent format, or None if no text content
    """
    try:
        text_content = None
        event_priority = 0  # Higher number = higher priority
        
        # Format 1: Processed format (HIGHEST PRIORITY - prefer this)
        # {'data': '...', 'delta': {'text': '...'}}
        if isinstance(event, dict) and 'data' in event:
            text_content = event['data']
            event_priority = 3
        
        # Format 2: Direct delta format (HIGH PRIORITY)
        # {'delta': {'text': '...'}}
        elif isinstance(event, dict) and 'delta' in event and 'text' in event['delta']:
            text_content = event['delta'].get('text')
            event_priority = 2
        
        # Format 3: Raw Bedrock format (LOWER PRIORITY - only if no processed format)
        # {'event': {'contentBlockDelta': {'delta': {'text': '...'}, 'contentBlockIndex': 0}}}
        elif isinstance(event, dict) and 'event' in event:
            bedrock_event = event['event']
            if 'contentBlockDelta' in bedrock_event:
                delta = bedrock_event['contentBlockDelta'].get('delta', {})
                text_content = delta.get('text')
                event_priority = 1
        
        # Format 4: Direct text (MEDIUM PRIORITY)
        elif isinstance(event, dict) and 'text' in event:
            text_content = event['text']
            event_priority = 2
        
        # Format 5: String content (LOW PRIORITY)
        elif isinstance(event, str):
            text_content = event
            event_priority = 1
        
        # Return normalized format if we found text content
        if text_content and text_content.strip():  # Ignore empty/whitespace-only content
            return {
                "type": "content_delta",
                "text": text_content,
                "priority": event_priority,
                "timestamp": None
            }
        
        # For non-text events (tool calls, etc.), pass through as-is but mark the type
        elif isinstance(event, dict):
            # Check if this might be a tool call or other important event
            event_str = str(event).lower()
            if any(keyword in event_str for keyword in ['tool', 'function', 'call']):
                return {
                    "type": "tool_event",
                    "raw_event": event
                }
            else:
                return {
                    "type": "other",
                    "raw_event": event
                }
        
        return None
        
    except Exception as e:
        logger.warning(f"Error normalizing streaming event: {e}")
        # Return the raw event if normalization fails
        return {
            "type": "raw",
            "raw_event": event
        }

@app.entrypoint
async def invoke(payload: Dict[str, Any]):
    """
    AgentCore Runtime entrypoint for the DPA agent.
    
    Args:
        payload: The request payload containing the user message and optional image
        
    Returns:
        Dictionary containing the agent's response
    """
    global agent_instance
    
    try:
        # Create fresh agent for each request to avoid conversation state issues
        # This prevents tool_use/tool_result mismatches between requests
        agent_instance = None
        agent = initialize_agent()
        
        # Extract user message from payload
        user_message = payload.get("prompt", payload.get("message", "Hello! How can I help you with product advertising today?"))
        
        # Handle image uploads if present (single or multiple)
        uploaded_image_paths = []
        
        # Handle single image (backward compatibility)
        if "image" in payload:
            image_path = _save_uploaded_image(payload["image"], payload.get("image_filename", "uploaded_image"))
            if image_path:
                uploaded_image_paths.append(image_path)
        
        # Handle multiple images
        if "images" in payload:
            images_data = payload["images"]
            if isinstance(images_data, list):
                for i, image_data in enumerate(images_data):
                    if isinstance(image_data, dict):
                        # Format: {"data": "base64...", "filename": "image.jpg"}
                        image_path = _save_uploaded_image(
                            image_data.get("data", image_data.get("image")),
                            image_data.get("filename", f"uploaded_image_{i+1}")
                        )
                    else:
                        # Format: ["base64_data1", "base64_data2", ...]
                        image_path = _save_uploaded_image(image_data, f"uploaded_image_{i+1}")
                    
                    if image_path:
                        uploaded_image_paths.append(image_path)
        
        # Modify user message to include image references
        if uploaded_image_paths:
            image_refs = []
            for i, path in enumerate(uploaded_image_paths, 1):
                image_refs.append(f"[UPLOADED IMAGE {i}: {path}]")
            
            image_instructions = "\n".join(image_refs)
            if len(uploaded_image_paths) == 1:
                analysis_instruction = "\n\nPlease analyze this uploaded image first using understand_image_nova_pro, then proceed with the requested task."
            else:
                analysis_instruction = f"\n\nPlease analyze these {len(uploaded_image_paths)} uploaded images first using understand_image_nova_pro for each one, then proceed with the requested task."
            
            user_message = f"{user_message}\n\n{image_instructions}{analysis_instruction}"
            logger.info(f"{len(uploaded_image_paths)} image(s) uploaded and saved")
        
        # Process the request with the agent
        logger.info(f"Processing request: {user_message[:100]}...")
        
        # Check if streaming is requested (default to streaming)
        enable_streaming = payload.get("stream", True)
        
        if enable_streaming:
            # Stream the response
            try:
                stream = agent.stream_async(user_message)
                seen_content = set()  # Track seen content to avoid duplicates
                last_content_hash = None
                
                async for event in stream:
                    logger.info(f"Streaming event: {event}")
                    # Normalize the streaming event format
                    normalized_event = _normalize_streaming_event(event)
                    if normalized_event:
                        # Deduplicate text content
                        if normalized_event.get("type") == "content_delta":
                            current_text = normalized_event.get("text", "")
                            if current_text:
                                # Create a hash of the content for deduplication
                                content_hash = hash(current_text)
                                
                                # Only yield if we haven't seen this exact content recently
                                if content_hash != last_content_hash:
                                    last_content_hash = content_hash
                                    yield normalized_event
                        else:
                            # Non-text events, yield as-is (tool calls, etc.)
                            yield normalized_event
            except Exception as agent_error:
                # If streaming fails, try to recover
                logger.error(f"Streaming agent execution failed: {agent_error}")
                
                # Check if it's a tool_use/tool_result error
                if "tool_use" in str(agent_error) and "tool_result" in str(agent_error):
                    logger.info("Detected tool_use/tool_result mismatch, reinitializing agent...")
                    
                    # Reinitialize the agent to clear conversation state
                    agent_instance = None
                    agent = initialize_agent()
                    
                    # Try again with fresh agent
                    stream = agent.stream_async(user_message)
                    seen_content = set()  # Reset deduplication for retry
                    last_content_hash = None
                    
                    async for event in stream:
                        normalized_event = _normalize_streaming_event(event)
                        if normalized_event:
                            # Deduplicate text content
                            if normalized_event.get("type") == "content_delta":
                                current_text = normalized_event.get("text", "")
                                if current_text:
                                    # Create a hash of the content for deduplication
                                    content_hash = hash(current_text)
                                    
                                    # Only yield if we haven't seen this exact content recently
                                    if content_hash != last_content_hash:
                                        last_content_hash = content_hash
                                        yield normalized_event
                            else:
                                # Non-text events, yield as-is
                                yield normalized_event
                else:
                    # Yield error response for streaming
                    yield {
                        "result": f"Error: {str(agent_error)}. Please try again or check your configuration.",
                        "status": "error",
                        "error": str(agent_error)
                    }
        else:
            # Non-streaming response (synchronous)
            try:
                result = agent(user_message)
                
                # Return the response in the expected format
                response = {
                    "result": result.message,
                    "status": "success"
                }
            except Exception as agent_error:
                # If agent execution fails, try to recover
                logger.error(f"Agent execution failed: {agent_error}")
                
                # Check if it's a tool_use/tool_result error
                if "tool_use" in str(agent_error) and "tool_result" in str(agent_error):
                    logger.info("Detected tool_use/tool_result mismatch, reinitializing agent...")
                    
                    # Reinitialize the agent to clear conversation state
                    agent_instance = None
                    agent = initialize_agent()
                    
                    # Try again with fresh agent
                    result = agent(user_message)
                    response = {
                        "result": result.message,
                        "status": "success"
                    }
                else:
                    # Re-raise other errors
                    raise agent_error
        
            # Include uploaded image paths in response if available (non-streaming only)
            if uploaded_image_paths:
                if len(uploaded_image_paths) == 1:
                    response["uploaded_image_path"] = uploaded_image_paths[0]
                else:
                    response["uploaded_image_paths"] = uploaded_image_paths
                    response["uploaded_image_count"] = len(uploaded_image_paths)
            
            yield response
        
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        # For streaming, yield the error; for non-streaming, return it
        error_response = {
            "result": f"Error: {str(e)}. Please try again or check your configuration.",
            "status": "error",
            "error": str(e)
        }
        
        # Check if we're in streaming mode
        enable_streaming = payload.get("stream", True)
        if enable_streaming:
            yield error_response
        else:
            yield error_response


def _save_uploaded_image(image_data: str, filename: str = "uploaded_image") -> Optional[str]:
    """
    Save uploaded image data to a temporary location.
    
    Args:
        image_data: Base64 encoded image data
        filename: Filename for the image (without extension)
        
    Returns:
        Path to saved image file, or None if failed
    """
    try:
        import base64
        import uuid
        from datetime import datetime
        
        if not image_data:
            return None
        
        # Handle different image data formats
        if isinstance(image_data, str):
            # Assume base64 encoded
            try:
                # Remove data URL prefix if present (data:image/jpeg;base64,...)
                if image_data.startswith('data:'):
                    image_data = image_data.split(',', 1)[1]
                
                decoded_image = base64.b64decode(image_data)
            except Exception as e:
                logger.error(f"Failed to decode base64 image: {e}")
                return None
        else:
            logger.error("Unsupported image data format")
            return None
        
        # Generate unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        
        # Remove extension from filename if present, we'll add .png
        if '.' in filename:
            filename = filename.rsplit('.', 1)[0]
        
        # Create full filename
        full_filename = f"{timestamp}_{unique_id}_{filename}.png"
        
        # Save to temporary directory
        import os
        temp_dir = "/tmp/uploaded_images"
        os.makedirs(temp_dir, exist_ok=True)
        
        image_path = os.path.join(temp_dir, full_filename)
        
        # Save image
        with open(image_path, "wb") as f:
            f.write(decoded_image)
        
        logger.info(f"Uploaded image saved: {image_path} ({len(decoded_image)} bytes)")
        return image_path
        
    except Exception as e:
        logger.error(f"Error saving uploaded image: {e}")
        return None


def cleanup_resources():
    """Cleanup resources when the agent shuts down."""
    global mcp_client
    if mcp_client:
        try:
            mcp_client.__exit__(None, None, None)
            logger.info("MCP client cleaned up successfully")
        except Exception as e:
            logger.error(f"Error cleaning up MCP client: {e}")

# Register cleanup function to be called on shutdown
import atexit
atexit.register(cleanup_resources)


def health_check():
    """Health check function for AgentCore Runtime."""
    try:
        # Try to initialize agent to verify everything is working
        agent = initialize_agent()
        return {"status": "healthy", "agent_ready": agent is not None}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}


if __name__ == "__main__":
    # For local testing
    app.run()