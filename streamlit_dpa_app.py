#!/usr/bin/env python3
"""
Streamlit Frontend for DPA Agent - AgentCore Runtime
Dynamic Product Advertising with Amazon Nova Models
"""

import streamlit as st
import base64
import json
import os
import uuid
from PIL import Image
from io import BytesIO
import time
import re
import boto3
import subprocess

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Configuration
AGENTCORE_TIMEOUT = int(os.getenv("AGENTCORE_TIMEOUT", "120"))  # 2 minutes default

# Page configuration
st.set_page_config(
    page_title="DPA Agent - Nova Models",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #FF9900;
        text-align: center;
        margin-bottom: 1rem;
    }
    .subtitle {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 10px;
    }
    .user-message {
        background: #f0f2f6;
        border-left: 4px solid #FF9900;
    }
    .assistant-message {
        background: #e8f4fd;
        border-left: 4px solid #0073e6;
    }
    .stButton > button {
        width: 100%;
        background: #FF9900;
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        font-weight: bold;
    }
    .stButton > button:hover {
        background: #e68a00;
    }
</style>
""", unsafe_allow_html=True)

import tempfile

class AgentCoreClient:
    def __init__(self):
        """Initialize AgentCore client"""
        # Get agent ARN from environment or use default
        self.agent_arn = os.getenv(
            "AGENTCORE_AGENT_ARN", 
            "arn:aws:bedrock-agentcore:us-west-2:322216473749:runtime/dpa_agent-D8Xdy45Kiy"
        )
        
        # Choose invocation method: 'cli' or 'boto3'
        self.method = os.getenv("AGENTCORE_METHOD", "cli")
        
        if self.method == "boto3":
            # Initialize boto3 client for bedrock-agentcore
            self.boto_client = boto3.client('bedrock-agentcore')
        
    def invoke_agent(self, prompt, images=None):
        """Invoke the AgentCore runtime"""
        print('invoke')
        try:

            
            # Prepare the request payload
            request_data = {"prompt": prompt}
            
            # Enable streaming by default (can be controlled via session state)
            enable_streaming = st.session_state.get('enable_streaming', True)
            request_data["stream"] = enable_streaming
            
            # Handle single image (backward compatibility)
            if images and len(images) == 1:
                request_data["image"] = images[0]
                request_data["image_filename"] = "uploaded_image.png"

            
            # Handle multiple images
            elif images and len(images) > 1:
                # Format for multiple images
                images_list = []
                for i, img_data in enumerate(images):
                    images_list.append({
                        "data": img_data,
                        "filename": f"uploaded_image_{i+1}.png"
                    })
                request_data["images"] = images_list
            if self.method == "boto3":
                
                return self._invoke_with_boto3(request_data)
            else:
                return self._invoke_with_cli(request_data)
                
        except Exception as e:
            error_msg = f"Error invoking agent: {str(e)}"
            st.error(error_msg)
            return error_msg

    def invoke_agent_with_streaming(self, prompt, images=None):
        """Invoke the AgentCore runtime with real-time streaming display in chat"""
        try:
            # Create a placeholder in the chat area for streaming response
            with st.container():
                assistant_placeholder = st.empty()
                
                # Show initial processing message
                assistant_placeholder.markdown(f'<div class="chat-message assistant-message"><strong>DPA Agent:</strong> 🤖 Processing your request...</div>', unsafe_allow_html=True)
                
                # Prepare the request payload
                request_data = {"prompt": prompt}
                
                # Enable streaming by default (can be controlled via session state)
                enable_streaming = st.session_state.get('enable_streaming', True)
                request_data["stream"] = enable_streaming
                
                # Handle single image (backward compatibility)
                if images and len(images) == 1:
                    request_data["image"] = images[0]
                    request_data["image_filename"] = "uploaded_image.png"
                
                # Handle multiple images
                elif images and len(images) > 1:
                    # Format for multiple images
                    images_list = []
                    for i, img_data in enumerate(images):
                        images_list.append({
                            "data": img_data,
                            "filename": f"uploaded_image_{i+1}.png"
                        })
                    request_data["images"] = images_list
                
                if self.method == "boto3":
                    return self._invoke_with_boto3_streaming(request_data, assistant_placeholder)
                else:
                    return self._invoke_with_cli_streaming(request_data, assistant_placeholder)
                
        except Exception as e:
            error_msg = f"Error invoking agent: {str(e)}"
            st.error(error_msg)
            return error_msg
    
    def _invoke_with_boto3(self, request_data):
        """Invoke using boto3 client with streaming support"""
        try:
            # Generate session ID for this request
            session_id = str(uuid.uuid4())
            
            # Prepare payload as binary data
            payload = json.dumps(request_data).encode()
            

            
            # Invoke the agent runtime
            response = self.boto_client.invoke_agent_runtime(
                agentRuntimeArn=self.agent_arn,
                runtimeSessionId=session_id,
                payload=payload
            )
            
            # Check if streaming is enabled in session state
            enable_streaming = st.session_state.get('enable_streaming', True)
            
            if enable_streaming:
                return self._handle_streaming_response(response)
            else:
                return self._handle_standard_response(response)

                
        except Exception as e:
            # Handle specific AgentCore errors
            error_msg = str(e)
            if "ValidationException" in error_msg:
                raise Exception(f"Invalid request parameters: {e}")
            elif "ResourceNotFoundException" in error_msg:
                raise Exception(f"Agent runtime not found. Check your ARN: {e}")
            elif "AccessDeniedException" in error_msg:
                raise Exception(f"Permission denied. Check IAM permissions for bedrock-agentcore:InvokeAgentRuntime: {e}")
            elif "ThrottlingException" in error_msg:
                raise Exception(f"Request rate limit exceeded. Please try again: {e}")
            else:
                raise Exception(f"Boto3 invocation failed: {e}")
    
    def _handle_streaming_response(self, response):
        """Handle real-time streaming response from AgentCore with normalized events"""
        try:
            content_type = response.get("contentType", "")
            response_stream = response.get("response")
            
            # Create placeholder for streaming content
            streaming_container = st.container()
            streaming_placeholder = streaming_container.empty()
            
            content_parts = []
            tool_calls = []
            
            # Handle streaming response
            if "text/event-stream" in content_type and hasattr(response_stream, 'iter_lines'):
                try:
                    for line in response_stream.iter_lines(chunk_size=10):
                        if line:
                            line_text = line.decode("utf-8")
                            
                            # Process streaming line
                            if line_text.startswith("data: "):
                                data_content = line_text[6:]  # Remove "data: " prefix
                                
                                # Try to parse as JSON and handle normalized events
                                try:
                                    event_data = json.loads(data_content)
                                    processed_content = self._process_streaming_event(event_data)
                                    
                                    if processed_content:
                                        if processed_content["type"] == "text":
                                            content_parts.append(processed_content["content"])
                                            
                                            # Update streaming display in real-time
                                            current_content = "".join(content_parts)
                                            streaming_placeholder.markdown(f"**🔄 Streaming Response:**\n\n{current_content}")
                                        
                                        elif processed_content["type"] == "tool_call":
                                            tool_calls.append(processed_content["content"])
                                            # Optionally show tool calls in real-time
                                            tool_info = f"\n\n*🔧 Using tool: {processed_content['content']}*"
                                            current_content = "".join(content_parts) + tool_info
                                            streaming_placeholder.markdown(f"**🔄 Streaming Response:**\n\n{current_content}")
                                        
                                except json.JSONDecodeError:
                                    # Fallback to old parsing method
                                    chunk_text = self._extract_agent_message_fallback(data_content)
                                    if chunk_text:
                                        content_parts.append(chunk_text)
                                        current_content = "".join(content_parts)
                                        streaming_placeholder.markdown(f"**🔄 Streaming Response:**\n\n{current_content}")
                                
                except Exception as e:
                    pass  # Continue with fallback
            
            # If no streaming content or fallback needed
            if not content_parts and hasattr(response_stream, 'read'):
                stream_content = response_stream.read().decode('utf-8')
                
                # Try to parse the response as JSON and extract the message
                try:
                    parsed_data = json.loads(stream_content)
                    parsed_content = self._extract_agent_message(parsed_data)
                    content_parts.append(parsed_content)
                except json.JSONDecodeError:
                    # If not JSON, use raw content
                    content_parts.append(stream_content)
            
            # Clear streaming placeholder and return final result
            streaming_placeholder.empty()
            final_result = "".join(content_parts) if content_parts else "No response content"
            
            return final_result
            
        except Exception as e:
            raise Exception(f"Streaming response error: {e}")
    
    def _handle_standard_response(self, response):
        """Handle standard (non-streaming) response from AgentCore"""
        try:
            response_stream = response.get("response")
            
            if hasattr(response_stream, 'read'):
                response_data = response_stream.read().decode('utf-8')
                
                try:
                    result = json.loads(response_data)
                    # Parse AgentCore response format
                    return self._extract_agent_message(result)
                        
                except json.JSONDecodeError:
                    return response_data
            else:
                return str(response_stream)
                
        except Exception as e:
            raise Exception(f"Standard response error: {e}")
    
    def _process_streaming_event(self, event_data):
        """Process normalized streaming events from the agent"""
        try:
            # Handle normalized event format from agent
            if isinstance(event_data, dict):
                event_type = event_data.get("type")
                
                if event_type == "content_delta":
                    # Text content from the agent
                    text_content = event_data.get("text", "")
                    if text_content:
                        return {
                            "type": "text",
                            "content": text_content
                        }
                
                elif event_type == "other":
                    # Tool calls or other events
                    raw_event = event_data.get("raw_event", {})
                    
                    # Try to extract tool call information
                    if isinstance(raw_event, dict):
                        # Look for tool use patterns
                        if "tool_use" in str(raw_event).lower():
                            tool_name = self._extract_tool_name(raw_event)
                            return {
                                "type": "tool_call",
                                "content": tool_name or "Unknown tool"
                            }
                
                elif event_type == "raw":
                    # Fallback to old parsing for raw events
                    raw_event = event_data.get("raw_event")
                    if raw_event:
                        parsed_text = self._extract_agent_message_fallback(raw_event)
                        if parsed_text:
                            return {
                                "type": "text",
                                "content": parsed_text
                            }
            
            return None
            
        except Exception as e:
            return None
    
    def _extract_tool_name(self, raw_event):
        """Extract tool name from raw event data"""
        try:
            # Look for common tool name patterns
            if isinstance(raw_event, dict):
                # Check various possible locations for tool names
                if "name" in raw_event:
                    return raw_event["name"]
                elif "tool_name" in raw_event:
                    return raw_event["tool_name"]
                elif "function" in raw_event and isinstance(raw_event["function"], dict):
                    return raw_event["function"].get("name")
                
                # Look deeper in nested structures
                for key, value in raw_event.items():
                    if isinstance(value, dict) and "name" in value:
                        return value["name"]
            
            return None
        except:
            return None
    
    def _extract_agent_message_fallback(self, response_data):
        """Fallback method for extracting agent messages from various formats"""
        try:
            if isinstance(response_data, str):
                return response_data
            elif isinstance(response_data, dict):
                # Try various common fields
                for field in ["text", "content", "message", "data"]:
                    if field in response_data:
                        return str(response_data[field])
            return str(response_data)
        except:
            return str(response_data)

    def _extract_agent_message(self, response_data):
        """Extract the actual agent message from AgentCore response format"""
        try:
            # Handle the specific AgentCore format: {"result": {"role": "assistant", "content": [{"text": "..."}]}, "status": "success"}
            if isinstance(response_data, dict) and 'result' in response_data:
                result = response_data['result']
                
                # Check if it's the Strands agent format with content array
                if isinstance(result, dict) and 'content' in result:
                    content = result['content']
                    
                    # Extract text from content array
                    if isinstance(content, list) and len(content) > 0:
                        text_parts = []
                        for item in content:
                            if isinstance(item, dict) and 'text' in item:
                                text_parts.append(item['text'])
                            else:
                                text_parts.append(str(item))
                        return '\n'.join(text_parts)
                    
                    # Handle direct text content
                    elif isinstance(content, str):
                        return content
                
                # Handle direct result text
                elif isinstance(result, str):
                    return result
            
            # Handle direct text response
            elif isinstance(response_data, dict) and 'text' in response_data:
                return response_data['text']
            
            # Handle message field
            elif isinstance(response_data, dict) and 'message' in response_data:
                return response_data['message']
            
            # Handle response field
            elif isinstance(response_data, dict) and 'response' in response_data:
                return str(response_data['response'])
            
            # If it's already a string, return as-is
            elif isinstance(response_data, str):
                return response_data
            
            # Fallback: return string representation
            else:
                return str(response_data)
                
        except Exception as e:
            # If all parsing fails, return the raw data as string
            return str(response_data)
    
    def _invoke_with_cli(self, request_data):
        """Invoke using AgentCore CLI"""
        try:
            # Write request to temp file to avoid command line length limits
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
                json.dump(request_data, tmp_file)
                request_file = tmp_file.name
            
            # Use agentcore CLI with file input
            cmd = ['agentcore', 'invoke', f'@{request_file}']
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=AGENTCORE_TIMEOUT
            )
            
            # Cleanup temp file
            os.unlink(request_file)
            
            if result.returncode == 0:
                # Parse the CLI output to extract the response
                output = result.stdout
                
                # Try to extract JSON response
                try:
                    # Look for JSON in the output
                    lines = output.split('\n')
                    
                    for line in lines:
                        if line.strip().startswith('{'):
                            try:
                                response_data = json.loads(line.strip())
                                # Use the same extraction method for consistency
                                extracted_message = self._extract_agent_message(response_data)
                                return extracted_message
                                
                            except json.JSONDecodeError:
                                continue
                    
                    # If no JSON found, return raw output
                    return output
                    
                except json.JSONDecodeError:
                    return output
            else:
                error_output = result.stderr
                raise Exception(f"CLI invocation failed: {error_output}")
                
        except subprocess.TimeoutExpired:
            raise Exception("Request timed out")
        except Exception as e:
            raise Exception(f"CLI invocation error: {e}")

    def _invoke_with_boto3_streaming(self, request_data, placeholder):
        """Invoke using boto3 client with real-time streaming display in chat"""
        try:
            # Generate session ID for this request
            session_id = str(uuid.uuid4())
            
            # Prepare payload as binary data
            payload = json.dumps(request_data).encode()
            
            # Invoke the agent runtime
            response = self.boto_client.invoke_agent_runtime(
                agentRuntimeArn=self.agent_arn,
                runtimeSessionId=session_id,
                payload=payload
            )
            
            # Handle streaming response with real-time display
            return self._handle_streaming_response_in_chat(response, placeholder)
                
        except Exception as e:
            # Handle specific AgentCore errors
            error_msg = str(e)
            if "ValidationException" in error_msg:
                raise Exception(f"Invalid request parameters: {e}")
            elif "ResourceNotFoundException" in error_msg:
                raise Exception(f"Agent runtime not found. Check your ARN: {e}")
            elif "AccessDeniedException" in error_msg:
                raise Exception(f"Permission denied. Check IAM permissions for bedrock-agentcore:InvokeAgentRuntime: {e}")
            elif "ThrottlingException" in error_msg:
                raise Exception(f"Request rate limit exceeded. Please try again: {e}")
            else:
                raise Exception(f"Boto3 invocation failed: {e}")

    def _invoke_with_cli_streaming(self, request_data, placeholder):
        """Invoke using AgentCore CLI with real-time streaming display in chat"""
        try:
            # Write request to temp file to avoid command line length limits
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
                json.dump(request_data, tmp_file)
                request_file = tmp_file.name
            
            # Use agentcore CLI with file input
            cmd = ['agentcore', 'invoke', f'@{request_file}']
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=AGENTCORE_TIMEOUT
            )
            
            # Cleanup temp file
            os.unlink(request_file)
            
            if result.returncode == 0:
                # Parse the CLI output and display in chat
                output = result.stdout
                
                # Try to extract JSON response
                try:
                    lines = output.split('\n')
                    
                    for line in lines:
                        if line.strip().startswith('{'):
                            try:
                                response_data = json.loads(line.strip())
                                extracted_message = self._extract_agent_message(response_data)
                                
                                # Display final response in chat
                                if placeholder and extracted_message:
                                    placeholder.markdown(f'<div class="chat-message assistant-message"><strong>DPA Agent:</strong> {extracted_message}</div>', unsafe_allow_html=True)
                                
                                return extracted_message
                                
                            except json.JSONDecodeError:
                                continue
                    
                    # If no JSON found, display raw output
                    if placeholder:
                        placeholder.markdown(f'<div class="chat-message assistant-message"><strong>DPA Agent:</strong> {output}</div>', unsafe_allow_html=True)
                    return output
                    
                except json.JSONDecodeError:
                    if placeholder:
                        placeholder.markdown(f'<div class="chat-message assistant-message"><strong>DPA Agent:</strong> {output}</div>', unsafe_allow_html=True)
                    return output
            else:
                error_output = result.stderr
                raise Exception(f"CLI invocation failed: {error_output}")
                
        except subprocess.TimeoutExpired:
            raise Exception("Request timed out")
        except Exception as e:
            raise Exception(f"CLI invocation error: {e}")

    def _handle_streaming_response_in_chat(self, response, placeholder):
        """Handle real-time streaming response with display in chat area"""
        try:
            content_type = response.get("contentType", "")
            response_stream = response.get("response")
            
            content_parts = []
            tool_calls = []
            
            # Handle streaming response
            if "text/event-stream" in content_type and hasattr(response_stream, 'iter_lines'):
                try:
                    for line in response_stream.iter_lines(chunk_size=10):
                        if line:
                            line_text = line.decode("utf-8")
                            
                            # Process streaming line
                            if line_text.startswith("data: "):
                                data_content = line_text[6:]  # Remove "data: " prefix
                                
                                # Try to parse as JSON and handle normalized events
                                try:
                                    event_data = json.loads(data_content)
                                    processed_content = self._process_streaming_event(event_data)
                                    
                                    if processed_content:
                                        if processed_content["type"] == "text":
                                            content_parts.append(processed_content["content"])
                                            
                                            # Update chat display in real-time
                                            current_content = "".join(content_parts)
                                            if placeholder:
                                                placeholder.markdown(f'<div class="chat-message assistant-message"><strong>DPA Agent:</strong> {current_content}</div>', unsafe_allow_html=True)
                                        
                                        elif processed_content["type"] == "tool_call":
                                            tool_calls.append(processed_content["content"])
                                            # Show tool usage in chat
                                            tool_info = f"\n\n*🔧 Using tool: {processed_content['content']}*"
                                            current_content = "".join(content_parts) + tool_info
                                            if placeholder:
                                                placeholder.markdown(f'<div class="chat-message assistant-message"><strong>DPA Agent:</strong> {current_content}</div>', unsafe_allow_html=True)
                                        
                                except json.JSONDecodeError:
                                    # Fallback to old parsing method
                                    chunk_text = self._extract_agent_message_fallback(data_content)
                                    if chunk_text:
                                        content_parts.append(chunk_text)
                                        current_content = "".join(content_parts)
                                        if placeholder:
                                            placeholder.markdown(f'<div class="chat-message assistant-message"><strong>DPA Agent:</strong> {current_content}</div>', unsafe_allow_html=True)
                                
                except Exception as e:
                    pass  # Continue with fallback
            
            # If no streaming content or fallback needed
            if not content_parts and hasattr(response_stream, 'read'):
                stream_content = response_stream.read().decode('utf-8')
                
                # Try to parse the response as JSON and extract the message
                try:
                    parsed_data = json.loads(stream_content)
                    parsed_content = self._extract_agent_message(parsed_data)
                    content_parts.append(parsed_content)
                except json.JSONDecodeError:
                    # If not JSON, use raw content
                    content_parts.append(stream_content)
                
                # Display final content
                final_result = "".join(content_parts) if content_parts else "No response content"
                if placeholder:
                    placeholder.markdown(f'<div class="chat-message assistant-message"><strong>DPA Agent:</strong> {final_result}</div>', unsafe_allow_html=True)
            
            final_result = "".join(content_parts) if content_parts else "No response content"
            return final_result
            
        except Exception as e:
            raise Exception(f"Streaming response error: {e}")

def encode_image_to_base64(image):
    """Convert PIL Image to base64 string"""
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return img_str

def extract_image_urls(text):
    """Extract image URLs from response text"""
    # Look for S3 URLs or local paths
    url_patterns = [
        r'https://[^\s]+\.png',
        r'https://[^\s]+\.jpg',
        r'https://[^\s]+\.jpeg',
        r'/tmp/[^\s]+\.png',
        r'/tmp/[^\s]+\.jpg'
    ]
    
    urls = []
    for pattern in url_patterns:
        matches = re.findall(pattern, text)
        urls.extend(matches)
    
    return urls

def main():
    # Initialize AgentCore client
    if 'agent_client' not in st.session_state:
        try:
            st.session_state.agent_client = AgentCoreClient()
            st.session_state.client_initialized = True
        except Exception as e:
            st.error(f"Failed to initialize AgentCore client: {e}")
            st.session_state.client_initialized = False
            return
    
    # Initialize chat history
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Header
    st.markdown('<div class="main-header">🎨 DPA Agent - Nova Models</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Dynamic Product Advertising with Amazon Nova Canvas, Reel & Pro</div>', unsafe_allow_html=True)
    
    # Sidebar with examples and configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Show current agent ARN
        current_arn = st.session_state.agent_client.agent_arn
        st.text_input(
            "Agent ARN:", 
            value=current_arn,
            disabled=True,
            help="Set AGENTCORE_AGENT_ARN environment variable to change"
        )
        
        # Show invocation method
        method = st.session_state.agent_client.method
        st.text_input(
            "Invocation Method:",
            value=method,
            disabled=True,
            help="Set AGENTCORE_METHOD=boto3 or cli (default: cli)"
        )
        
        # Connection status
        if st.session_state.get('client_initialized', False):
            st.success("✅ AgentCore client connected")
        else:
            st.error("❌ AgentCore client not connected")
        
        # Streaming toggle
        streaming_enabled = st.checkbox("Enable Real-time Streaming", value=True, 
                                       help="Show response as it streams in real-time")
        st.session_state.enable_streaming = streaming_enabled
        

    
    # Main chat interface
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Display chat messages
        for message in st.session_state.messages:
            with st.container():
                if message["role"] == "user":
                    st.markdown(f'<div class="chat-message user-message"><strong>You:</strong> {message["content"]}</div>', unsafe_allow_html=True)
                    
                    # Display uploaded images
                    if "images" in message:
                        cols = st.columns(min(len(message["images"]), 3))
                        for i, img_data in enumerate(message["images"]):
                            with cols[i % 3]:
                                img = Image.open(BytesIO(base64.b64decode(img_data)))
                                st.image(img, caption=f"Uploaded Image {i+1}", use_column_width=True)
                
                else:  # assistant
                    st.markdown(f'<div class="chat-message assistant-message"><strong>DPA Agent:</strong> {message["content"]}</div>', unsafe_allow_html=True)
                    
                    # Extract and display any generated images
                    image_urls = extract_image_urls(message["content"])
                    if image_urls:
                        st.write("**Generated Images:**")
                        cols = st.columns(min(len(image_urls), 2))
                        for i, url in enumerate(image_urls):
                            with cols[i % 2]:
                                if url.startswith('http'):
                                    st.image(url, caption=f"Generated Image {i+1}")
                                else:
                                    st.write(f"Image saved to: `{url}`")
    
    with col2:
        st.header("🖼️ Upload Images")
        uploaded_files = st.file_uploader(
            "Upload product images",
            type=['png', 'jpg', 'jpeg'],
            accept_multiple_files=True,
            help="Upload images for analysis or editing"
        )
        
        if uploaded_files:
            st.write(f"**{len(uploaded_files)} image(s) uploaded**")
            for i, file in enumerate(uploaded_files):
                img = Image.open(file)
                st.image(img, caption=f"Image {i+1}", use_column_width=True)
    
    # Chat input
    st.markdown("---")
    
    # Use example prompt if selected
    default_prompt = ""
    if 'example_prompt' in st.session_state:
        default_prompt = st.session_state.example_prompt
        del st.session_state.example_prompt
    
    user_input = st.text_area(
        "Enter your message:",
        value=default_prompt,
        height=100,
        placeholder="Describe what you want to create or ask about your images..."
    )
    
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        send_button = st.button("🚀 Send", type="primary")
    
    with col2:
        clear_button = st.button("🗑️ Clear Chat")
    
    if clear_button:
        st.session_state.messages = []
        st.rerun()
    
    if send_button and user_input.strip():
        # Check if client is initialized
        if not st.session_state.get('client_initialized', False):
            st.error("AgentCore client not initialized. Please check your configuration.")
            return
        
        # Prepare images
        images_data = []
        if uploaded_files:
            for file in uploaded_files:
                img = Image.open(file)
                img_base64 = encode_image_to_base64(img)
                images_data.append(img_base64)
        
        # Add user message
        user_message = {"role": "user", "content": user_input}
        if images_data:
            user_message["images"] = images_data
        st.session_state.messages.append(user_message)
        
        # Add user message to history
        st.session_state.messages.append(user_message)
        
        # Show processing with streaming in the chat area
        try:
            # Invoke agent with streaming display in chat
            response = st.session_state.agent_client.invoke_agent_with_streaming(user_input, images_data)
            
            # Add final assistant response to messages
            st.session_state.messages.append({
                "role": "assistant", 
                "content": response
            })
            
        except Exception as e:
            error_msg = f"Error processing request: {str(e)}"
            st.error(error_msg)
            st.session_state.messages.append({
                "role": "assistant",
                "content": error_msg
            })
        
        st.rerun()

if __name__ == "__main__":
    main()
