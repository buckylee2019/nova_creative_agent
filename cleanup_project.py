#!/usr/bin/env python3
"""
Cleanup script to remove non-essential files for AgentCore deployment.

This script removes legacy files, redundant documentation, and temporary files
while keeping only the essential files needed for AgentCore deployment.
"""

import os
import shutil
from pathlib import Path

def cleanup_project():
    """Remove non-essential files for AgentCore deployment."""
    
    # Files to delete (legacy/redundant)
    files_to_delete = [
        # Legacy Strands agent files
        "config.py", 
        "README.md",  # Use README_AGENTCORE.md instead
        
        # Streamlit-related files (not needed for AgentCore)
        "run_app.py",
        "streamlit_config.toml", 
        "streamlit_dpa_app.py",
        
        # Testing files (can be deleted after successful deployment)
        "simple_agentcore_test.py",
        "test_agentcore_agent.py",
        "simple_requirements.txt",
        
        # Comparison documentation (reference only)
        "STRANDS_VS_AGENTCORE.md",
        
        # Auto-generated files (will be recreated)
        ".bedrock_agentcore.yaml",
        "Dockerfile",
        ".dockerignore",
    ]
    
    # Directories to delete
    dirs_to_delete = [
        "__pycache__",
        "source",  # If it exists and is empty/not needed
    ]
    
    print("🧹 Cleaning up project for AgentCore deployment...")
    print("=" * 50)
    
    deleted_files = []
    kept_files = []
    errors = []
    
    # Delete files
    for file_path in files_to_delete:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                deleted_files.append(file_path)
                print(f"🗑️  Deleted: {file_path}")
            except Exception as e:
                errors.append(f"Error deleting {file_path}: {e}")
                print(f"❌ Error deleting {file_path}: {e}")
        else:
            print(f"⚪ Not found: {file_path}")
    
    # Delete directories
    for dir_path in dirs_to_delete:
        if os.path.exists(dir_path) and os.path.isdir(dir_path):
            try:
                shutil.rmtree(dir_path)
                deleted_files.append(f"{dir_path}/")
                print(f"🗑️  Deleted directory: {dir_path}/")
            except Exception as e:
                errors.append(f"Error deleting directory {dir_path}: {e}")
                print(f"❌ Error deleting directory {dir_path}: {e}")
    
    print("\n" + "=" * 50)
    print("📊 Cleanup Summary:")
    print(f"✅ Files deleted: {len(deleted_files)}")
    print(f"❌ Errors: {len(errors)}")
    
    if deleted_files:
        print(f"\n🗑️  Deleted files:")
        for file in deleted_files:
            print(f"   • {file}")
    
    if errors:
        print(f"\n❌ Errors:")
        for error in errors:
            print(f"   • {error}")
    
    # Show remaining essential files
    essential_files = [
        "dpa_agent.py",
        "dpa_mcp_server.py", 
        "agentcore_requirements.txt",
        "__init__.py",
        ".env",
        "deploy_script.py",
        "README_AGENTCORE.md",
        "deploy_to_agentcore.md",
        "TROUBLESHOOTING.md"
    ]
    
    print(f"\n✅ Essential files remaining:")
    for file in essential_files:
        if os.path.exists(file):
            kept_files.append(file)
            print(f"   ✓ {file}")
        else:
            print(f"   ❌ MISSING: {file}")
    
    print(f"\n🎯 Project is now clean and ready for AgentCore deployment!")
    print(f"📁 Essential files: {len(kept_files)}")
    print(f"🚀 Next step: python deploy_script.py")

def show_cleanup_preview():
    """Show what would be deleted without actually deleting."""
    files_to_delete = [
        "dpa_agent.py", "config.py", "requirements.txt", "README.md",
        "run_app.py", "streamlit_config.toml", "streamlit_dpa_app.py",
        "simple_agentcore_test.py", "test_agentcore_agent.py", "simple_requirements.txt",
        "STRANDS_VS_AGENTCORE.md", ".bedrock_agentcore.yaml", "Dockerfile", ".dockerignore"
    ]
    
    dirs_to_delete = ["__pycache__", "source"]
    
    print("🔍 Cleanup Preview - Files that would be deleted:")
    print("=" * 50)
    
    for file_path in files_to_delete:
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            print(f"🗑️  {file_path} ({size} bytes)")
        else:
            print(f"⚪ {file_path} (not found)")
    
    for dir_path in dirs_to_delete:
        if os.path.exists(dir_path) and os.path.isdir(dir_path):
            print(f"🗑️  {dir_path}/ (directory)")
        else:
            print(f"⚪ {dir_path}/ (not found)")
    
    print("\n✅ Files that would be kept:")
    essential_files = [
        "agentcore_dpa_agent.py", "dpa_mcp_server.py", "agentcore_requirements.txt",
        "__init__.py", ".env", "deploy_script.py", "README_AGENTCORE.md", 
        "deploy_to_agentcore.md", "TROUBLESHOOTING.md"
    ]
    
    for file in essential_files:
        if os.path.exists(file):
            print(f"   ✓ {file}")
        else:
            print(f"   ❌ MISSING: {file}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--preview":
        show_cleanup_preview()
    elif len(sys.argv) > 1 and sys.argv[1] == "--help":
        print("Usage:")
        print("  python cleanup_project.py           # Perform cleanup")
        print("  python cleanup_project.py --preview # Show what would be deleted")
        print("  python cleanup_project.py --help    # Show this help")
    else:
        print("⚠️  This will delete files permanently!")
        response = input("Continue with cleanup? (y/N): ").strip().lower()
        if response in ['y', 'yes']:
            cleanup_project()
        else:
            print("Cleanup cancelled.")
            print("Run with --preview to see what would be deleted.")