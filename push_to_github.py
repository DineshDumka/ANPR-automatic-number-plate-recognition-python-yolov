#!/usr/bin/env python3
"""
Script to push changes to the GitHub repository.
This script automates the process of committing and pushing changes to the GitHub repository.
"""

import os
import subprocess
import argparse
import sys

def run_command(command):
    """Run a shell command and return the output"""
    try:
        result = subprocess.run(command, shell=True, check=True, text=True, capture_output=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {command}")
        print(f"Error message: {e.stderr}")
        return None

def setup_git_config(name, email):
    """Set up Git configuration with name and email"""
    run_command(f'git config --local user.name "{name}"')
    run_command(f'git config --local user.email "{email}"')
    print(f"Git configured with name: {name} and email: {email}")

def check_git_status():
    """Check Git status and return modified files"""
    status = run_command('git status --porcelain')
    if not status:
        print("No changes to commit.")
        return []
    
    files = status.split('\n')
    modified_files = []
    
    for file in files:
        if file.strip():
            status_code = file[:2].strip()
            filename = file[3:].strip()
            modified_files.append((status_code, filename))
    
    return modified_files

def commit_changes(message):
    """Commit changes with the given message"""
    run_command('git add .')
    result = run_command(f'git commit -m "{message}"')
    if result:
        print(f"Changes committed: {result}")
        return True
    return False

def push_changes(branch="main"):
    """Push changes to the remote repository"""
    result = run_command(f'git push origin {branch}')
    if result is not None:
        print(f"Changes pushed to {branch} branch")
        return True
    return False

def set_remote_url(repo_url):
    """Set or update the remote repository URL"""
    # Check if remote already exists
    remote_check = run_command('git remote -v')
    
    if remote_check and 'origin' in remote_check:
        # Update existing remote
        run_command(f'git remote set-url origin {repo_url}')
        print(f"Updated remote URL to: {repo_url}")
    else:
        # Add new remote
        run_command(f'git remote add origin {repo_url}')
        print(f"Added remote URL: {repo_url}")

def initialize_repo_if_needed():
    """Initialize Git repository if not already initialized"""
    if not os.path.exists('.git'):
        run_command('git init')
        print("Initialized new Git repository")
        return True
    return False

def main():
    parser = argparse.ArgumentParser(description='Push changes to GitHub repository')
    parser.add_argument('--name', help='Git user name')
    parser.add_argument('--email', help='Git user email')
    parser.add_argument('--repo', help='GitHub repository URL')
    parser.add_argument('--message', '-m', default='Update ANPR system for Indian license plates', 
                        help='Commit message')
    parser.add_argument('--branch', default='main', help='Branch to push to')
    
    args = parser.parse_args()
    
    # Initialize repository if needed
    is_new_repo = initialize_repo_if_needed()
    
    # Set up Git configuration if provided
    if args.name and args.email:
        setup_git_config(args.name, args.email)
    
    # Set remote URL if provided
    if args.repo:
        set_remote_url(args.repo)
    
    # Check Git status
    modified_files = check_git_status()
    
    if not modified_files:
        print("No changes to commit. Exiting.")
        sys.exit(0)
    
    print("Modified files:")
    for status, filename in modified_files:
        print(f"  {status} {filename}")
    
    # Commit changes
    if commit_changes(args.message):
        # Push changes
        if args.branch:
            push_changes(args.branch)
        else:
            push_changes()
    else:
        print("Failed to commit changes. Exiting.")
        sys.exit(1)

if __name__ == "__main__":
    main() 