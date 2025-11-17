#!/usr/bin/env python3
"""
Comprehensive fix for Company Name issue
This script will:
1. Find where "Red6-ES" is coming from
2. Update all config files to use the correct company name
3. Rebuild the Docker container
"""

import os
import subprocess
import yaml
import json

def run_command(cmd, description=""):
    """Run a shell command and return output"""
    if description:
        print(f"\n{description}")
        print("-" * 60)
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        output = result.stdout + result.stderr
        print(output)
        return output
    except Exception as e:
        print(f"Error: {e}")
        return None

def main():
    print("=" * 80)
    print("COMPANY NAME DIAGNOSTIC AND FIX")
    print("=" * 80)
    
    # Step 1: Check local config file
    print("\n1. LOCAL CONFIG FILE (config/config_names.yaml)")
    print("-" * 80)
    config_path = "config/config_names.yaml"
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                content = f.read()
                print(f"File exists. Content:\n{content}")
                config = yaml.safe_load(content)
                print(f"\nParsed company_name: {config.get('company_name', 'NOT FOUND')}")
        except Exception as e:
            print(f"Error reading file: {e}")
    else:
        print("❌ File does NOT exist!")
    
    # Step 2: Check container config
    print("\n2. CONTAINER CONFIG FILE")
    print("-" * 80)
    run_command(
        "docker exec sentiment-python-v2 cat config/config_names.yaml 2>&1",
        "Reading config from container:"
    )
    
    # Step 3: Check what Python sees in container
    print("\n3. PYTHON CONFIG LOADER IN CONTAINER")
    print("-" * 80)
    run_command(
        '''docker exec sentiment-python-v2 python3 -c "
from config.config import load_all_configs
config = load_all_configs()
print('Company name from names_config:', config.get('names_config', {}).get('company_name', 'NOT FOUND'))
" 2>&1''',
        "What load_all_configs() returns:"
    )
    
    # Step 4: Check API endpoint
    print("\n4. API /api/config ENDPOINT")
    print("-" * 80)
    run_command(
        "curl -s http://localhost:8001/api/config",
        "What the API returns:"
    )
    
    # Step 5: Offer to fix
    print("\n5. FIX THE ISSUE")
    print("-" * 80)
    print("\nCurrent situation analysis:")
    print("- If you see 'Red6-ES' above, it's cached in the Docker image")
    print("- The config file needs to be updated and container rebuilt")
    
    company_name = input("\nEnter the correct company name (or press Enter to skip): ").strip()
    
    if company_name:
        # Update local config file
        print(f"\n✏️  Updating config/config_names.yaml with: {company_name}")
        config_content = f"""# Company branding configuration
# This file is used for PDF report generation and email notifications

# Company name to display in reports
company_name: "{company_name}"

# Other company-specific configurations can be added here
"""
        os.makedirs("config", exist_ok=True)
        with open(config_path, 'w') as f:
            f.write(config_content)
        print("✅ Config file updated")
        
        # Rebuild container
        print("\n🔨 Rebuilding Docker container...")
        print("This will take a few minutes...\n")
        
        commands = [
            "docker-compose stop python-service",
            "docker-compose rm -f python-service",
            "docker-compose build --no-cache python-service",
            "docker-compose up -d python-service"
        ]
        
        for cmd in commands:
            run_command(cmd)
        
        # Wait and verify
        print("\n⏳ Waiting for container to start...")
        run_command("sleep 10")
        
        print("\n6. VERIFICATION")
        print("-" * 80)
        run_command(
            "curl -s http://localhost:8001/api/config",
            "API /api/config now returns:"
        )
        
        print("\n✅ FIX COMPLETE!")
        print(f"Company name should now be: {company_name}")
    else:
        print("\n⏭️  Skipped fix")
    
    print("\n" + "=" * 80)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()
