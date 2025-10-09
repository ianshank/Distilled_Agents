#!/usr/bin/env python3
"""
Simple script to run MangoMAS SageMaker training jobs
Usage: python run_sagemaker_training.py
"""

import os
import sys
import asyncio
from pathlib import Path

# Add training directory to path
sys.path.append(str(Path(__file__).parent))

from scripts.deployment.launch_all_agents_sagemaker import MangoMASSageMakerLauncher

def setup_environment():
    """Setup environment variables and AWS configuration"""
    print("🔧 Setting up environment...")
    
    # Set default AWS region if not set
    if not os.getenv('AWS_REGION'):
        os.environ['AWS_REGION'] = 'us-east-1'
        print(f"✅ Set AWS_REGION to: {os.environ['AWS_REGION']}")
    
    # Check AWS authentication
    try:
        import boto3
        sts = boto3.client('sts')
        identity = sts.get_caller_identity()
        print(f"✅ AWS authentication valid for account: {identity['Account']}")
    except Exception as e:
        print(f"❌ AWS authentication error: {e}")
        print("Please configure AWS authentication using:")
        print("  aws configure")
        print("  or set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables")
        return False
    
    return True

def validate_training_files():
    """Validate that all required training files exist"""
    print("🔍 Validating training files...")
    
    data_files = [
        "product_manager_agent_real_data.jsonl",
        "sqe_agent_real_data.jsonl",
        "architect_agent_real_data.jsonl",
        "swe_agent_real_data.jsonl",
        "product_manager_agent.jsonl",
        "sqe_agent.jsonl",
        "architect_agent.jsonl",
        "swe_agent.jsonl",
        "vp_product_agent.jsonl",
        "devops_agent.jsonl",
        "tools_agent.jsonl"
    ]

    script_files = [
        "train_distilled_adapter.py"
    ]

    missing_files = []

    # Check data files
    for file in data_files:
        file_path = Path(f"data/training/{file}")
        if not file_path.exists():
            missing_files.append(file)
            print(f"❌ Missing: {file}")
        else:
            print(f"✅ Found: {file}")

    # Check script files
    for file in script_files:
        file_path = Path(f"scripts/training/{file}")
        if not file_path.exists():
            missing_files.append(file)
            print(f"❌ Missing: {file}")
        else:
            print(f"✅ Found: {file}")
    
    if missing_files:
        print(f"\n❌ Missing required files: {missing_files}")
        return False
    
    print("✅ All training files validated")
    return True

async def main():
    """Main function to run SageMaker training"""
    print("🎯 MangoMAS SageMaker Training Runner")
    print("=" * 50)
    
    # Setup environment
    if not setup_environment():
        print("❌ Environment setup failed")
        sys.exit(1)
    
    # Validate files
    if not validate_training_files():
        print("❌ Training file validation failed")
        sys.exit(1)
    
    # Configuration options
    region = os.getenv('AWS_REGION', 'us-east-1')
    role_arn = os.getenv('SAGEMAKER_ROLE_ARN')  # Optional
    
    print(f"\n🚀 Configuration:")
    print(f"   Region: {region}")
    print(f"   Role ARN: {role_arn or 'Using default'}")
    print(f"   Parallel execution: True")
    print(f"   Max concurrent jobs: 3")
    
    # Create launcher
    launcher = MangoMASSageMakerLauncher(region=region, role_arn=role_arn)
    
    try:
        # Launch all training jobs
        print(f"\n🚀 Starting training jobs...")
        results = await launcher.launch_all_jobs(parallel=True, max_concurrent=3)
        
        # Generate summary
        summary = launcher.generate_summary_report()
        print(summary)
        
        # Save results
        launcher.save_results("training_results.json")
        
        # Check success
        if results and len(results) == len(launcher.agent_configs):
            print("🎉 All training jobs completed successfully!")
            return 0
        else:
            print("⚠️  Some training jobs failed. Check the summary above.")
            return 1
            
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        return 1
    except Exception as e:
        print(f"❌ Training failed with error: {e}")
        return 1

if __name__ == "__main__":
    # Run the training
    exit_code = asyncio.run(main())
    sys.exit(exit_code) 