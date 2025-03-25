import os
from dotenv import load_dotenv, set_key
from openai import OpenAI
import time

# Load environment variables
load_dotenv()

# Initialize the client
client = OpenAI(
    api_key=os.getenv('OPENAI_API_KEY')  # Get API key from environment variable
)

def update_env_file(new_job_id, new_model_id):
    """Update .env file with new job and model IDs, rotating the old ones."""
    # Get current values
    current_job_id = os.getenv('FINE_TUNE_JOB_ID')
    current_model_id = os.getenv('CURRENT_MODEL_ID')
    
    # Update .env file with old values
    if current_job_id:
        set_key('.env', 'OLD_FINE_TUNE_JOB_ID', current_job_id)
    if current_model_id:
        set_key('.env', 'OLD_CURRENT_MODEL_ID', current_model_id)
    
    # Set new values
    set_key('.env', 'FINE_TUNE_JOB_ID', new_job_id)
    set_key('.env', 'CURRENT_MODEL_ID', new_model_id)
    print("✅ Updated .env file with new model and job IDs")
    print(f"Previous model ID {current_model_id} saved as OLD_CURRENT_MODEL_ID")
    print(f"Previous job ID {current_job_id} saved as OLD_FINE_TUNE_JOB_ID")

def check_status(job_id):
    """Check the status of a fine-tuning job"""
    try:
        job = client.fine_tuning.jobs.retrieve(job_id)
        return job.status
    except Exception as e:
        print(f"Error checking status: {e}")
        return "failed"

# Main execution wrapped in try-except
try:
    # 🔁 Step 1: Upload the training file
    print("Uploading training file...")
    with open("jon_training_data_from_convo.jsonl", "rb") as file:
        upload = client.files.create(
            file=file,
            purpose="fine-tune"
        )
    file_id = upload.id
    print(f"✅ File uploaded: {file_id}")

    # 🧠 Step 2: Start fine-tuning
    print("Starting fine-tune job on gpt-3.5-turbo...")
    fine_tune = client.fine_tuning.jobs.create(
        training_file=file_id,
        model="gpt-3.5-turbo"
    )

    job_id = fine_tune.id
    print(f"🚀 Fine-tune job started! Job ID: {job_id}")

    # 📊 Step 3: Monitor progress
    print("\nMonitoring fine-tuning progress...")
    status = "queued"
    while status not in ["succeeded", "failed"]:
        status = check_status(job_id)
        print(f"Status: {status}")
        if status == "succeeded":
            print("✅ Fine-tuning completed successfully!")
            # Get the fine-tuned model ID
            job = client.fine_tuning.jobs.retrieve(job_id)
            new_model_id = job.fine_tuned_model
            print(f"Your fine-tuned model ID: {new_model_id}")
            
            # Update .env file with new IDs
            update_env_file(job_id, new_model_id)
            
        elif status == "failed":
            print("❌ Fine-tuning failed.")
            job = client.fine_tuning.jobs.retrieve(job_id)
            print(f"Error: {job.error}")
            break
        else:
            time.sleep(60)  # Check every minute

except Exception as e:
    print(f"❌ An error occurred: {e}")
