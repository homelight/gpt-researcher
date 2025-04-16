from botocore.client import Config
import boto3
import json
from datetime import datetime, timezone
import hashlib
import logging
import sys, os
import inspect
# Configure logger to output to console
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)

KINESIS_STREAM_NAME = "web-researcher-external-api-events-prod-DataStream"
# Define fields for logging and Redshift table structure
LOG_FIELDS = [
    "id",
    "vendor_name",
    "api_name",
    "reference_id_type",
    "reference_id",
    "app_name",
    "app_context",
    "status",
    "hit_cache",
    "event_count",
    "event_time",
]

# def test():
#     log_data_to_redshift({
#        "vendor_name": "tavily",
#        "api_name": "search",
#        "reference_id_type": "lead_id",
#        "reference_id": "999999",
#        "app_name": "grazibot",
#        "app_context": "grazibot_tavily_websearch",
#        "status": "success",
#        "hit_cache": False,
#        "event_count": 1,
#        "event_time": datetime.datetime.now(datetime.timezone.utc)
#     })

# log_data_to_redshift({
#     "vendor_name": "brightdata",
#     "api_name": "search",
#     "app_context":"", #don't include. string added by get_call_stack_string()
#     "app_name": self.app_name,
#     "event_count": 1,
#     "event_time": datetime.now().isoformat(), #don't include. timestamp added by log_data_to_redshift()
#     "id": "123", #don't include. varchar added by log_data_to_redshift()
#     "reference_id_type": self.reference_id_type,
#     "reference_id": self.lead_id,
#     "status": "success",
#     "hit_cache": False,
# })

def get_call_stack_string(levels=4):
    # Get the current call stack; index 0 is this function.
    stack = inspect.stack()
    call_stack = []
    prev_filename = None

    for frame_info in stack[2:levels+2]:
        # Extract only the file name from the full file path.
        current_file = os.path.basename(frame_info.filename)
        if current_file != prev_filename:
            # Include the file name if it's different from the previous one.
            call_stack.append(f"{current_file}.{frame_info.function}")
            prev_filename = current_file
        else:
            call_stack.append(frame_info.function)
    
    return "::".join(call_stack[::-1])


def log_data_to_redshift(event_data):
    """Log data to Kinesis and then to Redshift

    Args:
        event_data (dict): The data to log
    """
    # Create an id
    created_at = str(datetime.now(timezone.utc))
    hash_str = (str(event_data["reference_id"]) + "_" + created_at).encode()
    hash_id = hashlib.md5(hash_str).hexdigest()
    event_data["id"] = hash_id

    # Convert datetime to string to make it JSON serializable
    if "event_time" in event_data and isinstance(event_data["event_time"], datetime):
        event_data["event_time"] = event_data["event_time"].isoformat()
    else:
        event_data["event_time"] = datetime.now(timezone.utc).isoformat()

    event_data["app_context"] = get_call_stack_string()

    index_map = {v: i for i, v in enumerate(LOG_FIELDS)}
    final_log_dict = {
        k: v
        for k, v in sorted(event_data.items(), key=lambda pair: index_map[pair[0]])
    }

    data_dict = json.dumps(final_log_dict)

    try:
        config = Config(
            connect_timeout=2,
            read_timeout=10,
            retries={"max_attempts": 2},
        )
        kinesis_client = boto3.client("kinesis", region_name="us-east-1", config=config)
        response = kinesis_client.put_record(
            StreamName=KINESIS_STREAM_NAME,
            Data=data_dict,
            PartitionKey=hash_id
        )
        logger.debug(f"Data sent to Kinesis: {data_dict}")
    except Exception as e:
        logger.error(f"Exception occurred: {str(e)}")
        logger.error("Could not log data into Redshift from Kinesis")



# # Redshift table DDL for web_researcher_external_api_events
# REDSHIFT_TABLE_DDL = """
# CREATE TABLE IF NOT EXISTS raw_import.web_researcher_external_api_events (
#     id VARCHAR(100),
#     vendor_name VARCHAR(100),
#     api_name VARCHAR(100),
#     reference_id_type VARCHAR(100),
#     reference_id VARCHAR(100),
#     app_name VARCHAR(100),
#     app_context VARCHAR(5000),
#     status VARCHAR(5000),
#     hit_cache BOOLEAN,
#     event_count INTEGER,
#     event_time TIMESTAMP
# )
# DISTKEY (reference_id)
# SORTKEY (event_time);
# """


# if __name__ == "__main__":
#     test()