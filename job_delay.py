import time

def delay_job(delay_seconds=120):
    """
    Adds a 2-minute delay between quantum job executions.
    
    Args:
        delay_seconds (int): Delay time in seconds, defaults to 120 seconds (2 minutes)
    """
    print(f"Adding delay of {delay_seconds} seconds between quantum jobs...")
    time.sleep(delay_seconds)
    print("Delay complete, continuing with next job.")