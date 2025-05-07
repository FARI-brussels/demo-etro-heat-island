import requests
import base64
import time
import threading
import json
from concurrent.futures import ThreadPoolExecutor

# Backend URL
BACKEND_URL = "http://127.0.0.1:5000/process_image"

def encode_image(image_path):
    """Encode an image file as base64"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def send_request(image_path, delay=0, request_id=None):
    """Send a request to the backend"""
    if delay > 0:
        print(f"Request {request_id}: Waiting {delay} seconds before sending...")
        time.sleep(delay)
    
    try:
        # Encode the image
        encoded_image = encode_image(image_path)
        
        # Create the payload
        payload = {"image": encoded_image}
        
        # Send the request
        print(f"Request {request_id}: Sending...")
        start_time = time.time()
        response = requests.post(BACKEND_URL, json=payload)
        elapsed = time.time() - start_time
        
        # Process the response
        if response.status_code == 200:
            response_data = response.json()
            status = response_data.get('status', 'unknown')
            
            if status == 'success':
                print(f"Request {request_id}: Successful (took {elapsed:.2f}s)")
                # Could save the processed image here if needed
                temperature = response_data.get('temperature', 'unknown')
                print(f"Request {request_id}: Temperature: {temperature}")
                return True
            elif status == 'cancelled':
                print(f"Request {request_id}: Cancelled - {response_data.get('message', '')}")
                return False
            else:
                print(f"Request {request_id}: Unknown status - {response_data}")
                return False
        else:
            print(f"Request {request_id}: Failed with status code {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"Request {request_id}: Exception - {str(e)}")
        return False

def test_concurrent_requests():
    """Test sending multiple concurrent requests"""
    image_path = "test/testheatmap.jpg"
    
    # Define the delays for each request (in seconds)
    requests_config = [
        {"id": 1, "delay": 0},    # Immediate request
        {"id": 2, "delay": 0.1},  # Slightly delayed request
        {"id": 3, "delay": 0.2},  # More delayed request
    ]
    
    print("Starting concurrent request test...")
    print("Sending 3 requests with different delays.")
    print("Only the last request (id=3) should complete successfully.")
    
    # Use a thread pool to send requests concurrently
    with ThreadPoolExecutor(max_workers=len(requests_config)) as executor:
        futures = []
        for req in requests_config:
            future = executor.submit(
                send_request, 
                image_path, 
                req["delay"], 
                req["id"]
            )
            futures.append((req["id"], future))
        
        # Wait for all futures to complete
        results = []
        for req_id, future in futures:
            result = future.result()
            results.append({"id": req_id, "success": result})
    
    # Print summary
    print("\nTest Results Summary:")
    for result in results:
        print(f"Request {result['id']}: {'Successful' if result['success'] else 'Cancelled/Failed'}")
    
    # Verify that only the last request was successful
    successful_requests = [r for r in results if r['success']]
    if len(successful_requests) == 1 and successful_requests[0]['id'] == 3:
        print("\nTest PASSED: Only the most recent request completed successfully.")
    else:
        print("\nTest FAILED: Unexpected results.")
        print(f"Successful requests: {successful_requests}")

if __name__ == "__main__":
    test_concurrent_requests() 