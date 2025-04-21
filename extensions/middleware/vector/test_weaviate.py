"""
Test script to verify Weaviate connection and basic functionality.
"""
import weaviate
import time
import sys

def test_weaviate_connection():
    """Test connection to Weaviate and create a test schema."""
    try:
        # Connect to Weaviate
        client = weaviate.Client("http://localhost:8080")
        
        # Check if Weaviate is ready
        if not client.is_ready():
            print("Weaviate is not ready yet. Waiting...")
            for _ in range(5):
                time.sleep(2)
                if client.is_ready():
                    print("Weaviate is now ready!")
                    break
            else:
                print("Failed to connect to Weaviate after multiple attempts.")
                sys.exit(1)
        
        # Get meta information
        meta = client.get_meta()
        print(f"Connected to Weaviate version: {meta['version']}")
        
        # Create a test class if it doesn't exist
        class_name = "TestMemory"
        try:
            # Check if class exists
            client.schema.get(class_name)
            print(f"Class '{class_name}' already exists.")
        except weaviate.exceptions.UnexpectedStatusCodeException:
            # Create class
            class_obj = {
                "class": class_name,
                "vectorizer": "none",  # We'll provide our own vectors
                "properties": [
                    {
                        "name": "content",
                        "dataType": ["text"],
                    },
                    {
                        "name": "metadata",
                        "dataType": ["object"],
                    }
                ]
            }
            client.schema.create_class(class_obj)
            print(f"Created test class '{class_name}'.")
        
        # Add a test object
        test_vector = [0.1] * 384  # Default vector dimension
        test_data = {
            "content": "This is a test memory for Weaviate integration.",
            "metadata": {
                "source": "test_script",
                "timestamp": time.time()
            }
        }
        
        # Add the object
        result = client.data_object.create(
            data_object=test_data,
            class_name=class_name,
            vector=test_vector
        )
        print(f"Added test object with ID: {result}")
        
        # Retrieve the object
        result = client.data_object.get_by_id(result, class_name=class_name)
        print("Successfully retrieved test object.")
        print(f"Test object content: {result['properties']['content']}")
        
        return True
    
    except Exception as e:
        print(f"Error testing Weaviate connection: {e}")
        return False

if __name__ == "__main__":
    print("Testing Weaviate connection...")
    success = test_weaviate_connection()
    
    if success:
        print("\n✅ Weaviate connection test successful!")
    else:
        print("\n❌ Weaviate connection test failed.")
        sys.exit(1) 