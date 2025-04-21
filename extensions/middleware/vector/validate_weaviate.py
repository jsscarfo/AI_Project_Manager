#!/usr/bin/env python
"""
Validate Weaviate connection script.
This script checks if a Weaviate instance is available and functioning properly.
"""

import argparse
import sys
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def validate_connection(config):
    """
    Validate connection to a Weaviate instance.
    
    Args:
        config (dict): Configuration for the connection with keys:
            - type: Connection type (local, remote, cloud)
            - host: Weaviate host (for local/remote)
            - port: Weaviate port (for local/remote)
            - url: Full URL (for cloud)
            - api_key: API key (for cloud)
            - timeout: Connection timeout in seconds
        
    Returns:
        bool: True if connection is successful, False otherwise
    """
    try:
        import weaviate
        
        connection_type = config.get("type", "local")
        timeout = config.get("timeout", 30)
        
        if connection_type == "cloud":
            # Cloud connection
            url = config.get("url", "")
            api_key = config.get("api_key", "")
            
            if not url or not api_key:
                logger.error("Cloud connection requires URL and API key")
                return False
                
            logger.info(f"Attempting to connect to Weaviate Cloud at {url}")
            
            # Configure auth with v4 syntax
            connection_params = weaviate.ConnectionParams.from_url(
                url=url,
                headers={"X-Weaviate-Api-Key": api_key}
            )
            
            # Create client
            client = weaviate.WeaviateClient(connection_params=connection_params)
        else:
            # Local or remote connection
            host = config.get("host", "localhost")
            port = config.get("port", "8080")
            url = f"http://{host}:{port}"
            
            logger.info(f"Attempting to connect to Weaviate at {url}")
            
            # Create client with v4 syntax
            connection_params = weaviate.ConnectionParams.from_url(url=url)
            client = weaviate.WeaviateClient(connection_params=connection_params)
        
        # Try to connect with timeout
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                # Check if Weaviate is ready
                is_ready = client.is_ready()
                if is_ready:
                    logger.info(f"Successfully connected to Weaviate")
                    
                    # Get metadata
                    try:
                        meta = client.get_meta()
                        logger.info(f"Weaviate version: {meta['version']}")
                        
                        # Show additional info for cloud
                        if connection_type == "cloud":
                            logger.info(f"Hostname: {meta.get('hostname', 'unknown')}")
                            
                    except Exception as e:
                        logger.warning(f"Could not retrieve metadata: {str(e)}")
                    
                    return True
                else:
                    logger.info("Weaviate is not ready yet, retrying...")
            except Exception as e:
                logger.warning(f"Connection attempt failed: {str(e)}")
            
            # Wait before retry
            time.sleep(2)
        
        logger.error(f"Failed to connect to Weaviate after {timeout} seconds")
        return False
    
    except ImportError:
        logger.error("Weaviate Python client not installed. Please install with 'pip install weaviate-client'")
        return False
    except Exception as e:
        logger.error(f"Unexpected error during validation: {str(e)}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate Weaviate connection")
    
    # Connection type
    parser.add_argument("--type", type=str, choices=["local", "remote", "cloud"], default="local", 
                        help="Connection type (local, remote, or cloud)")
    
    # Local/remote parameters
    parser.add_argument("--host", type=str, default="localhost", help="Weaviate host for local/remote")
    parser.add_argument("--port", type=str, default="8080", help="Weaviate port for local/remote")
    
    # Cloud parameters
    parser.add_argument("--url", type=str, help="Full URL for cloud instances (e.g., https://example.weaviate.network)")
    parser.add_argument("--api-key", type=str, help="API key for cloud instances")
    
    # General parameters
    parser.add_argument("--timeout", type=int, default=30, help="Connection timeout in seconds")
    
    args = parser.parse_args()
    
    # Build configuration
    config = {
        "type": args.type,
        "timeout": args.timeout
    }
    
    # Add type-specific parameters
    if args.type == "local":
        config["host"] = "localhost"
        config["port"] = "8080"
    elif args.type == "remote":
        config["host"] = args.host
        config["port"] = args.port
    elif args.type == "cloud":
        config["url"] = args.url
        config["api_key"] = args.api_key
    
    # Validate connection
    success = validate_connection(config)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1) 