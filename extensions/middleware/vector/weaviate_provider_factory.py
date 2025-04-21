"""
Weaviate Provider Factory - Abstracts the creation of Weaviate provider instances
allowing seamless switching between embedded and cloud deployments.
"""
import os
import logging
from typing import Dict, Any, Optional, Union
from pydantic import BaseModel

# Import provider implementation
try:
    from .weaviate_provider import WeaviateProvider, WeaviateConfig
except ImportError:
    # Handle relative import when running as standalone
    from weaviate_provider import WeaviateProvider, WeaviateConfig

logger = logging.getLogger(__name__)


class WeaviateDeploymentType:
    """Enum-like class for Weaviate deployment types"""
    EMBEDDED = "embedded"
    CLOUD = "cloud"
    DOCKER = "docker"


class WeaviateProviderFactory:
    """
    Factory class for creating Weaviate provider instances.
    
    This abstraction allows switching between different deployment types:
    - Embedded: runs in-process, ideal for development
    - Cloud: connects to a cloud-hosted Weaviate instance
    - Docker: connects to a locally hosted Docker instance
    """
    
    @staticmethod
    def create(
        deployment_type: str = None,
        config: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> WeaviateProvider:
        """
        Create a WeaviateProvider instance based on the deployment type.
        
        Args:
            deployment_type: The type of deployment to create (embedded, cloud, docker)
            config: Configuration dictionary for the provider
            **kwargs: Additional arguments passed to the provider
            
        Returns:
            An initialized WeaviateProvider instance
        """
        # Get deployment type from config, kwargs, or environment variable, with fallback to docker
        if deployment_type is None:
            deployment_type = os.environ.get("WEAVIATE_DEPLOYMENT_TYPE", WeaviateDeploymentType.DOCKER)
        
        config = config or {}
        
        # Set up default configurations based on deployment type
        if deployment_type == WeaviateDeploymentType.EMBEDDED:
            logger.info("Creating embedded Weaviate provider")
            return WeaviateProviderFactory._create_embedded_provider(config, **kwargs)
            
        elif deployment_type == WeaviateDeploymentType.CLOUD:
            logger.info("Creating cloud Weaviate provider")
            return WeaviateProviderFactory._create_cloud_provider(config, **kwargs)
            
        elif deployment_type == WeaviateDeploymentType.DOCKER:
            logger.info("Creating Docker-based Weaviate provider")
            return WeaviateProviderFactory._create_docker_provider(config, **kwargs)
            
        else:
            raise ValueError(f"Unknown deployment type: {deployment_type}")
    
    @staticmethod
    def _create_embedded_provider(config: Dict[str, Any], **kwargs) -> WeaviateProvider:
        """Create a provider that uses embedded Weaviate"""
        try:
            # Import here to avoid dependency issues if weaviate isn't installed
            import weaviate
            from weaviate.embedded import EmbeddedOptions
            
            # Default embedded settings
            persistence_path = config.get("persistence_data_path") or os.environ.get(
                "WEAVIATE_PERSISTENCE_PATH", 
                os.path.join(os.path.expanduser("~"), ".local", "share", "weaviate-embedded")
            )
            
            # Configure embedded options
            embedded_options = EmbeddedOptions(
                persistence_data_path=persistence_path,
                additional_env_vars=config.get("additional_env_vars", {})
            )
            
            # Create the client directly with embedded options
            client = weaviate.connect_to_embedded(
                version=config.get("version", "1.23.7"),
                additional_env_vars=config.get("additional_env_vars", {})
            )
            
            # Create provider config from supplied config
            provider_config = WeaviateConfig(
                class_name=config.get("class_name", "Memory"),
                batch_size=config.get("batch_size", 100),
                batch_dynamic=config.get("batch_dynamic", True),
                batch_timeout_retries=config.get("batch_timeout_retries", 3),
                vector_dimensions=config.get("vector_dimensions", 384),
                **kwargs
            )
            
            return WeaviateProvider(client=client, config=provider_config)
            
        except ImportError as e:
            logger.error(f"Failed to import weaviate for embedded mode: {e}")
            raise ImportError(
                "Weaviate Python client is required for embedded mode. "
                "Install with 'pip install \"weaviate-client>=3.22.0\"'"
            ) from e
    
    @staticmethod
    def _create_cloud_provider(config: Dict[str, Any], **kwargs) -> WeaviateProvider:
        """Create a provider that connects to a Weaviate Cloud instance"""
        try:
            import weaviate
            
            # Required cloud settings
            cloud_url = config.get("url") or os.environ.get("WEAVIATE_CLOUD_URL")
            api_key = config.get("api_key") or os.environ.get("WEAVIATE_API_KEY")
            
            if not cloud_url:
                raise ValueError("Weaviate Cloud URL is required for cloud deployment. "
                                "Set 'url' in config or WEAVIATE_CLOUD_URL environment variable.")
            
            # Create auth config if API key is provided
            auth_config = weaviate.auth.AuthApiKey(api_key) if api_key else None
            
            # Connect to cloud instance
            client = weaviate.Client(
                url=cloud_url,
                auth_client_secret=auth_config,
                additional_headers=config.get("additional_headers", {})
            )
            
            # Create provider config from supplied config
            provider_config = WeaviateConfig(
                class_name=config.get("class_name", "Memory"),
                batch_size=config.get("batch_size", 100),
                batch_dynamic=config.get("batch_dynamic", True),
                batch_timeout_retries=config.get("batch_timeout_retries", 3),
                vector_dimensions=config.get("vector_dimensions", 384),
                **kwargs
            )
            
            return WeaviateProvider(client=client, config=provider_config)
            
        except ImportError as e:
            logger.error(f"Failed to import weaviate for cloud mode: {e}")
            raise ImportError(
                "Weaviate Python client is required for cloud mode. "
                "Install with 'pip install \"weaviate-client>=3.22.0\"'"
            ) from e
    
    @staticmethod
    def _create_docker_provider(config: Dict[str, Any], **kwargs) -> WeaviateProvider:
        """Create a provider that connects to a local Docker-hosted Weaviate instance"""
        try:
            import weaviate
            
            # Default Docker settings
            host = config.get("host") or os.environ.get("WEAVIATE_HOST", "localhost")
            port = config.get("port") or os.environ.get("WEAVIATE_PORT", "8080")
            
            # Create connection URL
            url = f"http://{host}:{port}"
            
            # Connect to Docker instance
            client = weaviate.Client(
                url=url,
                additional_headers=config.get("additional_headers", {})
            )
            
            # Create provider config from supplied config
            provider_config = WeaviateConfig(
                class_name=config.get("class_name", "Memory"),
                batch_size=config.get("batch_size", 100),
                batch_dynamic=config.get("batch_dynamic", True),
                batch_timeout_retries=config.get("batch_timeout_retries", 3),
                vector_dimensions=config.get("vector_dimensions", 384),
                **kwargs
            )
            
            return WeaviateProvider(client=client, config=provider_config)
            
        except ImportError as e:
            logger.error(f"Failed to import weaviate for Docker mode: {e}")
            raise ImportError(
                "Weaviate Python client is required for Docker mode. "
                "Install with 'pip install \"weaviate-client>=3.22.0\"'"
            ) from e


# Simple usage example
if __name__ == "__main__":
    # Example: Create embedded provider
    embedded_provider = WeaviateProviderFactory.create(
        deployment_type=WeaviateDeploymentType.EMBEDDED,
        config={
            "class_name": "TestMemory",
            "persistence_data_path": "./weaviate-data"
        }
    )
    
    # Example: Create cloud provider
    cloud_provider = WeaviateProviderFactory.create(
        deployment_type=WeaviateDeploymentType.CLOUD,
        config={
            "url": "https://your-cluster.weaviate.cloud",
            "api_key": "your-api-key",
            "class_name": "TestMemory"
        }
    )
    
    # Example: Create Docker provider (default)
    docker_provider = WeaviateProviderFactory.create(
        config={
            "host": "localhost",
            "port": "8080",
            "class_name": "TestMemory"
        }
    ) 