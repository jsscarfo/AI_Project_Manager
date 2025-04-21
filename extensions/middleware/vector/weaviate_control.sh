#!/bin/bash

# Change to script directory
cd "$(dirname "$0")"

echo "Weaviate Control Center"
echo "======================"
echo
echo "1. Validate Local Weaviate Instance"
echo "2. Validate Remote Weaviate Instance"
echo "3. Test Embedded Weaviate (no Docker)"
echo "4. Test Weaviate Cloud Connection"
echo "5. Start Local Docker Instance"
echo "6. Stop Local Docker Instance"
echo "7. Restart Local Docker Instance"
echo "8. Run Batch Import Process"
echo "9. Update Weaviate Schema"
echo

read -p "Enter your choice (1-9): " choice

if [ "$choice" = "1" ]; then
    echo
    echo "Validating Local Weaviate..."
    python validate_weaviate.py --type local
    
    if [ $? -eq 0 ]; then
        echo "Local Weaviate connection is valid."
    else
        echo "Failed to connect to local Weaviate."
        echo "Make sure the container is running. You can start it with option 5."
    fi

elif [ "$choice" = "2" ]; then
    echo
    echo "Validating Remote Weaviate..."
    
    read -p "Enter Weaviate host (without http/https): " HOST
    read -p "Enter Weaviate port: " PORT
    
    python validate_weaviate.py --host $HOST --port $PORT --type remote
    
    if [ $? -eq 0 ]; then
        echo "Remote Weaviate connection is valid."
    else
        echo "Failed to connect to remote Weaviate."
    fi

elif [ "$choice" = "3" ]; then
    echo
    echo "Running Embedded Weaviate Test..."
    python weaviate_standalone_test.py
    
elif [ "$choice" = "4" ]; then
    echo
    echo "Running Weaviate Cloud Test..."
    python weaviate_cloud_test.py
    
elif [ "$choice" = "5" ]; then
    echo
    echo "Starting Weaviate Docker Instance..."
    docker-compose -f docker-compose.weaviate.yml up -d
    
    echo
    echo "Weaviate is starting. It will be available at http://localhost:8080"
    echo "Startup may take up to 30 seconds."
    
    echo
    read -p "Would you like to validate the connection when startup completes? (y/n): " validate
    if [ "$validate" = "y" ] || [ "$validate" = "Y" ]; then
        echo "Waiting for startup to complete..."
        sleep 30
        python validate_weaviate.py --type local
        
        if [ $? -eq 0 ]; then
            echo "Local Weaviate connection is valid."
        else
            echo "Failed to connect to local Weaviate. It may need more time to start."
        fi
    fi
    
elif [ "$choice" = "6" ]; then
    echo
    echo "Stopping Weaviate Docker Instance..."
    docker-compose -f docker-compose.weaviate.yml down
    
    echo "Weaviate has been stopped."
    
elif [ "$choice" = "7" ]; then
    echo
    echo "Restarting Weaviate Docker Instance..."
    docker-compose -f docker-compose.weaviate.yml down
    docker-compose -f docker-compose.weaviate.yml up -d
    
    echo
    echo "Weaviate has been restarted. It will be available at http://localhost:8080"
    echo "Startup may take up to 30 seconds."
    
    echo
    read -p "Would you like to validate the connection when restart completes? (y/n): " validate
    if [ "$validate" = "y" ] || [ "$validate" = "Y" ]; then
        echo "Waiting for startup to complete..."
        sleep 30
        python validate_weaviate.py --type local
        
        if [ $? -eq 0 ]; then
            echo "Local Weaviate connection is valid."
        else
            echo "Failed to connect to local Weaviate. It may need more time to start."
        fi
    fi
    
elif [ "$choice" = "8" ]; then
    echo
    echo "Running Batch Import Process..."
    
    echo "Select import source:"
    echo "1. Import from local JSON file"
    echo "2. Import from database"
    read -p "" import_source
    
    if [ "$import_source" = "1" ]; then
        read -p "Enter path to JSON file: " json_file
        echo "Importing from $json_file..."
        python batch_import.py --source file --file "$json_file"
    elif [ "$import_source" = "2" ]; then
        echo "Importing from database..."
        python batch_import.py --source database
    else
        echo "Invalid choice."
    fi
    
elif [ "$choice" = "9" ]; then
    echo
    echo "Updating Weaviate Schema..."
    python update_schema.py
    
    if [ $? -eq 0 ]; then
        echo "Schema updated successfully."
    else
        echo "Failed to update schema."
    fi
    
else
    echo
    echo "Invalid choice. Please run the script again."
fi

echo
echo "Operation complete!" 