@echo off
cd /d "%~dp0"

echo Weaviate Control Center
echo ======================
echo.
echo 1. Validate Local Weaviate Instance
echo 2. Validate Remote Weaviate Instance
echo 3. Test Embedded Weaviate (no Docker)
echo 4. Test Weaviate Cloud Connection
echo 5. Start Local Docker Instance
echo 6. Stop Local Docker Instance
echo 7. Restart Local Docker Instance
echo 8. Run Batch Import Process
echo 9. Update Weaviate Schema
echo.

set /p choice="Enter your choice (1-9): "

if "%choice%"=="1" (
    echo.
    echo Validating Local Weaviate...
    python validate_weaviate.py --type local
    
    if %ERRORLEVEL% EQU 0 (
        echo Local Weaviate connection is valid.
    ) else (
        echo Failed to connect to local Weaviate.
        echo Make sure the container is running. You can start it with option 5.
    )

) else if "%choice%"=="2" (
    echo.
    echo Validating Remote Weaviate...
    
    set /p HOST="Enter Weaviate host (without http/https): "
    set /p PORT="Enter Weaviate port: "
    
    python validate_weaviate.py --host %HOST% --port %PORT% --type remote
    
    if %ERRORLEVEL% EQU 0 (
        echo Remote Weaviate connection is valid.
    ) else (
        echo Failed to connect to remote Weaviate.
    )

) else if "%choice%"=="3" (
    echo.
    echo Running Embedded Weaviate Test...
    python weaviate_standalone_test.py
    
) else if "%choice%"=="4" (
    echo.
    echo Running Weaviate Cloud Test...
    python weaviate_cloud_test.py
    
) else if "%choice%"=="5" (
    echo.
    echo Starting Weaviate Docker Instance...
    docker-compose -f docker-compose.weaviate.yml up -d
    
    echo.
    echo Weaviate is starting. It will be available at http://localhost:8080
    echo Startup may take up to 30 seconds.
    
    echo.
    echo Would you like to validate the connection when startup completes? (Y/N)
    set /p validate=""
    if /i "%validate%"=="Y" (
        echo Waiting for startup to complete...
        timeout /t 30 /nobreak >nul
        python validate_weaviate.py --type local
        
        if %ERRORLEVEL% EQU 0 (
            echo Local Weaviate connection is valid.
        ) else (
            echo Failed to connect to local Weaviate. It may need more time to start.
        )
    )
    
) else if "%choice%"=="6" (
    echo.
    echo Stopping Weaviate Docker Instance...
    docker-compose -f docker-compose.weaviate.yml down
    
    echo Weaviate has been stopped.
    
) else if "%choice%"=="7" (
    echo.
    echo Restarting Weaviate Docker Instance...
    docker-compose -f docker-compose.weaviate.yml down
    docker-compose -f docker-compose.weaviate.yml up -d
    
    echo.
    echo Weaviate has been restarted. It will be available at http://localhost:8080
    echo Startup may take up to 30 seconds.
    
    echo.
    echo Would you like to validate the connection when restart completes? (Y/N)
    set /p validate=""
    if /i "%validate%"=="Y" (
        echo Waiting for startup to complete...
        timeout /t 30 /nobreak >nul
        python validate_weaviate.py --type local
        
        if %ERRORLEVEL% EQU 0 (
            echo Local Weaviate connection is valid.
        ) else (
            echo Failed to connect to local Weaviate. It may need more time to start.
        )
    )
    
) else if "%choice%"=="8" (
    echo.
    echo Running Batch Import Process...
    
    echo Select import source:
    echo 1. Import from local JSON file
    echo 2. Import from database
    set /p import_source=""
    
    if "%import_source%"=="1" (
        set /p json_file="Enter path to JSON file: "
        echo Importing from %json_file%...
        python batch_import.py --source file --file "%json_file%"
    ) else if "%import_source%"=="2" (
        echo Importing from database...
        python batch_import.py --source database
    ) else (
        echo Invalid choice.
    )
    
) else if "%choice%"=="9" (
    echo.
    echo Updating Weaviate Schema...
    python update_schema.py
    
    if %ERRORLEVEL% EQU 0 (
        echo Schema updated successfully.
    ) else (
        echo Failed to update schema.
    )
    
) else (
    echo.
    echo Invalid choice. Please run the script again.
)

echo.
echo Operation complete!
pause 