@echo off

set "requirements_txt=%~dp0\requirements.txt"
set "requirements_post_txt=%~dp0\requirements_post_win_py311_cu121.txt"
set "python_exec=..\..\..\python_embeded\python.exe"

echo Starting to install ComfyUI-LayerDivider...

if exist "%python_exec%" (
    echo Installing with ComfyUI Windows Portable Python Embeded Environment
    "%python_exec%" -s -m pip install onnxruntime-gpu --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/

    "%python_exec%" -s -m pip install -r "%requirements_txt%"
	
	"%python_exec%" -s -m pip install psd-tools --no-deps
	
    "%python_exec%" -s -m pip install -r "%requirements_post_txt%"    
) else (
    echo ERROR: Cannot find ComfyUI Windows Portable Python Embeded Environment "%python_exec%"
)

echo Install Finished. Press any key to continue...

pause