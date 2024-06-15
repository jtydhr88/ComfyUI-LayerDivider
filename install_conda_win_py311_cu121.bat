@echo off

set "requirements_txt=%~dp0\requirements.txt"
set "requirements_post_txt=%~dp0\requirements_post_win_py311_cu121.txt"

echo Starting to install ComfyUI-LayerDivider...

echo Installing with conda python3.11 Environment
pip install onnxruntime-gpu --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/

pip install -r "%requirements_txt%"

pip install psd-tools --no-deps

pip install -r "%requirements_post_txt%"    

echo Install Finished. Press any key to continue...

pause