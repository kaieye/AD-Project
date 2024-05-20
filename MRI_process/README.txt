step 1 : python fslinstaller.py
# after fsl is installed , modify the .profile
add the content below:
# FSL Setup
FSLDIR=/usr/local/fsl
PATH=${FSLDIR}/share/fsl/bin:${PATH}
export FSLDIR PATH
. ${FSLDIR}/etc/fslconf/fsl.sh

step 2 : modify pipeline.json 
# rawPath is the original MRI folder
# rootPath is the place stored many processed data
# rootPath/scans stored processed MRI
# rootPath/images stored images
# rootPath/good stored final case

step 3 : python pipeline.py

step 4 : python biasFieldCorrection.py
