#!/bin/bash
echo "BRANCH - ${BRANCH}"
if [[ ${BRANCH} == "refs/heads/stage-ci" ]]; then 
    FILE_NAME="drawerai-services/drawerai-dev-values.yaml"
    FILE_NAME3="stitch-my-sheet-services/stitch-my-sheet-dev-values.yaml"
elif [[ ${BRANCH} == "origin/stage-ci" ]]; then 
    FILE_NAME="drawerai-services/drawerai-dev-values.yaml"
    FILE_NAME3="stitch-my-sheet-services/stitch-my-sheet-dev-values.yaml"
elif [[ ${BRANCH} == "refs/heads/main" ]]; then 
    FILE_NAME="drawerai-services/drawerai-prod-values.yaml"
    FILE_NAME3="stitch-my-sheet-services/stitch-my-sheet-prod-values.yaml"
elif [[ ${BRANCH} == "origin/main" ]]; then 
    FILE_NAME="drawerai-services/drawerai-prod-values.yaml"
    FILE_NAME3="stitch-my-sheet-services/stitch-my-sheet-prod-values.yaml"
fi

FILE_NAME2="pdf-processing-service/values.yaml"

source ${WORKSPACE}/.env
cd ${WORKSPACE}/helm
echo ${IMAGE_TAG}

### For main product

yq ".[\"${SERVICE_NAME//_/-}\"].image.tag" $FILE_NAME
yq -iy ".[\"${SERVICE_NAME//_/-}\"].image.tag = \"$IMAGE_TAG\"" $FILE_NAME
yq ".[\"${SERVICE_NAME//_/-}\"].image.tag" $FILE_NAME

yq ".image.tag" $FILE_NAME2
yq -iy ".image.tag = \"$IMAGE_TAG\"" $FILE_NAME2
yq ".image.tag" $FILE_NAME2

### For sms product

yq ".[\"${SERVICE_NAME//_/-}\"].image.tag" $FILE_NAME3
yq -iy ".[\"${SERVICE_NAME//_/-}\"].image.tag = \"$IMAGE_TAG\"" $FILE_NAME3
yq ".[\"${SERVICE_NAME//_/-}\"].image.tag" $FILE_NAME3

