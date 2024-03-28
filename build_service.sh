#!/bin/bash

ECR_REPO="${AWS_ACCOUNT}.dkr.ecr.${AWS_REGION}.amazonaws.com"
DOCKER_REPO=${SERVICE_NAME}
# echo ${MVN_VERSION}
# echo ${ECR_REPO}
SHORT_BRANCH=`echo ${BRANCH} | awk -F "/" '{print $NF}'`
SHORT_HASH=`git rev-parse --short=10 HEAD`

if [ "$SHORT_BRANCH" == "dev-ci" ]; then
  IMAGE_TAG="dev-ci-latest"
elif [ "$SHORT_BRANCH" == "stage-ci" ]; then
  IMAGE_TAG="${SHORT_BRANCH}-${SHORT_HASH}-${BUILD_NUMBER}"
elif [ "$SHORT_BRANCH" == "main" ]; then
  IMAGE_TAG="${SHORT_BRANCH}-${SHORT_HASH}-${BUILD_NUMBER}"
else
  echo "Not sufficient branch!"
  exit 1
fi

echo "IMAGE_TAG: $IMAGE_TAG"
echo "Current image tag - ${IMAGE_TAG}"
EXISTING_TAG_COUNT=$(aws ecr list-images --repository-name ${DOCKER_REPO} --filter tagStatus=TAGGED --region ${AWS_REGION} --no-paginate --query "length(imageIds[?imageTag==\`${IMAGE_TAG}\`])")
echo "EXISTING_TAG_COUNT - ${EXISTING_TAG_COUNT}"
#IMAGE_TAG="latest"

echo Login to ECR
eval $(aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin ${ECR_REPO})

#docker build -t ${ECR_REPO}/${DOCKER_REPO}:${IMAGE_TAG} --build-arg /src_env/env_dev.env --no-cache=true . || exit 1
docker build -t ${ECR_REPO}/${DOCKER_REPO}:${IMAGE_TAG} --no-cache=true . || exit 1
docker images

echo "docker push ${ECR_REPO}/${DOCKER_REPO}:${IMAGE_TAG}"

docker push ${ECR_REPO}/${DOCKER_REPO}:${IMAGE_TAG} || exit 1
echo "Workspace is ${WORKSPACE}"
echo "IMAGE_TAG=${IMAGE_TAG}" > ${WORKSPACE}/.env
echo "${IMAGE_TAG}" > image_tag.txt
