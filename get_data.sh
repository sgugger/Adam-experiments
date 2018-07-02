#!/usr/bin/env bash
ROOT="data"
CAR_DIR="${ROOT}/cars"
C10_DIR="${ROOT}/cifar10"
WT2_DIR="${ROOT}/wikitext"

mkdir -p "${ROOT}"
mkdir -p "${CAR_DIR}"
mkdir -p "${C10_DIR}"
mkdir -p "${WT2_DIR}"

echo "Downloading the datasets"
wget -c "http://imagenet.stanford.edu/internal/car196/car_ims.tgz" -P "${CAR_DIR}"
wget -c "http://imagenet.stanford.edu/internal/car196/cars_annos.mat" -P "${CAR_DIR}"
wget -c "http://pjreddie.com/media/files/cifar.tgz" -P "${C10_DIR}"
wget -c "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip" -P "${WT2_DIR}"


echo "Uncompressing the datasets"
cd "${CAR_DIR}"
tar -xzf "car_ims.tgz"
cd ../..
cd "${C10_DIR}"
tar -xzf "cifar.tgz"
cd ../..
cd "${WT2_DIR}"
unzip -q "wikitext-2-v1.zip"
cd ../..

echo "Preparing the datasets"
python prepare_data.py


