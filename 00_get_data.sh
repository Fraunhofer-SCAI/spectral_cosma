#!/bin/bash

dataset=$1

echo "Download data for dataset: " $dataset

mkdir -p data
cd DATA
mkdir original_meshes

if [[ "$dataset" == "car_TRUCK" ]]
then
    dataset_spec=car_TRUCK_r4_2204
    wget https://owncloud.scai.fraunhofer.de/index.php/s/kBBSrZFX8kpEbdn/download/car_TRUCK_r4_2204.tar.gz
    cd original_meshes
    wget https://owncloud.scai.fraunhofer.de/index.php/s/aQep26pRwnamDyf/download/car_TRUCK.tar.gz
elif [[ "$dataset" == "car_YARIS" ]]
then 
    dataset_spec=car_YARIS_r4_2204
    wget https://owncloud.scai.fraunhofer.de/index.php/s/tx3c3pm4GraBP49/download/car_YARIS_r4_2204.tar.gz
    cd original_meshes
    wget https://owncloud.scai.fraunhofer.de/index.php/s/gzsiGYHYkaGTW4K/download/car_YARIS.tar.gz
elif [[ "$dataset" == "FAUST" ]]
then 
    dataset_spec=FAUST_r4_2203
    wget https://owncloud.scai.fraunhofer.de/index.php/s/FzLg4st7Jr8BNeZ/download/FAUST_r4_2203.tar.gz
    cd original_meshes
    wget https://owncloud.scai.fraunhofer.de/index.php/s/t7paqc2SmAsPKFq/download/FAUST.tar.gz
else
    echo "Not a valid dataset."
fi


echo "tar -xzvf $dataset.tar.gz && rm $dataset.tar.gz"
tar -xzvf $dataset.tar.gz && rm $dataset.tar.gz
echo "downloaded the irregular mesh data and putting it in: data/original_meshes/$dataset"

cd ..
echo "tar -xzvf $dataset_spec.tar.gz && rm $dataset_spec.tar.gz"
tar -xzvf $dataset_spec.tar.gz && rm $dataset_spec.tar.gz
echo "downloaded the data and putting it in: data/$dataset_spec"
