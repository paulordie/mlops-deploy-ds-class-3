#!/bin/bash

sudo apt-get update

sudo apt-get install wget -y

sudo apt install zlib1g-dev

sudo apt-get install -y libjpeg-dev
wget https://repo.anaconda.com/archive/Anaconda3-2021.05-Linux-x86_64.sh

bash Anaconda3-2021.05-Linux-x86_64.sh

source .bashrc

conda create -n deploy_env python=3.9

conda activate deploy_env

sudo apt-get install git-all