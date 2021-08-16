#!/bin/sh

version=92.0.4515.107
addr=64

while getopts v:a: flag
do
    # shellcheck disable=SC2220
    case "${flag}" in
        v) version=${OPTARG};;
        a) addr=${OPTARG};;
    esac
done

sudo apt-get update
sudo apt-get install -y unzip xvfb libxi6 libgconf-2-4

sudo apt-get install default-jdk

sudo curl -sS -o - https://dl-ssl.google.com/linux/linux_signing_key.pub | apt-key add
sudo echo "deb [arch=amd64]  http://dl.google.com/linux/chrome/deb/ stable main" >> /etc/apt/sources.list.d/google-chrome.list
sudo apt-get -y update
sudo apt-get -y install google-chrome-stable

# https://chromedriver.storage.googleapis.com/index.html

wget https://chromedriver.storage.googleapis.com/"$version"/chromedriver_linux"$addr".zip
# shellcheck disable=SC2086
unzip chromedriver_linux$addr.zip

sudo mv chromedriver /usr/bin/chromedriver
sudo chown root:root /usr/bin/chromedriver
sudo chmod +x /usr/bin/chromedriver

sudo rm chromedriver_linux"$addr".zip