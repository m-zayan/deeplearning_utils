#!/bin/sh

notebook=false

while getopts n: flag
do
    # shellcheck disable=SC2220
    case "${flag}" in
        n) notebook=true;;
    esac
done

if "$notebook"; then

  sudo rm -f /usr/bin/chromedriver
  sudo rm -f /usr/local/bin/chromedriver
  sudo rm -f /usr/local/share/chromedriver

else

  sudo rm -f /usr/bin/chromedriver

fi