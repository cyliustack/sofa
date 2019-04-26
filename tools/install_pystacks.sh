#!/usr/bin/env bash

echo -e "Installing Rust with using rustup"
curl https://sh.rustup.rs -sSf | sh
source $HOME/.cargo/env


echo -e "Installing py-spy with specific version fo sofa"
git clone https://github.com/notreal1995/py-spy.git
cd py-spy
python3.6 setup.py install
cd ..


