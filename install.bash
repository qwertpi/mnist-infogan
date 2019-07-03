echo 'Updating your packages'
sudo apt-get update && sudo apt-get upgrade -y
echo 'Installing Python, the Python Package Installer and software to download code from GitHub'
sudo apt-get install python3 python3-pip git -y
echo 'Downloading code from Github'
git clone https://github.com/qwertpi/mnist-infogan.git
cd mnist-dcgan
echo 'Installing the requried python libaries'
sudo -H pip3 install -U -r requirements.txt
sudo apt-get install libhdf5-serial-dev
echo 'Creating the required folders'
mkdir images
echo 'Done!'
