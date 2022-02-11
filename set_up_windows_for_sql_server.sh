# https://dev.to/bowmanjd/install-docker-on-windows-wsl-without-docker-desktop-34m9

apt update && apt install sudo
sudo apt install --no-install-recommends apt-transport-https ca-certificates curl gnupg2
source /etc/os-release
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
echo "deb [arch=amd64] https://download.docker.com/linux/${ID} ${VERSION_CODENAME} stable" | sudo tee /etc/apt/sources.list.d/docker.list
sudo apt update
sudo apt install docker-ce docker-ce-cli containerd.io
