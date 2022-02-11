set -e

# https://dev.to/bowmanjd/install-docker-on-windows-wsl-without-docker-desktop-34m9

apt update && apt install sudo
echo "installed sudo"
sudo apt install --no-install-recommends apt-transport-https ca-certificates curl gnupg2
echo "installed prereqs"
source /etc/os-release
ehcho "ran source"
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
echo "ran curl"
echo "deb [arch=amd64] https://download.docker.com/linux/${ID} ${VERSION_CODENAME} stable" | sudo tee /etc/apt/sources.list.d/docker.list
echo "ran echo"
sudo apt update
echo "ran apt update"
sudo apt install docker-ce docker-ce-cli containerd.io
echo "installed docker!"
