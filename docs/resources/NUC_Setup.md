# GFA/FVC NUC Setup

- Download latest version of Ubuntu Server 20.04 LST from [here](https://releases.ubuntu.com/20.04/ubuntu-20.04.1-live-server-amd64.iso).

- During the installation or later, set eth0 to static `10.25.10.XX` where `XX` is 21-26 for each one of the six NUCs for a given observatory, and `XX=20` for the FVC NUC. Set `eth1` to static `192.168.1.1`. In both cases, set the netmask to `255.255.0.0`.

- Create a user `sdss` and select the same password for all the NUCs.

- Enable `openssh` to allow incoming ssh connections on port 22.

- During the installation, do **not** install docker from snap.

- After installation, confirm that `/etc/netplan/01-network.yaml` contains

```yaml
network:
  version: 2
  renderer: networkd
  ethernets:
    enp3s0:
      dhcp4: no
      dhcp6: no
      addresses:
        - 10.25.10.21/16
      gateway4: 10.25.1.252
      nameservers:
        addresses: [192.41.211.10, 192.41.211.11]
    enp4s0:
      dhcp4: no
      dhcp6: no
      addresses:
        - 192.168.1.1/24
```

- Apply the configuration and disable the `systemd-networkd-wait-online` service:

```bash
sudo netplan apply
sudo systemctl disable systemd-networkd-wait-online.service
sudo systemctl mask systemd-networkd-wait-online.service
```

- In `/etc/hostname` set the computer hostname to `sdss-gfaX.apo.nmsu.edu` where `X=1-6` for the GFA NUCs. For the FVC NUC set the hostname to `sdss-fvc`. Replace `.apo.nmsu.edu` with `.lco.cl` for the LCO-bound NUCs.

- Edit the hosts file `/etc/hosts` to include

```text
127.0.0.1 localhost
127.0.1.1 sdss-gfaX

10.25.1.1       sdss-hub sdss-hub.apo.nmsu.edu
10.25.1.121     sdss-gfa1 sdss-gfa1.apo.nmsu.edu
10.25.1.122     sdss-gfa2 sdss-gfa2.apo.nmsu.edu
10.25.1.123     sdss-gfa3 sdss-gfa3.apo.nmsu.edu
10.25.1.124     sdss-gfa4 sdss-gfa4.apo.nmsu.edu
10.25.1.125     sdss-gfa5 sdss-gfa5.apo.nmsu.edu
10.25.1.126     sdss-gfa6 sdss-gfa6.apo.nmsu.edu
10.25.1.120     sdss-fvc sdss-fvc.apo.nmsu.edu
```

replacing `sdss-gfaX` with the correct NUC hostname. For the LCO-bound NUCs, replace `apo.nmsu.edu` with `.lco.cl`.

- Install `wpasupplicant` and `net-tools` (`sudo apt install wpasupplicant net-tools`). This seems to be needed for the wifi, in case we want to use it at some point.

- Make sure wifi is down with `sudo ip link set wls1 down`.

- Make sure firewall is disabled with `sudo ufw disable`.

- Add this key to `/home/sdss/.ssh/authorized_keys` and make the file read-only (`chmod 600 .ssh/authorized_keys`):

```text
ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABgQDBA+fGJtEZWD99GUAQESwg3PzsLMG2/TvBVQMnr5eXr59ZeO6gi89Ezi1y9Tni8pf0rPsNFr5iHgahi+OeHtSvc+hW2MieKe7xu8hEt5+pLdNnvoiBXGwW12aut+nbYEcdIsttUEqnNp7iJoAf9SWfFnIdhy6ArHeWWT83UHXIs+k3q5jdMIynFlvJxLfpR1n1t9Wq9c0+x6XxnJ6Q6ewQoPViiN885/tJq+4CB/TG+kbf5A8l3UyCTNt7YmWls6djrQtRZmDAzUSWJYA//3ebx9RH7ZmkTRTLPM/et98oec7BztKG+b5wGPqPCT/6XP3ofoHc+wvn8r8w85PmQVGwBBnDKBJu3L2ucPHJjm/YRMuPodR5wnKyZfZtrbbnXi/DajItYHaC6u4wAeMYC75zlyGL6Ev3yYDgJZ3BM69kUnjAWRM2JUVHQUWVSRl06QX+A07y2wckjF0md3oPluV5uTHLnp606Y4DCjI8slJDNZWLyNt/lBCvrimOTPQAMlU= albireo@albireopro.lan
```

- Install docker by following the instructions on [this page](https://docs.docker.com/engine/install/ubuntu/). After installing it, confirm that there is a docker group

```bash
sudo groupadd docker
sudo usermod -aG docker sdss
sudo systemctl restart docker
```

- Log out and log in again and test docker by doing `docker run hello-world` (requires the computer being connected to the internet). The output should be something like

```text
Unable to find image 'hello-world:latest' locally
latest: Pulling from library/hello-world
0e03bdcc26d7: Pull complete
Digest: sha256:8c5aeeb6a5f3ba4883347d3747a7249f491766ca1caa47e5da5dfcf6b9b717c0
Status: Downloaded newer image for hello-world:latest

Hello from Docker!
This message shows that your installation appears to be working correctly.

To generate this message, Docker took the following steps:
 1. The Docker client contacted the Docker daemon.
 2. The Docker daemon pulled the "hello-world" image from the Docker Hub.
    (amd64)
 3. The Docker daemon created a new container from that image which runs the
    executable that produces the output you are currently reading.
 4. The Docker daemon streamed that output to the Docker client, which sent it
    to your terminal.

To try something more ambitious, you can run an Ubuntu container with:
 $ docker run -it ubuntu bash

Share images, automate workflows, and more with a free Docker ID:
 https://hub.docker.com/

For more examples and ideas, visit:
 https://docs.docker.com/get-started/
```

- Install NFS client tools with `sudo apt -y install nfs-common`.

- Create the file `/etc/docker/daemon.json` with content

```json
{
    "insecure-registries":["10.25.1.1:5000", "sdss-hub:5000"],
    "hosts": ["tcp://0.0.0.0:2375", "unix:///var/run/docker.sock"]
}
```

- Create the file `/etc/systemd/system/docker.service.d/docker.conf` with content

```text
[Service]
ExecStart=
ExecStart=/usr/bin/dockerd
```

- Run `sudo systemctl daemon-reload` and `sudo systemctl restart docker`.

- Edit/create the file `/etc/udev/rules.d/99-usb.rules` with content

```text
SUBSYSTEM=="usb", ATTR{idVendor}=="0f18", GROUP="sudo", MODE="0666",SYMLINK+="flicam"
```

- Run `sudo service udev restart` and `sudo udevadm control --reload-rules`.
