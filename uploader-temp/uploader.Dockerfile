FROM ubuntu:22.04

RUN apt-get update && apt-get install -y openssh-server && mkdir -p /run/sshd

CMD ["sleep", "infinity"]
