mkdir -p data
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1e-OFd6kMtz1x1zO0RQh5hVYfD_yfdJii' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1e-OFd6kMtz1x1zO0RQh5hVYfD_yfdJii" -O data/replica.zip && rm -rf /tmp/cookies.txt
unzip data/replica.zip -d data/
rm data/replica.zip

