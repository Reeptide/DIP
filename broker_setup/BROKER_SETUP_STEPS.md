# Broker Setup Guide  
Role: Kafka Broker + Redis Coordinator  
Project: Distributed Image Processing Pipeline with Kafka

# ZeroTier Network Setup
```bash
sudo zerotier-cli join d5e5fb6537fdfba0
sudo zerotier-cli listnetworks

# my zerotier ip used in the project 
10.242.111.145

#start Zookeeper
cd /opt/kafka
bin/zookeeper-server-start.sh config/zookeeper.properties

#change kafka configuration file 
sudo nano /opt/kafka/config/server.properties

listeners=PLAINTEXT://0.0.0.0:9092
advertised.listeners=PLAINTEXT://10.242.111.145:9092
zookeeper.connect=10.242.111.145:2181

#start Kafka broker(keep the terminal running)
cd /opt/kafka
bin/kafka-server-start.sh config/server.properties

#create kafka topics 
cd /opt/kafka

bin/kafka-topics.sh --create --topic tasks --bootstrap-server 10.242.111.145:9092 --partitions 2 --replication-factor 1
bin/kafka-topics.sh --create --topic results --bootstrap-server 10.242.111.145:9092 --partitions 2 --replication-factor 1
bin/kafka-topics.sh --create --topic heartbeats --bootstrap-server 10.242.111.145:9092 --partitions 1 --replication-factor 1

#to verify topics are created
bin/kafka-topics.sh --list --bootstrap-server 10.242.111.145:9092

# Redis cofiguration 
sudo nano /etc/redis/redis.conf

bind 0.0.0.0
protected-mode no
port 6379

#Restart Redis 
sudo systemctl restart redis-server

#test Redis(output should be PONG)
redis-cli -h 10.242.111.145 ping

#Monitor Kafka Traffic (For Testing)
cd /opt/kafka

 # tasks stream
bin/kafka-console-consumer.sh --topic tasks --bootstrap-server 10.242.111.145:9092 --from-beginning

 # results stream
bin/kafka-console-consumer.sh --topic results --bootstrap-server 10.242.111.145:9092 --from-beginning

# worker heartbeat stream
bin/kafka-console-consumer.sh --topic heartbeats --bootstrap-server 10.242.111.145:9092 --from-beginning






