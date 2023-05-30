from queue import Queue
import random, threading, time
import cv2

# 生产者类
class Producer(threading.Thread):
    def __init__(self, name, queue):
        threading.Thread.__init__(self, name=name)
        self.queue = queue

    def run(self):
        for i in range(1, 3):
            print("{} is producing {} to the queue!".format(self.getName(), i))
            img = cv2.imread('ins/shu' + str(i) + '.png')
            imgid = [i, img]
            self.queue.put(imgid)
        print("%s finished!" % self.getName())


# 消费者类
class Consumer(threading.Thread):
    def __init__(self, name, queue):
        threading.Thread.__init__(self, name=name)
        self.queue = queue

    def run(self):
        for i in range(1, 3):
            val = self.queue.get()
            cv2.imshow('img', val[1])
            cv2.waitKey(1000)
            print("{} is consuming {} in the queue.".format(self.getName(), val[0]))
        print("%s finished!" % self.getName())


def main():
    queue = Queue()
    producer = Producer('Producer', queue)
    consumer = Consumer('Consumer', queue)

    producer.start()
    consumer.start()

    producer.join()
    consumer.join()
    print('All threads finished!')


if __name__ == '__main__':
    main()