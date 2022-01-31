from DoSomething import DoSomething
import time
import json


class Subscriber(DoSomething):
    def notify(self, topic, msg):
        input_json = json.loads(msg)
        print(topic, input_json['alarm'])


if __name__ == "__main__":
    test = Subscriber("monitoring client")
    test.run()
    test.myMqttClient.mySubscribe("/homework3/ex1/alarm")

    while True:
        time.sleep(1)
