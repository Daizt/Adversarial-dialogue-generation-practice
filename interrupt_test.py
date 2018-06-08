import time

def train():
    print("training...")
    time.sleep(5)

def update_params():
    print("Params updated.")

def test():
    interrupt_flag = False
    while True:
        if interrupt_flag:
            print("Stop training.")
            break
        else:

            try:
                train()
                update_params()
                # print("you can stop training.")
                # time.sleep(5)
                # print("Continue training...")
            except KeyboardInterrupt:
                interrupt_flag = True

test()
