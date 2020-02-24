    queue = Manager().Queue(2)

    for i in range(3):
        try:
            queue.put(1)
        except Exception as e:
            print(e)
