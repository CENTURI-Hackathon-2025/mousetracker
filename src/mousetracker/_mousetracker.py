class MouseTracker:
    def __init__(self, *args):
        print("Hello from MouseTracker")
        self.arg = args[0]


def run_app():
    my_obj = MouseTracker("Hi!")
    print(f"{my_obj.arg = }")
