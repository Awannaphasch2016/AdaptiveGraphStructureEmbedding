class First(object):
    def __init__(self):
        print("first")

# class Second(First):
class Second():
    def __init__(self):
        print("second")

# class Third(First):
class Third():
    def __init__(self):
        print("third")

# class Fourth(Second, Third):
class Fourth(Third, Second):
    def __init__(self):
        super(Fourth, self).__init__()
        print("that's it")
if __name__ == '__main__':
    f= Fourth()
    print(Fourth.mro())