#what is this for?
p
import functools

class SwitchedDecorator:
    def __init__(self, enabled_func):
        self._enabled = False
        self._enabled_func = enabled_func
    @property
    def enabled(self):
        return self._enabled

    @enabled.setter
    def enabled(self, new_value):
        if not isinstance(new_value, bool):
            raise ValueError("enabled can only be set to a boolean value")
        self._enabled = new_value

    def __call__(self, target):
        if self._enabled:
            return self._enabled_func(target)
        return target


def deco_func(target):
    """This is the actual decorator function.  It's written just like any other decorator."""
    def g(*args,**kwargs):
        print("your function has been wrapped")
        return target(*args,**kwargs)
    functools.update_wrapper(g, target)
    return g


# This is where we wrap our decorator in the SwitchedDecorator class.
my_decorator = SwitchedDecorator(deco_func)

# Now my_decorator functions just like the deco_func decorator,
# EXCEPT that we can turn it on and off.
my_decorator.enabled=True

@my_decorator
def example1():
    print("example1 function")

# we'll now disable my_decorator.  Any subsequent uses will not
# actually decorate the target function.
my_decorator.enabled=False
@my_decorator
def example2():
    print("example2 function")

example1()
example2()

def json_file(fname):
    def decorator(function):
        signature= inspect.signature(function)
        def wrapper(*args, **kwargs):
            bound_args= signature.bind(*args, **kwargs)
            file_name= fname.format(**bound_args.arguments)
            if os.path.isfile(file_name):
                with open(file_name, 'r') as f:
                    ret = json.load(f)
            else:
                with open(file_name,'w') as f:
                    ret = function(*args, **kwargs)
                    json.dump(ret, f)
            return ret
        return wrapper
    return decorator



