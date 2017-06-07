import random      
import functools    
import operator    
import itertools   
import collections 
import inspect      

## Functional Tools
def test_map():
    map(int, ['12', '-2', '0'])
    map(len, ['hello', 'world'])
    map(lambda s: s[::-1], ['hello', 'world'])
    map(lambda n: (n, n ** 2, n ** 3), range(2, 6))
    map(lambda l, r: l * r, zip(range(2, 5), range(3, 9, 2)))

def filtertest():
    filter(lambda x: int(x) >= 0, ['12', '-2', '0'])
    filter(lambda x: x == 'world', ['hello', 'world'])
    filter(lambda x: x[0] == 'S', ['Stanford', 'Cal', 'UCLA'])
    filter(lambda n: n % 3 == 0 or n % 5 == 0, range(20))

## Useful Modules
def gcd(a, b):
    while b != 0:
        a, b = b, a % b
    return a

def lcm(*args):
    return functools.reduce(lambda x, y: x * y / gcd(x, y), args)

def fact(n):
    return functools.reduce(operator.mul, range(n))

def testfact():
    fact(3)
    fact(7)

## Custom Comparison
def testcomparisons():
    words = ['pear', 'cabbage', 'apple', 'bananas']
    min(words) 
    words.sort(key=lambda s: s[-1])
    words

    max(words, key=len)
    x= min(words, key=lambda s: s[1::2]) 
    print(x)

def highest_alphanumeric_score():
    def alpha_score(upper_letters):
        return sum(map(lambda l: 1 + ord(l) - ord('A'), upper_letters))

    def two_best(words):
        words.sort(key=lambda word: alpha_score(filter(str.isupper, word)), reverse=True)
        return words[:2]

    print(two_best(['hEllO', 'wOrLD', 'i', 'aM', 'PyThOn']))


## Iterators
def iterator_consumption():
    it = iter(range(100))
    67 in it

    next(it)
    37 in it 
    next(it) 

def test_itertools():
    for el in itertools.permutations('XKCD', 2):
        print(el, end=', ')

    itertools.starmap(operator.mul, itertools.zip_longest([3,5,7],[2,3], fillvalue=1))

## Linear Algebra
def dot_product(u, v):
    assert len(u) == len(v)
    return sum(itertools.starmap(operator.mul, zip(u, v)))

def transpose(m):
    return tuple(zip(*m))

def transpose_lazy(m):
    return zip(*m)

def matmul(m1, m2):
    return tuple(map(lambda row: tuple(dot_product(row, col) for col in transpose(m2)), m1))

def matmul_lazy(m1, m2):
    return map(lambda row: (dot_product(row, col) for col in transpose(m2)), m1)


## Generators
def generate_triangles():
    n = 0
    total = 0
    while True:
        total += n
        n += 1
        yield total

def triangles_under(n):
    for triangle in generate_triangles():
        if triangle >= n:
            break
        print(triangle)

def make_divisibility_test(n):
    return lambda m: m % n == 0


def generate_composites():
    tests = []
    i = 2
    while True:
        if not any(map(lambda test: test(i), tests)):
            tests.append(make_divisibility_test(i))

        else:
            yield i
        i += 1

def nth_composite(n):

    g = generate_composites()
    for i in range(n - 1):
        next(g)
    return next(g)

## Decorators
def bind_args(function, *args, **kwargs):
    sig = inspect.Signature.from_function(function)
    ba = sig.bind(*args, **kwargs)
    return ba.arguments


def print_args(function):
    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        bound_arguments = bind_args(function, *args, **kwargs)
        print("{name}({call})".format(
            name=function.__name__,
            call=', '.join("{}={}".format(arg, val) for arg, val in bound_arguments.items())
        ))
        retval = function(*args, **kwargs)
        if retval is not None:
            print("(return) {!r}".format(retval))
        return retval
    return wrapper

def cache(function):
    function._cache = {}
    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        key = (args, tuple(kwargs.items()))
        if key in function._cache:
            return function._cache[key]
        retval = function(*args, **kwargs)
        function._cache[key] = retval
        return retval
    return wrapper

@cache
def fib(n):
    return fib(n-1) + fib(n-2) if n > 2 else 1

def cache_challenge(max_size=None, eviction_policy='LRU'):
    assert eviction_policy in ['LRU', 'MRU', 'random']
    def decorator(function):
        function._cache = collections.OrderedDict()
        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            key = (args, tuple(kwargs.items()))
            if key in function._cache:
                function._cache.move_to_end(key)
                return function._cache[key]
            retval = function(*args, **kwargs)
            if max_size and len(function._cache) == max_size:
                if eviction_policy == 'LRU':
                    function._cache.popitem(last=False)
                elif eviction_policy == 'MRU':
                    function._cache.popitem(last=True)
                else:
                    randkey = random.choice(list(function._cache.keys()))
                    function._cache.pop(randkey)
            # Now that we know there's space, insert the element
            function._cache[key] = retval
            return retval
        return wrapper
    return decorator

@cache_challenge(max_size=16, eviction_policy='LRU')
def fib(n):
    return fib(n-1) + fib(n-2) if n > 2 else 1


def enforce_types(function):
    expected = function.__annotations__
    if not expected:
        return function
    assert(all(map(lambda exp: type(exp) == type, expected.values())))
    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        bound_arguments = bind_args(function, *args, **kwargs)
        for arg, val in bound_arguments.items():
            if arg in expected and not isinstance(val, expected[arg]):
                print("(Bad Argument Type!) argument '{arg}={val}': expected {exp}, received {r}".format(
                    arg=arg,
                    val=val,
                    exp=expected[arg],
                    r=type(val)
                ))

        retval = function(*args, **kwargs)
        if 'return' in expected and not isinstance(retval, expected['return']):
            print("(Bad Return Value!) return '{ret}': expected {exp}, received {r}".format(
                ret=retval,
                exp=expected['return'],
                r=type(retval)
            ))
        return retval
    return wrapper

@enforce_types


def enforce_types_challenge(severity=1):
    assert severity in [0, 1, 2]
    if severity == 0:
        return lambda function: function

    def message(msg):
        if severity == 1:
            print(msg)
        else:
            raise TypeError(msg)

    def decorator(function):
        expected = function.__annotations__
        if not expected:
            return function
        assert(all(map(lambda exp: type(exp) == type, expected.values())))

        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            bound_arguments = bind_args(function, *args, **kwargs)
            for arg, val in bound_arguments.items():
                if arg in expected and not isinstance(val, expected[arg]):
                    msg("(Bad Argument Type!) argument '{arg}={val}': expected {exp}, received {r}".format(
                        arg=arg,
                        val=val,
                        exp=expected[arg],
                        r=type(val)
                    ))

            retval = function(*args, **kwargs)

            if 'return' in expected and not isinstance(retval, expected['return']):
                msg("(Bad Return Value!) return '{ret}': expected {exp}, received {r}".format(
                    ret=retval,
                    exp=expected['return'],
                    r=type(retval)
                ))
            return retval
        return wrapper
    return decorator


@enforce_types_challenge(severity=2)
def bar(a: list, b: str) -> int:
    return 0

@enforce_types_challenge() 
def baz(a: bool, b: str) -> str:
    return ''

if __name__ == '__main__':
    """Runs each of the lab solution functions and prints the docstring"""
