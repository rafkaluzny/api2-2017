import math

def hello():
    print("Hello, World!")


def tictactoe():

    row = '|'.join(['  '] * 3)      
    div = '\n{}\n'.format('-' * 8)
    print(row, row, row, sep=div)


def tictactoe_zen():
    s = """\
  |  |
--+--+--
  |  |
--+--+--
  |  |  \
"""
    print(s)



def fizzbuzz(n):

    count = 0
    for i in range(n):
        if i % 3 == 0 or i % 5 == 0:
            count += i
    return count


def collatz_len(n):

    length = 1
    while n > 1:
        if n % 2 == 0:
            n /= 2
        else:
            n = 3 * n + 1
        length += 1 
    return length

def converter():
    print("Convert")
    while True:
        try:
            fahr = float(input("Temperature F? "))
        except KeyboardInterrupt:
            print("\nExiting converter...")
            break
        except ValueError as exc:
            print(exc)
        else:
            cels = (fahr - 32) * 5 / 9
            print("It is {} degrees Celsius".format(cels))



def gcd(a, b):
    while b:
        a, b = b, a % b
    return a


def flip_dict(d):
    out = {}
    for key, value in d.items():
        if value not in out:
            out[value] = []
        out[value].append(key)
    return out

def comprehension_read():
    print([x for x in [1, 2, 3, 4]])
    print([n - 2 for n in range(10)])
    print([k % 10 for k in range(41) if k % 3 == 0])
    print([s.lower() for s in ['PythOn', 'iS', 'cOoL'] if s[0] < s[-1]])

    arr = [[3,2,1], ['a','b','c'], [('do',), ['re'], 'mi']]
    print([el.append(el[0] * 4) for el in arr]) 
    print(arr)

    print([letter for letter in "pYthON" if letter.isupper()])
    print({len(w) for w in ["its", "the", "remix", "to", "ignition"]})


def comprehension_write():

    arr = [0, 1, 2, 3]
    print([2 * num + 1 for num in arr])  

    arr = [3, 5, 9, 8]
    print([num % 3 == 0 for num in arr]) 

    arr = ["TA_sam", "TA_guido", "student_poohbear", "student_htiek"]
    print([name[3:] for name in arr if name.startswith('TA_')])  

    arr = ['apple', 'orange', 'pear']
    print([fruit[0].upper() for fruit in arr]) 
    print([fruit for fruit in arr if 'p' in fruit]) 
    print([(fruit, len(fruit)) for fruit in arr])
    print({fruit:len(fruit) for fruit in arr})



def generate_pascal_row(row):

    if not row: return [1]
    return [left + right for left, right in zip([0] + row, row + [0])]


def is_triangle_number(num):

    discrim = 8 * num + 1
    base = int(math.sqrt(discrim))
    return base * base == discrim


#hello();
#tictactoe();
#print(fizzbuzz(41));
#print(collatz_len(13));
#converter()
#print(gcd(10,25));
#print(flip_dict({"CA": "US", "NY": "US", "ON": "CA"}));
#comprehension_read();
#comprehension_write();
#print(generate_pascal_row([1,2,1]));
print(is_triangle_number(1));

