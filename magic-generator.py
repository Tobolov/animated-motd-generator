import math
from typing import List, Dict

# https://www.geeksforgeeks.org/smallest-special-prime-which-is-greater-than-or-equal-to-a-given-number/
def check_prime(sieve, num: int): 
    while (num): 
        if (sieve[num] == False): 
            return False
        num = int(num / 10) 
    return True
def prime_larger_than(n: int):
    sieve = [True for i in range(n * 10 + 1)] 
    sieve[0] = False
    sieve[1] = False
    for i in range(2, n * 10 + 1): 
        if (sieve[i]): 
            for j in range(i * i, n * 10 + 1, i): 
                sieve[j] = False
    while (True): 
        if (check_prime(sieve, n)): 
            return n
            break
        else: 
            n += 1


def sequence_to_magic_number(sequence: List[int]) -> int:
    q = prime_larger_than(max(sequence) + 1)
    magic_number: int = 0
    for symbol in sequence:
        magic_number += symbol
        magic_number *= q

    return int(f"{str(oct(q))[2:]}8{str(magic_number)}")

def magic_number_to_sequence(number: int) -> List[int]:
    magic_string = str(number)
    sentinal_index = magic_string.find("8")
    q: int = int(int(magic_string[0:sentinal_index],8))
    magic_number: int = int(magic_string[sentinal_index + 1:])

    sequence: list(int) = []
    while magic_number > 0:
        magic_number //= q
        sequence.append(magic_number % q)
    return sequence[::-1]

def extract_sequence_on_8(number: int) -> List[int]:
    return [int(int(item, 8)) for item in str(number).split("8")[1:]]

def sequence_join_on_8(sequence: List[int]) -> int:
    return int(f"8{'8'.join([str(oct(item))[2:] for item in sequence])}")

def generate_magic_number(message: str) -> int:
    # generate map of number -> character
    char_lookup: dict(chr, int) = {char:i for (i, char) in enumerate(set(list(message)), start=0)}
    #char_lookup[' '] = char_lookup.get(' ', len(char_lookup))

    # calculate max line length of message and line height
    message_lines: list(str) = message.splitlines()
    y: int = len(message_lines)
    x: int = max([len(line) for line in message_lines])

    # generate prime larger than len(char_lookup) and x
    q: int = prime_larger_than(len(char_lookup))
    r: int = prime_larger_than(max(x+1,q+1))

    # generate the magic number
    message_number: int = 0
    for line in message_lines:
        for character in line:
            message_number += char_lookup[character]
            message_number *= q
        # ensure each line encoded is fixed to x length
        for space in range(x - len(line)):
            message_number += char_lookup[' ']
            message_number *= q
        message_number *= r

    symbol_lookup: dict(int, chr) = {i:char for char,i in char_lookup.items()}

    # generate the symbol lookup number
    symbol_lookup_number: int = 0
    sequence = []
    for i, char in symbol_lookup.items():
        sequence.append(i)
        sequence.append(ord(char))
    symbol_lookup_number = sequence_join_on_8(sequence)

    # generate a config number 
    sequence = [symbol_lookup_number, x, y, q, r]
    config_number = sequence_join_on_8(sequence)

    magic_number = sequence_join_on_8([message_number, config_number])
    #print(f"x: {x}, y: {y}, q: {q}, r: {r}, symbol_lookup: {symbol_lookup}, message_number: {message_number}")
    return magic_number

def decode_magic_number(number:int):
    message_number, config_number = extract_sequence_on_8(number)
    symbol_lookup_number, x, y, q, r = extract_sequence_on_8(config_number)
    symbol_lookup_sequence = extract_sequence_on_8(symbol_lookup_number)
    symbol_lookup: Dict[int, chr] = {}
    for i, char in zip(symbol_lookup_sequence[0::2], symbol_lookup_sequence[1::2]):
        symbol_lookup[i] = chr(char)

    decoded_string = ""
    for row in range(y):
        message_number //= r
        line = ""
        for column in range(x):
            message_number //= q
            symbol = int(message_number % q)
            line = symbol_lookup[symbol] + line
        decoded_string = line + "\n" + decoded_string
    decoded_string[:-1]

    return decoded_string

decode_magic_number_min = lambda z: (lambda s: lambda n: (a:= s['e'](n),s.update({'g':a[0]}),b := s['e'](a[1]),u := s['e'](b[0]),f := {i:chr(c) for i, c in zip(u[0::2], u[1::2])},[s.update({'g':s['g'] // b[4]}) or s.update({'d': "".join([f[(s.update({'g':s['g'] // b[3]}) or s.get('g')) % b[3]] for p in range(b[1])][::-1]) + f"\n{s['d']}"}) for w in range(b[2])][0:0] or s['d'][:-1],s['d']))({'d': "", 'e': lambda a: [int(int(i, 8)) for i in str(a).split("8")[1:]]})(z)[-1]


if __name__ == "__main__":
    message = """this is a test message
Which I will decoDE! HAHAH YEAH
Another line for good measure"""
    magic_number = generate_magic_number(message)
    print(decode_magic_number_min(magic_number))
