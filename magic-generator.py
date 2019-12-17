import math

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


def generate_magic_number(message: str):
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
    magic_number: int = 0
    for line in message_lines:
        for character in line:
            magic_number += char_lookup[character]
            magic_number *= q
        # ensure each line encoded is fixed to x length
        for space in range(x - len(line)):
            magic_number += char_lookup[' ']
            magic_number *= q
        magic_number *= r

    symbol_lookup: dict(int, chr) = {i:char for char,i in char_lookup.items()}
    print(f"y: {y}, x: {x}, q: {q}, r: {r}, symbol_lookup: {symbol_lookup}")
    print("Magic Number: %s" % magic_number)

    # decode
    decoded_string = ""
    for row in range(y):
        magic_number //= r
        line = ""
        for column in range(x):
            magic_number = magic_number // q
            symbol = int(magic_number % q)
            line = symbol_lookup[symbol] + line
        decoded_string = line + "\n" + decoded_string
    decoded_string[:-1]

    print(decoded_string)


if __name__ == "__main__":
    message = """this is a test message
Which I will decoDE! HAHAH YEAH"""
    generate_magic_number(message)
    #print("Magic Number: %s" % generate_magic_number(message))