import math
from typing import List, Dict
import random
import time

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

# pylint: disable=all
decode_magic_number_min = lambda z: (lambda s: lambda n: (a:= s['e'](n),s.update({'g':a[0]}),b := s['e'](a[1]),u := s['e'](b[0]),f := {i:chr(c) for i, c in zip(u[0::2], u[1::2])},[s.update({'g':s['g'] // b[4]}) or s.update({'d': "".join([f[(s.update({'g':s['g'] // b[3]}) or s.get('g')) % b[3]] for p in range(b[1])][::-1]) + f"\n{s['d']}"}) for w in range(b[2])][0:0] or s['d'][:-1],s['d']))({'d': "", 'e': lambda a: [int(int(i, 8)) for i in str(a).split("8")[1:]]})(z)[-1]

def animate_string_explosion(message: str) -> None:
    # constants
    gets_infected = lambda displacement: random.uniform(0, 1) < (0.01 if displacement[1] != 0 else 0.9)
    is_infected = lambda pixel: pixel['life_till_set'] is not None
    random_infected_pixel = lambda: {'char': random.choice("*#"), 'life_till_set': random.uniform(2, 15), 'life_till_spread': random.uniform(0.2, 1)}
    neighbour_displacements = [[0, 1], [0, -1], [1, 0], [-1, 0]]
    clamp = lambda n, smallest, largest: max(smallest, min(n, largest))
    time_scale = 100

    # calculate max line length of message and line height
    message_lines: list(str) = message.splitlines()
    message_y: int = len(message_lines)
    message_x: int = max([len(line) for line in message_lines])

    # generate state map
    state = [[{'char': ' ', 'life_till_set': None, 'life_till_spread': None} for y in range(message_y)] for x in range(message_x)]
    
    # start explosion
    medium_x = message_x // 2
    medium_y = message_y // 2
    state[random.randrange(medium_x - medium_x // 2, medium_x + medium_x // 2)][random.randrange(medium_y - medium_y // 2, medium_y + medium_y // 2)] = random_infected_pixel()

    # enable fancy escape sequences
    __import__("os").system("")

    # animation loop
    last_time = time.time()
    while True:
        # calculate delta time
        current_time = time.time()
        delta_time = (current_time - last_time) * time_scale
        last_time = current_time

        # update life values
        for y in range(message_y):
            for x in range(message_x):
                pixel = state[x][y]
                if pixel['life_till_set'] is not None:
                    pixel['life_till_set'] -= delta_time
                if pixel['life_till_spread'] is not None:
                    pixel['life_till_spread'] -= delta_time
                state[x][y] = pixel

        # animate step
        all_pixels_set = True
        for y in range(message_y):
            for x in range(message_x):
                pixel = state[x][y]

                if is_infected(pixel):
                    if pixel['life_till_set'] < 0:
                        # infection at final form
                        pixel['char'] = message_lines[y][x]
                    else:
                        all_pixels_set = False
                    
                    if pixel['life_till_spread'] < 0:
                        # spread to neighbours
                        random.shuffle(neighbour_displacements)
                        for displacement in neighbour_displacements:
                            neighbour_x = clamp(x + displacement[0], 0, message_x - 1)
                            neighbour_y = clamp(y + displacement[1], 0, message_y - 1)
                            neighbour = state[neighbour_x][neighbour_y]

                            if (is_infected(neighbour)): continue

                            if (gets_infected(displacement)):
                                neighbour = random_infected_pixel()

                                # update state 
                                state[neighbour_x][neighbour_y] = neighbour

                else:
                    all_pixels_set = False

                # update state
                state[x][y] = pixel
        
        # convert state to string 
        state_string = ""
        for y in range(message_y):
            for x in range(message_x):
                state_string += state[x][y]['char']
            state_string += '\n'
        state_string[:-1]

        # print state
        print(state_string, end='')
        
        if all_pixels_set:
            break

        # restore cursor position
        print(f"\033[{message_y + 1}A")


def animate_string_explosion_min(message):
    # variables
    time_scale = 100
    debis = "*#"
    life_till_set_min = 2
    life_till_set_max = 15
    life_till_spread_min = 0.2
    life_till_spread_max = 1
    spread_chance_x = 0.9
    spread_chance_y = 0.01

    random_infected_pixel = lambda: {'char': random.choice(debis), 'life_till_set': random.uniform(life_till_set_min, life_till_set_max), 'life_till_spread': random.uniform(life_till_spread_min, life_till_spread_max)}
    neighbour_displacements = [[0, 1], [0, -1], [1, 0], [-1, 0]]
    clamp = lambda n, smallest, largest: max(smallest, min(n, largest))


    # calculate max line length of message and line height
    message_lines: list(str) = message.splitlines()
    message_y: int = len(message_lines)
    message_x: int = max([len(line) for line in message_lines])

    # generate state map
    state = [[{'char': ' ', 'life_till_set': None, 'life_till_spread': None} for y in range(message_y)] for x in range(message_x)]
    
    # start explosion
    medium_x = message_x // 2
    medium_y = message_y // 2
    state[random.randrange(medium_x - medium_x // 2, medium_x + medium_x // 2)][random.randrange(medium_y - medium_y // 2, medium_y + medium_y // 2)] = random_infected_pixel()

    # enable fancy escape sequences
    __import__("os").system("")

    # animation loop
    last_time = time.time()
    while True:
        # calculate delta time
        current_time = time.time()
        delta_time = (current_time - last_time) * time_scale
        last_time = current_time

        # update life values
        for y in range(message_y):
            for x in range(message_x):
                pixel = state[x][y]
                if pixel['life_till_set'] is not None:
                    pixel['life_till_set'] -= delta_time
                if pixel['life_till_spread'] is not None:
                    pixel['life_till_spread'] -= delta_time
                state[x][y] = pixel

        # animate step
        all_pixels_set = True
        for y in range(message_y):
            for x in range(message_x):
                pixel = state[x][y]

                if pixel['life_till_set'] is not None:
                    if pixel['life_till_set'] < 0:
                        # infection at final form
                        pixel['char'] = message_lines[y][x]
                    else:
                        all_pixels_set = False
                    
                    if pixel['life_till_spread'] < 0:
                        # spread to neighbours
                        random.shuffle(neighbour_displacements)
                        for displacement in neighbour_displacements:
                            neighbour_x = clamp(x + displacement[0], 0, message_x - 1)
                            neighbour_y = clamp(y + displacement[1], 0, message_y - 1)
                            neighbour = state[neighbour_x][neighbour_y]

                            if (neighbour['life_till_set'] is not None): continue

                            if (random.uniform(0, 1) < (spread_chance_y if displacement[1] != 0 else spread_chance_x)):
                                neighbour = random_infected_pixel()

                                # update state 
                                state[neighbour_x][neighbour_y] = neighbour

                else:
                    all_pixels_set = False

                # update state
                state[x][y] = pixel
        
        # convert state to string 
        state_string = ""
        for y in range(message_y):
            for x in range(message_x):
                state_string += state[x][y]['char']
            state_string += '\n'
        state_string[:-1]

        # print state
        print(state_string, end='')
        
        if all_pixels_set:
            break

        # restore cursor position
        print(f"\033[{message_y + 1}A")

if __name__ == "__main__":
    message = r"""_____/\\\\\\\\\\_______/\\\____________/\\\\____________/\\\\_____/\\\\\\\\\_____/\\\________/\\\_        
 ___/\\\///////\\\__/\\\\\\\___________\/\\\\\\________/\\\\\\___/\\\\\\\\\\\\\__\///\\\____/\\\/__       
  __\///______/\\\__\/////\\\___________\/\\\//\\\____/\\\//\\\__/\\\/////////\\\___\///\\\/\\\/____      
   _________/\\\//_______\/\\\___________\/\\\\///\\\/\\\/_\/\\\_\/\\\_______\/\\\_____\///\\\/______     
    ________\////\\\______\/\\\___________\/\\\__\///\\\/___\/\\\_\/\\\\\\\\\\\\\\\_______\/\\\_______    
     ___________\//\\\_____\/\\\___________\/\\\____\///_____\/\\\_\/\\\/////////\\\_______\/\\\_______   
      __/\\\______/\\\______\/\\\___________\/\\\_____________\/\\\_\/\\\_______\/\\\_______\/\\\_______  
       _\///\\\\\\\\\/_______\/\\\___________\/\\\_____________\/\\\_\/\\\_______\/\\\_______\/\\\_______ 
        ___\/////////_________\///____________\///______________\///__\///________\///________\///________
"""
    magic_number = generate_magic_number(message)
    animate_string_explosion(decode_magic_number_min(magic_number))