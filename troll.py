import random
import string


def generate_random_stream():
    target = "ap"
    total_characters = 0
    for i in range(10000):
        stream = ""
        while target not in stream:
            # Generate a random character
            char = random.choice(string.ascii_lowercase)

            # Add the character to the stream
            stream += char
            total_characters += 1

            # If the stream becomes too long, truncate it
            if len(stream) > len(target):
                stream = stream[-len(target):]

            # Print the current character
            print(char, end="")

    print("\n")
    print(total_characters/10000)


generate_random_stream()