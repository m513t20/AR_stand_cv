from SnakeLogic import start_snake

def main():
    data = """{ "matrix": [{"id": 1, "cords": [0, 0], "turn": 2}, {"id": 2, "cords": [2, 2], "turn": 0}, {"id": 3, "cords": [1, 1], "turn": 2}, {"id": 2, "cords": [1, 2], "turn": 1}, {"id": 1, "cords": [2, 0], "turn": 2}] }"""

    start_snake(data)

if __name__ == "__main__":
    main()