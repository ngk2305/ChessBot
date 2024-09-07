import importlib.util
from pathlib import Path

def Load_bot(color):

    if color == 'w':
        color = 'White'
    else:
        color = 'Black'
    choice = input(f'{color} is played by:\n'
                   '0.Human\n'
                   '1.SimpleEngine\n'
                   '2.SimpleNeuralEngine\n'
                   '3.SimpleNeuralEngineV2\n'
                   '4.NEv3\n'
                   '5.NEv4\n'
                   '6.NEv5\n'
                   '7.RollOutV1\n'
                   '8.RollOutV2\n'
                   )


    match choice:
        case '0':
            return 1,0
        case '1':
            folder_name = 'Engines/SimpleEngine'
        case '2':
            folder_name = 'Engines/SimpleNeuralEngine'
        case '3':
            folder_name = 'Engines/NeuralEngineV2'
        case '4':
            folder_name = 'Engines/NeuralEngineV3'
        case '5':
            folder_name = 'Engines/NeuralEngineV4'
        case '6':
            folder_name = 'Engines/NeuralEngineV5'
        case '7':
            folder_name = 'Engines/RollOutV1'
        case '8':
            folder_name = 'Engines/RollOutV2'

    bot_path = Path(folder_name) / 'bot.py'

    # Import the bot module 
    module_name = folder_name  # You can adjust this to suit your needs
    spec = importlib.util.spec_from_file_location(module_name, bot_path)
    bot_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bot_module)

    new_bot = bot_module.Agent()

    return new_bot,1
if __name__ == '__main__':
    Load_bot('w')