import Game
def main():
    wp= input('White is played by:\n'
              '0.Bot\n'
              '1.Human\n')
    bp= input('Black is played by:\n'
              '0.Bot\n'
              '1.Human\n')
    return wp,bp

if __name__=='__main__':
    wp,bp=main()
    Game.main(wp,bp,1)