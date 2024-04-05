import os
import pandas as pd
import math


def convert_to_score(evaluation):

    if evaluation.startswith('#+'):
        return 1
    elif evaluation.startswith('#-'):
        return 0
    else:
        try:
            evaluation=int(evaluation)/100
            if evaluation > 6:
                score = 1
            elif evaluation < -6:
                score = 0
            else:
                score = 1 / (1 + math.exp(-0.8*evaluation))
            return round(score, 3)
        except ValueError:
            return None


def data_preprocess(input_file, output_file):
    df = pd.read_csv(input_file)
    # Apply the conversion function to the 'Evaluation' column
    df['Score'] = df['Evaluation'].apply(convert_to_score)
    df = df.drop(columns=['Evaluation'])
    # Save the modified DataFrame to a new CSV file
    df.to_csv(output_file, index=False)
    print(f"Preprocessed data saved to {output_file}")


if __name__ == '__main__':

    directory_path = '/Users/test/Documents/GitHub/ChessBot/ML/Data/rawData'

    output_dir = '/Users/test/Documents/GitHub/ChessBot/ML/Data/processedData'

    for i in range(130):
        file_name = f'miniData_{str(i+1)}.csv'
        file_path = os.path.join(directory_path, file_name)

        output_name = f'pData_{str(i+1)}.csv'
        output_path = os.path.join(output_dir, output_name)

        data_preprocess(file_path, output_path)
