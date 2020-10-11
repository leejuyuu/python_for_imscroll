from pathlib import Path
import numpy as np
import pandas as pd
import prompt_toolkit as ptk
from python_for_imscroll import utils
from python_for_imscroll import fitting
from python_for_imscroll import visualization as vis
import gui.image_view as iv

def main():
    session = ptk.PromptSession(reserve_space_for_menu=0)
    path_completer = ptk.completion.PathCompleter(expanduser=True)
    base_path = Path(session.prompt('Enter the directory containing all analyzed data: ',
                                    completer=path_completer)).expanduser().resolve()
    while not base_path.is_dir():
        base_path = Path(session.prompt('Sorry, enter the directory again: ',
                                        completer=path_completer)).expanduser().resolve()
    print('Current directory is', base_path, sep='\n')
    sub_dir_list = [f for f in base_path.iterdir() if f.is_dir()]
    sub_dir_name = [path.name for path in sub_dir_list]
    max_len = max([len(i) for i in sub_dir_name])
    # print([path.name.ljust(max_len+5) for path in sub_dir_list])
    selected_entries = pd.DataFrame(columns=['date', 'filename', 'foldername'])
    while True:
        print('Current selected entries:')
        print(selected_entries)
        print()
        choice = session.prompt(
        """a) add
s) start analysis
q) quit

a/s/q>""")
        if choice == 'q':
            return
        if choice == 's':
            break
        if choice != 'a':
            continue
        dir_name = session.prompt('Enter a date string: ', completer=None)
        print()
        if dir_name not in sub_dir_name:
            continue
        try:
            df = utils.read_excel(base_path / dir_name / (dir_name + 'parameterFile.xlsx'), sheet_name='all')
        except:
            continue
        print(df[['filename', 'foldername']])
        while True:
            str_number_list = session.prompt('Enter files to add to processing list: ')
            try:
                adding_entries_idx = [int(i) for i in str_number_list.strip(', []').split(',')]
                break
            except:
                continue
        adding_entries = df.loc[adding_entries_idx, ['filename', 'foldername']]
        adding_entries['date'] = dir_name
        selected_entries = pd.concat([selected_entries, adding_entries], ignore_index=True)


    for row in selected_entries.index:
        try:
            date = selected_entries.loc[row, 'date']
            filestr = selected_entries.loc[row, 'filename']
            on_file = np.load(base_path / date / (date + 'imscroll') / (filestr + '_first_on.npz'))
            off_file = np.load(base_path / date / (date + 'imscroll') / (filestr + '_off.npz'))
        except FileNotFoundError:
            print(f'{date} lane {filestr} file not found')
            continue
        selected_entries.loc[row, 'k_on'] = on_file['param'][:2].max()
        selected_entries.loc[row, 'k_off'] = off_file['param'].item()

    print(selected_entries)
    path = iv.save_file_path_dialog().with_suffix('.xlsx')
    selected_entries.to_excel(path, engine='openpyxl')



if __name__ == '__main__':
    main()
