import pandas as pd




def GetExperimentsToProcess(mice, dates, protocol, experimental_record):

    df_mice = []
    df_dates = []
    df_protocol = []
    df_fiber_side = []
    for i, mouse in enumerate(mice):
        if protocol == '':
            df_mice.append(mouse)
            df_dates.append(dates[i])
            df_fiber_side.append(experimental_record[(experimental_record['mouse'] == mouse) & (experimental_record['date'] == dates[i])]['fiber_side'].values[0])
            protocol1 = experimental_record[(experimental_record['mouse'] == mouse) & (experimental_record['date'] == dates[i])]['protocol1'].values[0]
            df_protocol.append(protocol1)
            protocol2 = experimental_record[(experimental_record['mouse'] == mouse) & (experimental_record['date'] == dates[i])]['protocol2'].values[0]
            if not pd.isna(protocol2):
                df_mice.append(mouse)
                df_dates.append(dates[i])
                df_fiber_side.append(experimental_record[(experimental_record['mouse'] == mouse) & (
                            experimental_record['date'] == dates[i])]['fiber_side'].values[0])
                df_protocol.append(protocol2)
        else:
            df_mice.append(mouse)
            df_dates.append(dates[i])
            df_fiber_side.append(experimental_record[(experimental_record['mouse'] == mouse) & (experimental_record['date'] == dates[i])]['fiber_side'].values[0])
            df_protocol.append(protocol)
    experiments_to_process = pd.DataFrame({'mouse': df_mice, 'date': df_dates, 'fiber_side': df_fiber_side, 'protocol': df_protocol})
    return experiments_to_process



def preprocessData(experiments_to_process):

    for mouse in experiments_to_process['mouse']:
        print(mouse)
